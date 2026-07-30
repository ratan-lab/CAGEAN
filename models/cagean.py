"""
CAGEAN: Cross-Attention G4 Epigenomic Architecture Network.

Architecture:
  Sequence stem  : Conv1d(4→64, k=10, s=10) → BatchNorm → ReLU
                   → sinusoidal PE → 4-layer Transformer encoder (pre-norm)
  Epigenomic stem: Conv1d(1→64, k=10, s=10) → InstanceNorm1d → ReLU
                   → sinusoidal PE
  Cross-attention: seq=Q, epi=K/V  (100×100 weight matrix, head-averaged)
  Head           : mean-pool → Linear(64→1)

Input:  (B, 5, 1000) — channels 0:4 one-hot DNA, channel 4 epigenomic signal
Output: (B, 1)       — pre-sigmoid logit

~187 K parameters.
"""
import math
import torch
import torch.nn as nn


class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CAGEAN(nn.Module):
    """
    CAGEAN: sequence and epigenomic streams fused via cross-attention.

    The epigenomic stem uses InstanceNorm1d, which normalises each sample by
    its own statistics at inference time. This removes the dependence on
    training-cell running statistics (BatchNorm) that causes cross-cell
    performance collapse.

    Args:
        d_model (int): Token embedding dimension. Default 64.
        nhead (int): Number of attention heads. Default 4.
        num_layers (int): Transformer encoder depth. Default 4.
        dim_feedforward (int): FFN inner dimension. Default 192.
        stem_kernel (int): Strided conv kernel / stride (produces 100 tokens
            from 1000-bp input). Default 10.
        dropout (float): Dropout in Transformer and cross-attention. Default 0.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 192,
        stem_kernel: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Sequence stem — BatchNorm is appropriate; DNA is cell-type invariant.
        self.seq_stem = nn.Sequential(
            nn.Conv1d(4, d_model, kernel_size=stem_kernel, stride=stem_kernel),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        self.pe = _SinusoidalPE(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            norm_first=True,
            batch_first=True,
            dropout=dropout,
        )
        self.seq_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )
        # Epigenomic stem — InstanceNorm normalises per sample, not per dataset.
        self.epi_stem = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=stem_kernel, stride=stem_kernel),
            nn.InstanceNorm1d(d_model, affine=True),
            nn.ReLU(),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc   = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 5, 1000) — first 4 channels one-hot DNA, last channel epi.
        Returns:
            (B, 1) logit.
        """
        seq = x[:, :4, :]   # (B, 4, 1000)
        epi = x[:, 4:, :]   # (B, 1, 1000)

        seq_tokens = self.seq_stem(seq).permute(0, 2, 1)   # (B, 100, d)
        seq_tokens = self.pe(seq_tokens)
        seq_repr   = self.seq_encoder(seq_tokens)           # (B, 100, d)

        epi_repr = self.epi_stem(epi).permute(0, 2, 1)     # (B, 100, d)
        epi_repr = self.pe(epi_repr)

        attn_out, _ = self.cross_attn(self.norm(seq_repr), epi_repr, epi_repr)
        fused = seq_repr + attn_out                         # (B, 100, d)
        return self.fc(fused.mean(dim=1))                   # (B, 1)
