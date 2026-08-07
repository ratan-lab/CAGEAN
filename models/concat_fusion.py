"""
ConcatFusionInstanceNorm: concatenation-fusion ablation of CAGEAN.

Architecture matches CAGEAN exactly except the cross-attention layer is replaced
by a linear projection of the concatenated sequence and epigenomic representations,
with a sequence residual bypass:

    fused = concat_proj(cat([seq_repr, epi_repr], dim=-1)) + seq_repr

This leaves cross-attention as the sole architectural variable distinguishing
this model from CAGEAN, enabling a controlled test of its contribution.

Input:  (B, 5, 1000) — channels 0:4 one-hot DNA, channel 4 epigenomic signal
Output: (B, 1)       — pre-sigmoid logit

~179 K parameters (same stems as CAGEAN; no cross-attention weights).
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


class ConcatFusionInstanceNorm(nn.Module):
    """
    Concatenation-fusion ablation of CAGEAN.

    Identical sequence Transformer stem, instance-normalized epigenomic stem,
    and sinusoidal positional encoding as CAGEAN. Fusion replaces cross-attention
    with linear projection of the concatenated representations plus a sequence
    residual bypass.

    Args:
        d_model (int): Token embedding dimension. Default 64.
        nhead (int): Number of attention heads in the sequence encoder. Default 4.
        num_layers (int): Transformer encoder depth. Default 4.
        dim_feedforward (int): FFN inner dimension. Default 192.
        stem_kernel (int): Strided conv kernel / stride. Default 10.
        dropout (float): Dropout in the Transformer encoder. Default 0.
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
        self.epi_stem = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=stem_kernel, stride=stem_kernel),
            nn.InstanceNorm1d(d_model, affine=True),
            nn.ReLU(),
        )
        # Fusion: project concatenated representations back to d_model
        self.concat_proj = nn.Linear(2 * d_model, d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 5, 1000) — first 4 channels one-hot DNA, last channel epi.
        Returns:
            (B, 1) logit.
        """
        seq = x[:, :4, :]
        epi = x[:, 4:, :]

        seq_tokens = self.seq_stem(seq).permute(0, 2, 1)
        seq_tokens = self.pe(seq_tokens)
        seq_repr   = self.seq_encoder(seq_tokens)           # (B, 100, d)

        epi_repr = self.epi_stem(epi).permute(0, 2, 1)
        epi_repr = self.pe(epi_repr)

        # Concatenate and project, then add sequence residual
        fused = self.concat_proj(torch.cat([seq_repr, epi_repr], dim=-1))
        fused = fused + seq_repr                            # (B, 100, d)
        return self.fc(fused.mean(dim=1))                   # (B, 1)
