"""
SeqOnlyTransformer: CAGEAN architecture without the epigenomic stem or
cross-attention. Trained from scratch on DNA sequence alone.

Used to isolate the contribution of the Transformer architecture from
epigenomic signal: Δ(arch) = SeqOnlyTransformer − seq-only CNN (epiG4NN).

Input:  (B, 4, 1000) — one-hot DNA, no epigenomic channel
Output: (B, 1)       — pre-sigmoid logit

~155 K parameters (vs ~187 K for CAGEAN; difference = epi stem + cross-attn).
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


class SeqOnlyTransformer(nn.Module):
    """
    Sequence-only Transformer baseline matching CAGEAN's sequence branch.

    Identical stem and encoder to CAGEAN. Cross-attention over the
    epigenomic representation is removed; the forward pass is pure sequence.

    Args: same as CAGEAN (d_model, nhead, num_layers, dim_feedforward,
          stem_kernel, dropout).
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
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4, 1000) — one-hot DNA sequence only.
        Returns:
            (B, 1) logit.
        """
        tokens = self.seq_stem(x).permute(0, 2, 1)   # (B, 100, d)
        tokens = self.pe(tokens)
        repr_  = self.seq_encoder(tokens)              # (B, 100, d)
        return self.fc(repr_.mean(dim=1))              # (B, 1)
