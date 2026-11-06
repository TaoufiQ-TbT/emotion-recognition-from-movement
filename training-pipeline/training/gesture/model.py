"""The dual-stream Transformer of stage 4.

One encoder reads the 3D pose sequence and another reads the LMA features. Their outputs are
averaged over time, concatenated, and classified (late fusion).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # A buffer: saved with the model, but never trained.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the encoding of each time step to x, shaped (B, T, d_model)."""
        T = x.size(1)
        return x + self.pe[:, :T, :]


class DualStreamEmotionTransformer(nn.Module):
    """
    Two parallel Transformer encoders for pose and Laban streams, with fusion head.
    """

    def __init__(
        self,
        pose_dim: int,
        laban_dim: int,
        num_emotions: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 500,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pose_proj = nn.Linear(pose_dim, d_model)
        self.laban_proj = nn.Linear(laban_dim, d_model)

        # Has no weights, so both streams can use the same instance.
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # nn.TransformerEncoder deep-copies this layer, so the two streams don't share weights.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )

        self.pose_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.laban_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(2 * d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_emotions),
        )

    def forward(self, pose_seq: torch.Tensor, laban_seq: torch.Tensor, mask=None) -> torch.Tensor:
        """
        pose_seq: (B, T, J, 3)
        laban_seq: (B, T, F)
        mask: (B, T) bool padding mask (True where padded)
        """
        B, T, J, _ = pose_seq.shape

        pose_flat = pose_seq.view(B, T, J * 3)
        pose_emb = self.pos_encoder(self.pose_proj(pose_flat))
        pose_enc = self.pose_encoder(pose_emb, src_key_padding_mask=mask)

        # Average over the real time steps only: padded ones must not count.
        if mask is not None:
            lens = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
            h_pose = (pose_enc * (~mask).unsqueeze(-1)).sum(dim=1) / lens
        else:
            h_pose = pose_enc.mean(dim=1)

        laban_emb = self.pos_encoder(self.laban_proj(laban_seq))
        laban_enc = self.laban_encoder(laban_emb, src_key_padding_mask=mask)

        if mask is not None:
            lens = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
            h_laban = (laban_enc * (~mask).unsqueeze(-1)).sum(dim=1) / lens
        else:
            h_laban = laban_enc.mean(dim=1)

        # Late fusion: each stream is summarized on its own before they meet.
        h = torch.cat([h_pose, h_laban], dim=-1)
        logits = self.fusion_head(h)
        return logits