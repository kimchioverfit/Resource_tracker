import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn


class TabularFTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        scaler_mean: np.ndarray | torch.Tensor,
        scaler_scale: np.ndarray | torch.Tensor,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        eps: float = 1e-12,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model
        self.eps = eps

        mean_t = scaler_mean if torch.is_tensor(scaler_mean) else torch.tensor(scaler_mean, dtype=torch.float32)
        scale_t = scaler_scale if torch.is_tensor(scaler_scale) else torch.tensor(scaler_scale, dtype=torch.float32)
        self.register_buffer("scaler_mean", mean_t.view(1, -1))
        self.register_buffer("scaler_scale", scale_t.view(1, -1))

        self.feature_weight = nn.Parameter(torch.randn(n_features, d_model))
        self.feature_bias = nn.Parameter(torch.zeros(n_features, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = (x - self.scaler_mean) / (self.scaler_scale + self.eps)
        x_ = x.unsqueeze(-1)
        tokens = x_ * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        out = self.head(pooled)
        return out


def build_model(
    n_features: int,
    scaler: StandardScaler,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 4,
    dropout: float = 0.1,
):
    model = TabularFTTransformer(
        n_features=n_features,
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )
    return model
