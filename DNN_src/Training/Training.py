import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


CSV_PATH = Path(r"merged.csv")
TARGET_COL = "DeltaSec"

BATCH_SIZE = 256
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-4
VALID_RATIO = 0.2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BEST_PTH_PATH = Path("ft_transformer_best.pth")
BEST_PT_PATH = Path("ft_transformer_best.pt")
META_JSON_PATH = Path("ft_transformer_best_meta.json")

EXCLUDE_COLS = set()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ChunkDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().view(-1, 1)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TabularFTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
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

        self.feature_weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(n_features, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            layer_norm_eps=1e-5,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model, eps=1e-5)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tok = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        x_tok = self.encoder(x_tok)
        x_tok = self.norm(x_tok)
        pooled = x_tok.mean(dim=1)
        out = self.head(pooled)
        return out


@dataclass
class ScalerMeta:
    mean: np.ndarray
    scale: np.ndarray
    feature_cols: list[str]


def fit_standard_scaler(X: np.ndarray, eps: float = 1e-12) -> ScalerMeta:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < eps, 1.0, scale)
    return ScalerMeta(mean=mean, scale=scale, feature_cols=[])


def transform_standard_scaler(X: np.ndarray, meta: ScalerMeta) -> np.ndarray:
    return (X - meta.mean) / meta.scale


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(n, 1)


def train_one_epoch(model, loader, opt, loss_fn, grad_clip: float = 1.0) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(n, 1)


def export_torchscript(model: nn.Module, n_features: int, pt_path: Path):
    model.eval()
    example = torch.zeros(1, n_features, dtype=torch.float32)
    scripted = torch.jit.trace(model.cpu(), example)
    scripted.save(str(pt_path))


def main():
    set_seed(SEED)

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    feature_cols = [c for c in df.columns if c != TARGET_COL and c not in EXCLUDE_COLS]

    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_s = pd.to_numeric(df[TARGET_COL], errors="coerce")

    valid_mask = X_df.notna().all(axis=1) & y_s.notna()
    X_df = X_df[valid_mask].copy()
    y_s = y_s[valid_mask].copy()

    X = X_df.to_numpy(dtype=np.float32)
    y = y_s.to_numpy(dtype=np.float32)

    n = len(X)
    n_valid = int(n * VALID_RATIO)
    n_train = n - n_valid

    X_train, y_train = X[:n_train], y[:n_train]
    X_valid, y_valid = X[n_train:], y[n_train:]

    scaler = fit_standard_scaler(X_train)
    scaler.feature_cols = feature_cols

    X_train_s = transform_standard_scaler(X_train, scaler)
    X_valid_s = transform_standard_scaler(X_valid, scaler)

    train_ds = ChunkDataset(X_train_s, y_train)
    valid_ds = ChunkDataset(X_valid_s, y_valid)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TabularFTTransformer(n_features=X_train_s.shape[1]).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn)
        va_loss = evaluate(model, valid_loader, loss_fn)
        print(f"[{epoch:04d}/{EPOCHS}] train={tr_loss:.6f} valid={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            ckpt = {
                "model_state": model.state_dict(),
                "n_features": int(X_train_s.shape[1]),
                "feature_cols": feature_cols,
                "scaler_mean": scaler.mean.astype(np.float32),
                "scaler_scale": scaler.scale.astype(np.float32),
            }
            torch.save(ckpt, BEST_PTH_PATH)

    ckpt = torch.load(BEST_PTH_PATH, map_location="cpu")
    best_model = TabularFTTransformer(n_features=ckpt["n_features"]).cpu()
    best_model.load_state_dict(ckpt["model_state"])
    best_model.eval()

    export_torchscript(best_model, ckpt["n_features"], BEST_PT_PATH)

    meta = {
        "feature_cols": ckpt["feature_cols"],
        "scaler_mean": ckpt["scaler_mean"].tolist(),
        "scaler_scale": ckpt["scaler_scale"].tolist(),
        "target_col": TARGET_COL,
        "model_type": "TabularFTTransformer",
    }
    META_JSON_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
