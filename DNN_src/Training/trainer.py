import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data_utils import load_and_preprocess, build_dataloaders
from modeling import build_model
from training_config import TrainingConfig, CSV_PATH, TARGET_COL, DEVICE


def train_one_epoch(model, data_loader, optimizer, criterion, device: str):
    model.train()
    train_losses = []

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    return float(np.mean(train_losses))


@torch.no_grad()
def eval_one_epoch(model, data_loader, criterion, device: str):
    model.eval()
    val_losses = []

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = criterion(pred, yb)
        val_losses.append(loss.item())

    return float(np.mean(val_losses))


@torch.no_grad()
def export_torchscript(model: nn.Module, out_pt: str):
    was_training = model.training

    model.eval()
    try:
        scripted = torch.jit.script(model)
        scripted.save(out_pt)
    finally:
        model.train(was_training)


def save_pt_metadata(
    model,
    scaler,
    feature_cols,
    pt_path: Path,
    train_loss: float,
    val_loss: float,
    epoch: int | None = None,
):
    meta_path = pt_path.with_suffix(".json")
    meta = {
        "feature_cols": [str(col) for col in feature_cols],
        "n_features": int(model.n_features),
        "d_model": int(model.d_model),
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
    }
    if epoch is not None:
        meta["epoch"] = int(epoch)

    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return meta_path


def _pt_path_with_losses(base_path: Path, train_loss: float, val_loss: float) -> Path:
    return base_path.with_name(
        f"{base_path.stem}_train{train_loss:.4f}_val{val_loss:.4f}{base_path.suffix}"
    )


def save_best(
    model,
    scaler,
    feature_cols,
    best_model_path: Path,
    best_pt_path: Path,
    train_loss: float,
    val_loss: float,
):
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    best_pt_path.parent.mkdir(parents=True, exist_ok=True)
    best_pt_path = _pt_path_with_losses(best_pt_path, train_loss, val_loss)

    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "feature_cols": feature_cols,
            "d_model": model.d_model,
            "n_features": model.n_features,
        },
        best_model_path,
    )
    export_torchscript(model, str(best_pt_path))
    meta_path = save_pt_metadata(
        model,
        scaler,
        feature_cols,
        best_pt_path,
        train_loss,
        val_loss,
    )
    print(
        f"  -> best model updated ({val_loss:.4f}) saved {best_model_path}, {best_pt_path}, {meta_path}"
    )


def save_checkpoint(
    model,
    scaler,
    feature_cols,
    ckpt_dir: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    val_tag = f"{val_loss:.4f}"
    train_tag = f"{train_loss:.4f}"
    ckpt_path = ckpt_dir / f"ft_transformer_epoch{epoch}_train{train_tag}_val{val_tag}.pth"
    pt_path = ckpt_dir / f"ft_transformer_epoch{epoch}_train{train_tag}_val{val_tag}.pt"

    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "feature_cols": feature_cols,
            "d_model": model.d_model,
            "n_features": model.n_features,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        ckpt_path,
    )

    export_torchscript(model, str(pt_path))
    meta_path = save_pt_metadata(
        model,
        scaler,
        feature_cols,
        pt_path,
        train_loss,
        val_loss,
        epoch=epoch,
    )
    print(f"  -> checkpoint saved (epoch={epoch}): {ckpt_path}, {pt_path}, {meta_path}")


def train(config: TrainingConfig = TrainingConfig(), input_path: Path = CSV_PATH):
    (X_train, y_train), (X_val, y_val), scaler, feature_cols = load_and_preprocess(
        input_path, TARGET_COL, config.valid_ratio, file_glob=config.csv_glob
    )

    train_loader, val_loader = build_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        config.batch_size,
    )

    model = build_model(
        n_features=X_train.shape[1],
        scaler=scaler,
        d_model=128,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
    ).to(DEVICE)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float("inf")

    ckpt_dir = Path("checkpoints")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = eval_one_epoch(model, val_loader, criterion, DEVICE)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best(
                model,
                scaler,
                feature_cols,
                config.best_model_path,
                config.best_pt_path,
                train_loss,
                best_val_loss,
            )

        if epoch % 100 == 0:  # 100 epoch save interval
            save_checkpoint(
                model,
                scaler,
                feature_cols,
                ckpt_dir,
                epoch,
                train_loss,
                val_loss,
            )

    print("Training finished. Best val_loss:", best_val_loss)
