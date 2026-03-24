import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader


class ChunkDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


STATION_DUMMY_PREFIX = "StationId"
STATION_DUMMY_SEP = "__"
CURRENT_METADATA_COLUMNS = {
    "Filename",
    "Site",
    "ChunkIndex",
    "StartTime",
    "EndTime",
    "NumRows",
}
LEGACY_METADATA_COLUMNS = {
    "original_filename",
    "site",
    "chunk_index",
    "chunk_start_time",
    "chunk_end_time",
    "RowCount",
}
COMMON_DROP_COLUMNS = {
    "equipment",
    "date",
    "CPU Cores(Physical)",
    "CPU Threads(Logical)",
    "CPU Base(MHz)",
    "Total RAM(MB)",
    "NTFS MFT Reads/sec",
    "NTFS MFT Writes/sec",
    "status",
    "details",
    "Predicted_DeltaSec",
    "Diff",
    "Avg_DeltaSec",
    "Avg_Predicted_DeltaSec",
    "Avg_Diff",
    "R2",
    "Spike",
}
PREPROCESSED_FILE_PATTERN = re.compile(
    r"^(?P<station>.+?)_\d{4}-\d{2}-\d{2}_site\d+_chunk_(?:avg|chunked)\.csv$"
)


def _infer_station_id(csv_path: Path, base_dir: Path | None = None) -> str | None:
    if base_dir is not None:
        try:
            relative_path = csv_path.relative_to(base_dir)
        except ValueError:
            relative_path = None

        if relative_path is not None and len(relative_path.parts) >= 2:
            return relative_path.parts[0]

    match = PREPROCESSED_FILE_PATTERN.match(csv_path.name)
    if match:
        return match.group("station")

    return None


def _read_training_csv(csv_path: Path, base_dir: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if base_dir is not None and "StationId" not in df.columns:
        station_id = _infer_station_id(csv_path, base_dir=base_dir)
        if station_id:
            df.insert(0, "StationId", station_id)

    return df


def _load_training_dataframe(input_path: Path, file_glob: str) -> pd.DataFrame:
    if input_path.is_file():
        print(f"[INFO] Loading training csv: {input_path}")
        return _read_training_csv(input_path)

    if input_path.is_dir():
        csv_paths = sorted(p for p in input_path.rglob(file_glob) if p.is_file())
        if not csv_paths:
            raise FileNotFoundError(
                f"No training csv files matched '{file_glob}' under directory: {input_path}"
            )

        frames = [_read_training_csv(csv_path, base_dir=input_path) for csv_path in csv_paths]
        df = pd.concat(frames, ignore_index=True, sort=False)
        print(f"[INFO] Loading training directory: {input_path}")
        print(f"[INFO] Matched training csv files: {len(csv_paths)}")
        return df

    raise FileNotFoundError(f"Training input path not found: {input_path}")


def _encode_station_id(df: pd.DataFrame, drop_invalid_rows: bool) -> pd.DataFrame:
    if "StationId" not in df.columns:
        return df

    station = df["StationId"].astype("string").str.strip()
    valid_mask = station.notna() & station.ne("")

    if drop_invalid_rows:
        dropped = int((~valid_mask).sum())
        if dropped:
            print(f"[warn] Dropping {dropped} rows with missing/blank StationId.")
        df = df.loc[valid_mask].copy().reset_index(drop=True)
        station = station.loc[valid_mask].reset_index(drop=True)

    station = station.where(valid_mask)
    dummies = pd.get_dummies(
        station,
        prefix=STATION_DUMMY_PREFIX,
        prefix_sep=STATION_DUMMY_SEP,
        dtype=np.float32,
    )
    df = df.drop(columns=["StationId"])
    dummies.index = df.index
    return pd.concat([df, dummies], axis=1)


def load_and_preprocess(input_path: Path, target_col: str, valid_ratio: float, file_glob: str = "*_chunk_avg.csv"):
    df = _load_training_dataframe(input_path, file_glob=file_glob)
    df = _encode_station_id(df, drop_invalid_rows=True)

    if target_col not in df.columns:
        raise ValueError(f"{target_col} column not found in CSV.")
    if df.empty:
        raise ValueError("No rows available after StationId filtering.")

    c_free_col = "VOL C FreeMB"
    d_free_col = "VOL D FreeMB"

    if c_free_col in df.columns and d_free_col in df.columns:
        df["C_FreeGB"] = df[c_free_col] / 1024.0
        df["D_FreeGB"] = df[d_free_col] / 1024.0
        df["min_free_gb"] = df[["C_FreeGB", "D_FreeGB"]].min(axis=1)
        df["is_any_volume_low"] = (df["min_free_gb"] < 10).astype(float)
    else:
        print("[warn] VOL C/D FreeMB columns not found; volume features skipped.")

    drop_candidates = CURRENT_METADATA_COLUMNS | LEGACY_METADATA_COLUMNS | COMMON_DROP_COLUMNS
    drop_features = [col for col in df.columns if (col in drop_candidates) or ("imp" in col)]

    y = df[target_col].values.astype(np.float32)

    feature_cols = [
        c for c in df.columns
        if c != target_col and c not in drop_features
    ]

    X = df[feature_cols].copy()
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(
            "Non-numeric feature columns remain after preprocessing: "
            + ", ".join(non_numeric_cols)
        )
    X = X.apply(lambda col: col.fillna(col.median()))
    X = X.values.astype(np.float32)

    N = len(X)
    split = int(N * (1 - valid_ratio))

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    scaler = StandardScaler()
    scaler.fit(X_train)

    print(f"Samples: {N}")
    print(f"Train: {len(X_train)},  Valid: {len(X_val)}")
    print(f"Features: {len(feature_cols)}")
    print("Feature list:")
    for c in feature_cols:
        print("  ", c)

    return (X_train, y_train), (X_val, y_val), scaler, feature_cols


def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
):
    train_ds = ChunkDataset(X_train, y_train)
    val_ds = ChunkDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
