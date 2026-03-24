from dataclasses import dataclass
from pathlib import Path

import torch


CSV_PATH = Path(r"C:\Users\Administrator\Desktop\MEM_LensAA_Training\preprocessed")  # file or directory
CSV_GLOB = "*_chunk_avg.csv"
TARGET_COL = "DeltaSec"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 1_000_000
    lr: float = 1e-4
    valid_ratio: float = 0.2
    csv_glob: str = CSV_GLOB
    best_model_path: Path = Path("ft_transformer_best.pth")
    best_pt_path: Path = Path("ft_transformer_best.pt")
