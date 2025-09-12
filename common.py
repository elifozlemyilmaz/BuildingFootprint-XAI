"""
Ortak yardımcı fonksiyonlar (path, I/O, seed, logging).
"""

from __future__ import annotations
import os, random, logging
from pathlib import Path
import numpy as np

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=level
    )

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

def list_files(root: str | Path, exts=(".tif",".tiff",".png",".jpg",".jpeg")):
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])

def stem_no_ext(path: str | Path) -> str:
    return Path(path).stem
