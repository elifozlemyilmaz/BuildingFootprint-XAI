# -*- coding: utf-8 -*-
"""
TR: Ortak yardımcılar (kayıt, seed, yol işlemleri)
EN: Common utilities (logging, seeding, path helpers)
"""
from __future__ import annotations
import os, random, logging
from pathlib import Path
from typing import Iterable, Tuple, List
import numpy as np
import torch

def setup_logging(level=logging.INFO) -> None:
    """TR: Basit logging ayarı | EN: Simple logging config"""
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=level)

def ensure_dir(p: str | Path) -> Path:
    """TR: Klasörü oluştur (varsa dokunma) | EN: Create directory if missing"""
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def set_seed(seed: int = 42) -> None:
    """TR: Tekrarlanabilirlik için seed | EN: Seed for reproducibility"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_device(batch, device: str):
    """TR: Batch'i cihaza taşı | EN: Move batch to device"""
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device) if hasattr(batch, "to") else batch

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-7) -> torch.Tensor:
    """
    TR: Binary/one-hot için basit Dice. 
    EN: Simple Dice for binary/one-hot predictions.
    """
    pred = pred.float(); target = target.float()
    inter = (pred*target).sum()
    union = pred.sum() + target.sum()
    return (2*inter + eps)/(union + eps)

def one_hot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """TR: [B,1,H,W] -> [B,C,H,W] | EN: To one-hot"""
    b, _, h, w = mask.shape
    oh = torch.zeros(b, num_classes, h, w, device=mask.device, dtype=torch.float32)
    return oh.scatter_(1, mask.long(), 1.0) if mask.shape[1] != num_classes else mask

def save_gray_png(path: Path, arr: np.ndarray) -> None:
    """TR: Gri PNG kaydet | EN: Save grayscale PNG"""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr.astype(np.uint8))

def save_rgb_png(path: Path, arr: np.ndarray) -> None:
    """TR: RGB PNG kaydet | EN: Save RGB PNG"""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 3 and arr.shape[0] in (3,4):
        arr = np.transpose(arr, (1,2,0))
    cv2.imwrite(str(path), cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR))
