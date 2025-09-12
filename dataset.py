# -*- coding: utf-8 -*-
"""
TR: İki veri seti desteği: SPOT6/7, MAXAR_İzmir.
EN: Two dataset support: SPOT6/7, MAXAR_İzmir.

Varsayım / Assumption:
- Görüntüler: {root}/images/*.png
- Maskeler : {root}/masks/*.png   (0=background, 1=building)
- Eğitim/Doğrulama/Test listeleri opsiyonel txt dosyaları (image stem list).
  Örn: lists/spot67_train.txt, lists/spot67_val.txt, lists/spot67_test.txt
       lists/maxar_izmir_train.txt, ...
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple
import cv2, numpy as np, torch
from torch.utils.data import Dataset

def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None: raise FileNotFoundError(path)
    return m

class SegDataset(Dataset):
    """
    TR: Basit segmentasyon veri kümesi sarmalayıcı.
    EN: Simple segmentation dataset wrapper.
    """
    def __init__(self, root: str | Path, split_list: Optional[Path]=None, size: Optional[int]=None):
        self.root = Path(root)
        self.img_dir = self.root/"images"
        self.msk_dir = self.root/"masks"
        if split_list and Path(split_list).exists():
            stems = [l.strip() for l in Path(split_list).read_text().splitlines() if l.strip()]
            self.items = [(self.img_dir/f"{s}.png", self.msk_dir/f"{s}.png") for s in stems]
        else:
            imgs = sorted(self.img_dir.glob("*.png"))
            self.items = [(ip, self.msk_dir/f"{ip.stem}.png") for ip in imgs]
        self.size = size

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        ip, mp = self.items[idx]
        img = _read_rgb(ip)
        msk = _read_mask(mp)
        if self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        # to tensor
        img_t = torch.from_numpy(img).float().permute(2,0,1) / 255.0
        msk_t = torch.from_numpy(msk).long().unsqueeze(0)  # [1,H,W]
        return {"image": img_t, "mask": msk_t, "stem": ip.stem}

def make_dataset(name: str, root: str | Path, split: str="train",
                 size: int|None=None, list_dir: str|Path|None=None) -> SegDataset:
    """
    TR: Veri seti fabrika fonksiyonu.
    EN: Dataset factory.
    """
    name = name.lower()
    if name not in {"spot67", "maxar_izmir"}:
        raise ValueError(f"Unknown dataset name: {name} (expected 'spot67' or 'maxar_izmir')")
    # split list path (optional)
    sp = None
    if list_dir:
        sp = Path(list_dir)/f"{name}_{split}.txt"  # e.g., lists/spot67_train.txt
    return SegDataset(root=root, split_list=sp, size=size)
