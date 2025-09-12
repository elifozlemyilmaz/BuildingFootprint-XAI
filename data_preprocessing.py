"""
Maxar/SPOT 6-7 veri ön-işleme:
- Raster okuma (rasterio)
- Kırpma/pencereleme (sliding window)
- Patch kaydetme (görüntü & maske)
- Train/val/test bölme (opsiyonel)
Not: Maskelerle görüntülerin uzamsal eşlenik olduğundan emin olun (aynı georeferans & piksel grid).
"""

from __future__ import annotations
import argparse, math, json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import cv2

from src.common import setup_logging, ensure_dir, set_seed, list_files, stem_no_ext

def read_raster(path: Path):
    ds = rasterio.open(path)
    arr = ds.read()  # shape: (C, H, W)
    transform = ds.transform
    crs = ds.crs
    return ds, arr, transform, crs

def save_patch_image(out_path: Path, patch: np.ndarray, is_mask: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if patch.ndim == 3 and patch.shape[0] in (1,3,4):  # (C,H,W) -> (H,W,C)
        patch = np.transpose(patch, (1,2,0))
    # normalize image channels if needed (uint8 önerilir)
    if not is_mask:
        patch = np.clip(patch, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    else:
        # mask: tek kanallı 0/1 (veya sınıf id) uint8
        if patch.ndim == 3 and patch.shape[2] == 1:
            patch = patch[..., 0]
        cv2.imwrite(str(out_path), patch.astype(np.uint8))

def sliding_windows(H, W, size, stride):
    for y in range(0, max(1, H - size + 1), stride):
        for x in range(0, max(1, W - size + 1), stride):
            yield x, y

def process_pair(img_path: Path, msk_path: Path, out_img_dir: Path, out_msk_dir: Path,
                 patch_size: int, stride: int, min_foreground_ratio: float | None):
    with rasterio.open(img_path) as di, rasterio.open(msk_path) as dm:
        assert di.width == dm.width and di.height == dm.height, \
            f"Size mismatch: {img_path} vs {msk_path}"
        C = di.count
        H, W = di.height, di.width

        for x, y in sliding_windows(H=H, W=W, size=patch_size, stride=stride):
            win = Window.from_slices((y, y + patch_size), (x, x + patch_size))
            if y + patch_size > H or x + patch_size > W:
                continue

            img = di.read(window=win)  # (C, ps, ps)
            msk = dm.read(1, window=win)  # (ps, ps)

            if min_foreground_ratio is not None:
                fg_ratio = float((msk > 0).sum()) / (patch_size * patch_size)
                if fg_ratio < min_foreground_ratio:
                    continue

            name = f"{stem_no_ext(img_path)}_{y:05d}_{x:05d}.png"
            save_patch_image(out_img_dir / name, img, is_mask=False)
            save_patch_image(out_msk_dir / name, msk, is_mask=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True, help="Girdi görüntü klasörü (Maxar RGB vs.)")
    ap.add_argument("--masks_dir", type=str, required=True, help="Girdi maske klasörü (aynı isimlerle)")
    ap.add_argument("--out_dir",   type=str, required=True, help="Çıkış kök klasörü")
    ap.add_argument("--patch_size", type=int, default=512)
    ap.add_argument("--stride",     type=int, default=512)
    ap.add_argument("--min_fg_ratio", type=float, default=None,
                    help="Örn. 0.01 -> %1'den az bina olan patch'leri atla")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_logging()
    set_seed(args.seed)

    images = list_files(args.images_dir, exts=(".tif",".tiff",".png",".jpg",".jpeg"))
    masks  = list_files(args.masks_dir,  exts=(".tif",".tiff",".png",".jpg",".jpeg"))
    name_to_mask = {p.stem: p for p in masks}

    out_img_dir = ensure_dir(Path(args.out_dir) / "images")
    out_msk_dir = ensure_dir(Path(args.out_dir) / "masks")

    cnt = 0
    for img_path in images:
        key = Path(img_path).stem
        if key not in name_to_mask:
            continue
        msk_path = name_to_mask[key]
        process_pair(
            Path(img_path), Path(msk_path),
            out_img_dir, out_msk_dir,
            patch_size=args.patch_size,
            stride=args.stride,
            min_foreground_ratio=args.min_fg_ratio
        )
        cnt += 1

if __name__ == "__main__":
    main()
