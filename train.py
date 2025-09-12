# -*- coding: utf-8 -*-
"""
TR: Genel eğitim script'i (2 dataset, 3 model). Basit CE kaybı ve IoU takibi.
EN: Generic training script (2 datasets, 3 models). Simple CE loss and IoU tracking.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from src.common import setup_logging, set_seed, ensure_dir, to_device
from src.datasets import make_dataset
from src.models import build_model, logits_to_mask

def iou_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int=2, eps: float=1e-7):
    """
    TR: Basit sınıf-ortalamalı IoU (argmax sonrası).
    EN: Simple mean IoU over classes (after argmax).
    """
    pred = pred.squeeze(1)  # [B,H,W]
    target = target.squeeze(1)
    ious = []
    for c in range(num_classes):
        p = (pred==c); t=(target==c)
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        iou = (inter + eps)/(union + eps)
        ious.append(iou)
    return torch.stack(ious).mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["massachusetts","maxar_izmir"])
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--list_dir", type=str, default=None)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--model", type=str, required=True, choices=["unet++","deeplabv3+","pspnet"])
    ap.add_argument("--encoder", type=str, default="resnet34")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    setup_logging(); set_seed(args.seed)
    out = ensure_dir(args.out_dir)

    # Datasets & loaders
    train_ds = make_dataset(args.dataset, args.root, "train", args.img_size, args.list_dir)
    val_ds   = make_dataset(args.dataset, args.root, "val",   args.img_size, args.list_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = build_model(args.model, num_classes=args.num_classes, encoder_name=args.encoder).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_iou, best_path = -1.0, None

    for epoch in range(1, args.epochs+1):
        # --- Train ---
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            batch = to_device(batch, args.device)
            x, y = batch["image"], batch["mask"].squeeze(1)  # CE expects [B,H,W] targets
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item()*x.size(0)
        tr_loss /= len(train_loader.dataset)

        # --- Val ---
        model.eval()
        vl_loss, vl_iou = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, args.device)
                x, y = batch["image"], batch["mask"]
                logits = model(x)
                loss = crit(logits, y.squeeze(1))
                vl_loss += loss.item()*x.size(0)
                pred = logits_to_mask(logits)
                vl_iou += iou_score(pred, y, args.num_classes)
        vl_loss /= len(val_loader.dataset)
        vl_iou /= len(val_loader)

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={vl_loss:.4f} | val_mIoU={vl_iou:.4f}")

        # Save best
        if vl_iou > best_iou:
            best_iou = vl_iou
            best_path = Path(out)/f"best_{args.model.replace('+','plus')}_{args.dataset}.pth"
            torch.save(model.state_dict(), best_path)

    # Save training summary
    if best_path:
        Path(out/"summary.json").write_text(json.dumps({"best_mIoU": best_iou, "weights": str(best_path)}, indent=2))

if __name__ == "__main__":
    main()
