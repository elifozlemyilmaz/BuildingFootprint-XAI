# -*- coding: utf-8 -*-
"""
TR: Genel eğitim script'i (SPOT6/7 & MAXAR_İzmir; U-Net++, DeepLabv3+, PSPNet).
    - Kayıp: CrossEntropy + (opsiyonel) Dice Loss
    - Metrikler (val): mIoU, Dice, Recall, Precision, Accuracy (+ sınıf-1 IoU/Dice/Rec/Prec)
EN: Generic training script (SPOT6/7 & MAXAR_İzmir; U-Net++, DeepLabv3+, PSPNet).
    - Loss: CrossEntropy + (optional) Dice Loss
    - Metrics (val): mIoU, Dice, Recall, Precision, Accuracy (+ class-1 IoU/Dice/Rec/Prec)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from src.common import setup_logging, set_seed, ensure_dir, to_device, one_hot
from src.datasets import make_dataset
from src.models import build_model, logits_to_mask

# ------------------------ Dice Loss ------------------------ #
class DiceLoss(nn.Module):
    """
    TR: Çok sınıflı dice loss (softmax olasılıkları + one-hot hedef).
    EN: Multi-class Dice loss (softmax probs + one-hot targets).
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        # logits: [B,C,H,W], target_ids: [B,H,W]
        probs = torch.softmax(logits, dim=1)                               # [B,C,H,W]
        target_oh = nn.functional.one_hot(target_ids, probs.size(1))       # [B,H,W,C]
        target_oh = target_oh.permute(0, 3, 1, 2).float()                  # [B,C,H,W]
        inter = (probs * target_oh).sum(dim=(0,2,3))                       # [C]
        union = probs.sum(dim=(0,2,3)) + target_oh.sum(dim=(0,2,3))        # [C]
        dice_c = (2*inter + self.eps) / (union + self.eps)                 # [C]
        dice = dice_c.mean()
        return 1.0 - dice

# ---------------------- Metric helpers --------------------- #
def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int=2, eps: float=1e-7):
    """
    TR: Sınıf-ortalamalı IoU (argmax sonrası).
    EN: Mean IoU across classes (after argmax).
    """
    pred = pred.squeeze(1)   # [B,H,W]
    target = target.squeeze(1)
    ious = []
    for c in range(num_classes):
        p = (pred==c); t=(target==c)
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        iou = (inter + eps)/(union + eps)
        ious.append(iou)
    return torch.stack(ious).mean().item()

def binary_confusion(pred: torch.Tensor, target: torch.Tensor, fg_class: int=1):
    """
    TR: İkili karışıklık matrisi (hedef sınıf=fg_class) -> TP, FP, TN, FN.
    EN: Binary confusion for a given foreground class -> TP, FP, TN, FN.
    """
    pred = pred.squeeze(1)
    target = target.squeeze(1)
    p = (pred==fg_class)
    t = (target==fg_class)
    tp = (p & t).sum().item()
    fp = (p & ~t).sum().item()
    fn = (~p & t).sum().item()
    tn = (~p & ~t).sum().item()
    return tp, fp, tn, fn

def binary_metrics_from_conf(tp, fp, tn, fn, eps: float=1e-7):
    """
    TR: Recall, Precision, Accuracy, Dice ve IoU hesapla.
    EN: Compute Recall, Precision, Accuracy, Dice and IoU.
    """
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    dice = (2*tp) / (2*tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return dict(recall=recall, precision=precision, accuracy=accuracy, dice=dice, iou=iou)

# --------------------------- Main -------------------------- #
def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--dataset", type=str, required=True, choices=["spot67","maxar_izmir"])
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--list_dir", type=str, default=None)
    ap.add_argument("--img_size", type=int, default=512)
    # Model
    ap.add_argument("--model", type=str, required=True, choices=["unet++","deeplabv3+","pspnet"])
    ap.add_argument("--encoder", type=str, default="resnet34")
    ap.add_argument("--num_classes", type=int, default=2)
    # Train
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Loss mix
    ap.add_argument("--dice_weight", type=float, default=0.5,
                    help="TR: Toplam kayıpta Dice katsayısı | EN: Weight of Dice in total loss")
    ap.add_argument("--fg_class", type=int, default=1,
                    help="TR: İstatistikler için ön-plan sınıf id | EN: Foreground class id for stats")
    args = ap.parse_args()

    setup_logging(); set_seed(args.seed)
    out = ensure_dir(args.out_dir)

    # Datasets & loaders
    train_ds = make_dataset(args.dataset, args.root, "train", args.img_size, args.list_dir)
    val_ds   = make_dataset(args.dataset, args.root, "val",   args.img_size, args.list_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model & losses
    model = build_model(args.model, num_classes=args.num_classes, encoder_name=args.encoder).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    best_miou, best_path = -1.0, None

    for epoch in range(1, args.epochs+1):
        # ====================== TRAIN ====================== #
        model.train()
        tr_loss_sum, n_tr = 0.0, 0
        for batch in train_loader:
            batch = to_device(batch, args.device)
            x = batch["image"]                                 # [B,3,H,W]
            y_ids = batch["mask"].squeeze(1).long()            # [B,H,W]
            opt.zero_grad(set_to_none=True)
            logits = model(x)                                   # [B,C,H,W]
            loss_ce = ce_loss(logits, y_ids)
            loss_dice = dice_loss(logits, y_ids) if args.dice_weight>0 else 0.0
            loss = loss_ce + args.dice_weight*loss_dice
            loss.backward()
            opt.step()
            bs = x.size(0)
            tr_loss_sum += float(loss.item()) * bs
            n_tr += bs
        tr_loss = tr_loss_sum / max(1, n_tr)

        # ======================= VAL ======================= #
        model.eval()
        vl_loss_sum, n_vl = 0.0, 0
        # accumulators for metrics
        tp_tot = fp_tot = tn_tot = fn_tot = 0
        miou_sum, dice_sum = 0.0, 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, args.device)
                x = batch["image"]                               # [B,3,H,W]
                y = batch["mask"]                                # [B,1,H,W]
                y_ids = y.squeeze(1).long()                      # [B,H,W]
                logits = model(x)
                # loss
                loss_ce = ce_loss(logits, y_ids)
                loss_dice = dice_loss(logits, y_ids) if args.dice_weight>0 else 0.0
                loss = loss_ce + args.dice_weight*loss_dice
                bs = x.size(0)
                vl_loss_sum += float(loss.item()) * bs
                n_vl += bs
                # predictions
                pred = logits_to_mask(logits)                    # [B,1,H,W]
                # mIoU (macro)
                miou_sum += mean_iou(pred, y, args.num_classes)
                # Dice (binary, fg_class)
                # convert to confusion and from it compute dice also (keeps consistency)
                tp, fp, tn, fn = binary_confusion(pred, y, fg_class=args.fg_class)
                tp_tot += tp; fp_tot += fp; tn_tot += tn; fn_tot += fn

        vl_loss = vl_loss_sum / max(1, n_vl)
        # aggregate binary metrics
        bin_stats = binary_metrics_from_conf(tp_tot, fp_tot, tn_tot, fn_tot)
        # report also macro mIoU; for "Dice (macro)" approximate via class-1 Dice from confusion
        miou = miou_sum / max(1, len(val_loader))

        # Pretty print
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f} | val_loss={vl_loss:.4f} | "
            f"val_mIoU={miou:.4f} | "
            f"val_fg(IoU/Dice/Rec/Prec/Acc)="
            f"{bin_stats['iou']:.4f}/{bin_stats['dice']:.4f}/{bin_stats['recall']:.4f}/"
            f"{bin_stats['precision']:.4f}/{bin_stats['accuracy']:.4f}"
        )

        # Save best (by mIoU)
        if miou > best_miou:
            best_miou = miou
            tag = f"{args.model.replace('+','plus')}_{args.dataset}"
            best_path = Path(out)/f"best_{tag}.pth"
            torch.save(model.state_dict(), best_path)

    # Save summary JSON
    if best_path:
        Path(out/"summary.json").write_text(
            json.dumps(
                {
                    "best_mIoU": best_miou,
                    "weights": str(best_path),
                    "dice_weight": args.dice_weight,
                    "fg_class": args.fg_class
                },
                indent=2
            )
        )

if __name__ == "__main__":
    main()
