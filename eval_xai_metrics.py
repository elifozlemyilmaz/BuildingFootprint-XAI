# -*- coding: utf-8 -*-
"""
TR: Test kümesi üzerinde XAI haritaları ve 10 metrik hesabı (SPOT6/7, MAXAR_İzmir).
EN: Compute XAI maps & all 10 metrics over a test set (SPOT6/7, MAXAR_İzmir).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import setup_logging, ensure_dir, to_device
from dataset import make_dataset
from models import build_model
from xai import xai_dispatch
from xai_metrics import compute_all_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["spot67","maxar_izmir"])
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--list_dir", type=str, default=None)
    ap.add_argument("--split", type=str, default="test", choices=["val","test"])
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--model", type=str, required=True, choices=["unet++","deeplabv3+","pspnet"])
    ap.add_argument("--encoder", type=str, default="resnet34")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--xai_method", type=str, required=True, choices=["saliency","ig","gradshap"])
    ap.add_argument("--xai_class", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    setup_logging()
    ds = make_dataset(args.dataset, args.root, args.split, args.img_size, args.list_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args.model, num_classes=args.num_classes, encoder_name=args.encoder)
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().to(args.device)

    all_scores = []
    for batch in loader:
        batch = to_device(batch, args.device)
        x, y = batch["image"], batch["mask"]  # x: [B,3,H,W]; y: [B,1,H,W]
        a = xai_dispatch(args.xai_method, model, x, target_class=args.xai_class).float()  # [B,1,H,W]
        scores = compute_all_metrics(model, x, y, a, target_class=args.xai_class)
        all_scores.append(scores)

    keys = list(all_scores[0].keys())
    means = {k: float(np.mean([d[k] for d in all_scores])) for k in keys}

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"method": args.xai_method, "model": args.model,
                                               "dataset": args.dataset, "class": args.xai_class,
                                               "metrics_mean": means}, indent=2))
    print("Saved:", args.out_json)

if __name__ == "__main__":
    main()
