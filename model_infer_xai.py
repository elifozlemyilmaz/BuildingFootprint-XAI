from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import segmentation_models_pytorch as smp

from src.common import setup_logging, ensure_dir

def load_unetpp(num_classes: int, encoder_name: str = "resnet34", encoder_weights: str | None = "imagenet"):
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes
    )
    return model

def read_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def preprocess(img: np.ndarray) -> torch.Tensor:
    # [H,W,3] -> [1,3,H,W], 0..1 float
    t = torch.from_numpy(img).float() / 255.0
    t = t.permute(2,0,1).unsqueeze(0)
    return t

def postprocess_mask(logits: torch.Tensor) -> np.ndarray:
    # logits: [1, C, H, W]
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred

def simple_grad_saliency(model: torch.nn.Module, x: torch.Tensor, target_class: int = 1):
    """
    Çok sınıflı maskede 'target_class' için logits gradyanı tabanlı saliency.
    x: [1,3,H,W] requires_grad True
    """
    x = x.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)  # [1,C,H,W]
    # sınıf logitlerinin toplamını maximizasyon hedefi olarak al
    target = logits[:, target_class:target_class+1, :, :].sum()
    target.backward()
    sal = x.grad.detach().abs().mean(dim=1, keepdim=False).squeeze(0)  # [H,W]
    sal = sal.cpu().numpy()
    # normalize 0..255
    sal = (255 * (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)).astype(np.uint8)
    return sal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="Girdi RGB görüntü yolu")
    ap.add_argument("--weights", type=str, required=False, default=None, help="Model ağırlık dosyası (.pth)")
    ap.add_argument("--out_dir", type=str, required=True, help="Çıkış klasörü")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--encoder", type=str, default="resnet34")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--xai_class", type=int, default=1, help="Saliency için hedef sınıf id (ör. bina=1)")
    args = ap.parse_args()

    setup_logging()
    out_dir = ensure_dir(args.out_dir)

    # Model
    model = load_unetpp(num_classes=args.num_classes, encoder_name=args.encoder)
    if args.weights:
        sd = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(sd, strict=True)
    model.eval().to(args.device)

    # Görüntü oku & tensöre çevir
    img = read_rgb(args.image)
    x = preprocess(img).to(args.device)

    with torch.no_grad():
        logits = model(x)
    pred = postprocess_mask(logits)

    # XAI: gradient saliency
    sal = simple_grad_saliency(model, x, target_class=args.xai_class)

    # Kayıt
    base = Path(args.image).stem
    cv2.imwrite(str(out_dir / f"{base}_pred.png"), pred * (255 // max(1, args.num_classes - 1)))
    cv2.imwrite(str(out_dir / f"{base}_saliency.png"), sal)

if __name__ == "__main__":
    main()
