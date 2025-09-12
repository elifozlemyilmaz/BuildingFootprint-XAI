# -*- coding: utf-8 -*-
"""
TR: 3 model: U-Net++, DeepLabv3+, PSPNet (smp kullanımı önerildi).
EN: 3 models: U-Net++, DeepLabv3+, PSPNet (using segmentation_models_pytorch).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def build_model(model_name: str, num_classes: int=2, in_channels: int=3,
                encoder_name: str="resnet34", encoder_weights: str|None="imagenet") -> nn.Module:
    """
    TR: Model kurucu.
    EN: Model builder.
    """
    m = model_name.lower()
    if m in ["unet++","unetpp","unetplusplus","u-net++"]:
        model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                 in_channels=in_channels, classes=num_classes)
    elif m in ["deeplabv3+","deeplab","deeplabv3plus"]:
        model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                  in_channels=in_channels, classes=num_classes)
    elif m in ["pspnet","psp"]:
        model = smp.PSPNet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                           in_channels=in_channels, classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

def logits_to_mask(logits: torch.Tensor) -> torch.Tensor:
    """TR: [B,C,H,W] -> [B,1,H,W] sınıf id | EN: argmax to class id mask"""
    return torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)
