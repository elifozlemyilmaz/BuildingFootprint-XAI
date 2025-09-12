# -*- coding: utf-8 -*-
"""
TR: 3 XAI yöntemi: Saliency (grad), IntegratedGradients, GradientShap (Captum).
EN: 3 XAI methods: Saliency (grad), IntegratedGradients, GradientShap (Captum).
"""
from __future__ import annotations
from typing import Literal
import torch
from captum.attr import IntegratedGradients, GradientShap

def saliency_map(model: torch.nn.Module, x: torch.Tensor, target_class: int) -> torch.Tensor:
    """
    TR: Basit gradyan tabanlı saliency (giriş gradyanlarının |.|).
    EN: Simple gradient-based saliency (|input gradients|).
    Returns: [B,1,H,W] uint8 (0..255 scaled)
    """
    x = x.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)  # [B,C,H,W]
    target = logits[:, target_class:target_class+1].sum()
    target.backward()
    sal = x.grad.detach().abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
    sal = (255*(sal - sal.min())/(sal.max()-sal.min()+1e-8)).clamp(0,255).byte()
    return sal

def integrated_gradients_map(model: torch.nn.Module, x: torch.Tensor, target_class: int, steps: int=50) -> torch.Tensor:
    """
    TR: Integrated Gradients (Captum).
    EN: Integrated Gradients via Captum.
    """
    def fwd(inp):
        return model(inp)[:, target_class]
    ig = IntegratedGradients(fwd)
    baseline = torch.zeros_like(x)
    attributions = ig.attribute(x, baseline, target=None, n_steps=steps, internal_batch_size=x.size(0))
    sal = attributions.abs().mean(dim=1, keepdim=True)
    sal = (255*(sal - sal.min())/(sal.max()-sal.min()+1e-8)).clamp(0,255).byte()
    return sal

def gradient_shap_map(model: torch.nn.Module, x: torch.Tensor, target_class: int, stdevs: float=0.09, nsamples: int=20) -> torch.Tensor:
    """
    TR: GradientShap (Captum) – stokastik entegre gradyan varyantı.
    EN: GradientShap – stochastic integrated gradient variant.
    """
    def fwd(inp):
        return model(inp)[:, target_class]
    gs = GradientShap(fwd)
    baseline_dist = torch.stack([torch.zeros_like(x), torch.ones_like(x)*x.mean()])  # two baselines
    attributions = gs.attribute(x, baselines=baseline_dist, stdevs=stdevs, n_samples=nsamples)
    sal = attributions.abs().mean(dim=1, keepdim=True)
    sal = (255*(sal - sal.min())/(sal.max()-sal.min()+1e-8)).clamp(0,255).byte()
    return sal

def xai_dispatch(method: Literal["saliency","ig","gradshap"], model, x, target_class: int):
    """
    TR: XAI yöntem seçici.
    EN: XAI method dispatcher.
    """
    m = method.lower()
    if m == "saliency":  return saliency_map(model, x, target_class)
    if m == "ig":        return integrated_gradients_map(model, x, target_class)
    if m == "gradshap":  return gradient_shap_map(model, x, target_class)
    raise ValueError(f"Unknown XAI method: {method}")
