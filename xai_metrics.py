# -*- coding: utf-8 -*-
"""
TR: 10 XAI metriği (quantus tabanlı). Her metrik için tek arayüz.
EN: 10 XAI metrics (quantus-based). Single interface per metric.

Metrix list:
1) Continuity
2) FaithfulnessEstimate
3) AUC
4) Sparseness
5) Complexity
6) Relevance Rank Accuracy (RRA)
7) Relevance Mass Accuracy (RMA)
8) Faithfulness Correlation
9) Infidelity
10) Model Parameter Randomisation Test (MPRT) – summary score (lower is better)
"""
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import torch
from quantus import (
    Continuity, FaithfulnessEstimate, AUC, Sparseness, Complexity,
    RelevanceRankAccuracy, RelevanceMassAccuracy, FaithfulnessCorrelation,
    Infidelity, ModelParameterRandomisation
)

def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return x

def compute_all_metrics(
    model: torch.nn.Module,
    inputs: torch.Tensor,           # [B,3,H,W], 0..1
    masks: torch.Tensor,            # [B,1,H,W], class id (0/1)
    atts: torch.Tensor,             # [B,1,H,W], 0..255 (will be normalized)
    target_class: int = 1
) -> Dict[str, float]:
    """
    TR: 10 metriği bir arada hesaplar, ortalama döndürür.
    EN: Computes all 10 metrics, returns mean per metric.
    """
    x = inputs.detach()
    y = (masks==target_class).float()  # binary map for class
    a = atts.float()/255.0

    x_np = _to_numpy(x)
    y_np = _to_numpy(y)
    a_np = _to_numpy(a)

    # Common kwargs for quantus
    common = dict(
        nr_samples=10,    # some metrics use sampling
        abs=True,
        normalise=True
    )

    results = {}

    results["Continuity"] = np.mean( Continuity(**common)(model=model, x=x_np, y=y_np, a=a_np) )
    results["FaithfulnessEstimate"] = np.mean( FaithfulnessEstimate(**common)(model=model, x=x_np, y=y_np, a=a_np) )
    results["AUC"] = np.mean( AUC(**common)(model=model, x=x_np, y=y_np, a=a_np) )
    results["Sparseness"] = np.mean( Sparseness(**common)(model=model, x=x_np, y=y_np, a=a_np) )
    results["Complexity"] = np.mean( Complexity(**common)(model=model, x=x_np, y=y_np, a=a_np) )
    results["RRA"] = np.mean( RelevanceRankAccuracy(**common)(model=model, x=x_np, y=y_np, a=a_np) )
    results["RMA"] = np.mean( RelevanceMassAccuracy(**common)(model=model, x=x_np, y=y_np, a=a_np) )
    results["FaithfulnessCorr"] = np.mean( FaithfulnessCorrelation(**common)(model=model, x=x_np, y=y_np, a=a_np) )
    results["Infidelity"] = np.mean( Infidelity(**common)(model=model, x=x_np, y=y_np, a=a_np) )

    # MPRT returns several values across layers; we summarise with mean.
    mprt_scores = ModelParameterRandomisation(**common)(model=model, x=x_np, y=y_np, a=a_np)
    results["MPRT"] = float(np.mean(mprt_scores))

    return results
