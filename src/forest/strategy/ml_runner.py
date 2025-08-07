"""
forest.strategy.ml_runner
=========================

Warstwa adaptacji: ONNXModel  ➜  sygnał (‑1 | 0 | 1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from forest.ml.infer import ONNXModel

__all__ = ["ONNXModel", "ml_signal"]


def ml_signal(
    model: ONNXModel,
    features: pd.DataFrame,
    threshold: float = 0.55,
) -> pd.Series:
    """
    Zamienia predykcje modelu na sygnał tradingowy.

    • Model zakładamy binarny: [p(no‑trade), p(long)]  
    • threshold > p(no‑trade) → 1  (LONG)  
    • threshold < p(no‑trade) → −1 (SHORT)  
    • w pozostałych przypadkach 0 (flat)
    """
    X = features.values.astype(np.float32, copy=False)
    probs: np.ndarray = model.predict_proba(X)

    if probs.shape[1] != 2:  # pragma: no cover
        raise ValueError(f"Expected binary classifier, got shape {probs.shape}")

    long_p = probs[:, 1]
    short_p = probs[:, 0]

    signal = np.where(long_p >= threshold, 1, np.where(short_p >= threshold, -1, 0))
    return pd.Series(signal, index=features.index, name="ml_signal")

