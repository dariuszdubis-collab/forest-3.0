"""
forest.ml.infer
===============

Lekka nakładka na onnxruntime – ładowanie modelu ONNX oraz metoda
`predict_proba(X)`.  Plik importujemy **dopiero** gdy potrzebny jest tryb ML,
dzięki temu build „core” nie wymaga biblioteki onnxruntime ani NumPy < 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────  lazy import  ──────────────────────────
try:  # pragma: no cover – gałąź core CI nie ma onnxruntime
    import onnxruntime as ort  # noqa: WPS433 (external binary)
except Exception as err:  # pylint: disable=broad-except
    ort: Any | None = None
    _IMPORT_ERR: Exception | None = err
else:
    _IMPORT_ERR = None


class ONNXModel:
    """Prosty wrapper na `onnxruntime.InferenceSession` (CPU‑only)."""

    def __init__(self, model_path: str | Path) -> None:
        if ort is None:  # NumPy‑ABI lub brak zależności opcjonalnych
            raise RuntimeError(
                "onnxruntime is unavailable – install extras `[ml]` "
                "or check NumPy/ABI mismatch."
            ) from _IMPORT_ERR

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)

        self._sess = ort.InferenceSession(  # type: ignore[attr-defined]
            str(path),
            providers=["CPUExecutionProvider"],
        )
        self._in_name = self._sess.get_inputs()[0].name
        self._out_name = self._sess.get_outputs()[0].name

    # ------------------------------------------------------------------------- #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # shape: (n, n_classes)
        """Zwraca prawdopodobieństwa klas."""
        if X.dtype != np.float32:  # ONNX najczęściej oczekuje float32
            X = X.astype(np.float32, copy=False)
        return self._sess.run([self._out_name], {self._in_name: X})[0]

