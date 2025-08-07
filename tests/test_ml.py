import numpy as np
import pandas as pd
import pytest
from forest.ml.infer import ONNXModel
from forest.strategy.ml_runner import ml_signal


@pytest.mark.ml  # <-- ten marker pozwala wykluczyć ML w jobie core‑CI
def test_ml_signal_dummy(tmp_path):
    """
    Dummy‐model ORT (1 cecha) zawsze zwraca p=[0.2, 0.8] → LONG=1.
    """

    # --- 1. budujemy minimalny model w sklearn + skl2onnx
    skl = pytest.importorskip("sklearn.linear_model")        # skip jeżeli brak
    skl2onnx = pytest.importorskip("skl2onnx")               # "

    X = np.random.randn(10, 1).astype(np.float32)
    y = np.ones(10, dtype=int)          # same LONG‑sygnały
    clf = skl.LogisticRegression().fit(X, y)

    from skl2onnx.common.data_types import FloatTensorType
    onx = skl2onnx.convert_sklearn(clf, initial_types=[("x", FloatTensorType([None, 1]))])
    model_path = tmp_path / "dummy.onnx"
    model_path.write_bytes(onx.SerializeToString())

    # --- 2. ONNXModel → ml_signal
    model = ONNXModel(model_path)
    feats = pd.DataFrame(X, columns=["feat"])
    sig = ml_signal(model, feats, threshold=0.55)

    assert (sig == 1).all()

