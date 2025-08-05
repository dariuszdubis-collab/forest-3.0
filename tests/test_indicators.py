import numpy as np
from forest.core.indicators import ema


def test_ema_simple():
    data = np.arange(1, 11, dtype=float)  # [110]
    result = ema(data, period=3)
    # pierwsze 2 pozycje = nan
    assert np.isnan(result[0:2]).all()
    # 3ci element = średnia [1,2,3] = 2
    assert result[2] == 2
    # ostatni element powinien być > poprzedniego przy rosnącym trendzie
    assert result[-1] > result[-2]

def test_invalid_period():
    import pytest
    with pytest.raises(ValueError):
        ema(np.array([1, 2, 3]), period=0)
