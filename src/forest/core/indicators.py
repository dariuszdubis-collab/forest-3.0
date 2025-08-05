import numpy as np

def ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average for a 1D NumPy array.
    :param prices: array of floats
    :param period: lookback window (e.g. 10, 20)
    :return: array same length, first `period-1` values = np.nan
    """
    if period < 1:
        raise ValueError("period must be 1")
    alpha = 2 / (period + 1)
    ema = np.empty_like(prices, dtype=float)
    ema[:] = np.nan
    if len(prices) == 0:
        return ema
    ema[period - 1] = np.mean(prices[:period])
    for i in range(period, len(prices)):
        ema[i] = prices[i] * alpha + ema[i - 1] * (1 - alpha)
    return ema
