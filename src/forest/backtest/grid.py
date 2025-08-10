# src/forest/backtest/grid.py
from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import joblib
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from forest.backtest.engine import run_backtest
from forest.backtest.risk import RiskManager

# ---------------- persistent cache (for backtest runs) --------------------
_CACHE_DIR = Path.home() / ".cache" / "forest_grid"
_MEMORY = joblib.Memory(_CACHE_DIR, verbose=0)


# ---------------- wynik pojedynczego przebiegu ----------------
@dataclass(slots=True)
class GridResult:
    params: Dict[str, Any]
    equity_end: float
    max_dd: float
    cagr: float
    rar: float  # CAGR / max_dd
    sharpe: float  # annualised Sharpe ratio


# ---------------- generator kombinacji parametrów -------------------
def param_grid(**param_ranges) -> Iterable[dict]:
    """Generuje listę słowników ze wszystkimi kombinacjami parametrów (pełna siatka)."""
    # deterministyczna kolejność kluczy
    keys, values = zip(*sorted(param_ranges.items()))
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# ---------------- funkcja pomocnicza: hash danych OHLC -------------------
def _hash_df(df: pd.DataFrame) -> str:
    """Oblicza unikatowy hash dla danych (wartości + kolumny), używany do cache."""
    h = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update(",".join(df.columns).encode())
    return h.hexdigest()


# ---------------- pojedynczy bieg symulacji (z cache) --------------------
@_MEMORY.cache(ignore=["df", "make_risk"])
def _single_run_cached(
    df_hash: str,
    params: Tuple[Tuple[str, Any], ...],
    df: pd.DataFrame,
    make_risk: Callable[[], RiskManager],
) -> GridResult:
    """Wykonuje pojedynczy backtest dla danych i zadanych parametrów, zwraca agregaty wyników."""
    p = dict(params)
    fast, slow = p["fast"], p["slow"]

    rm = make_risk()
    res = run_backtest(df, rm, fast, slow)

    equity = res["equity"]
    equity_end = float(equity.iloc[-1])

    dd = (equity.cummax() - equity) / equity.cummax()
    max_dd = float(dd.max())

    # CAGR – roczna stopa zwrotu z uwzględnieniem liczby dni w danych
    days = (res.index[-1] - res.index[0]).days or 1
    years = days / 365.25
    cagr = (equity_end / rm.capital) ** (1 / years) - 1 if years > 0 else 0.0

    rar = cagr / max_dd if max_dd > 0 else 0.0

    # Sharpe – na podstawie dziennych zmian equity
    # Uwaga: fill_method=None usuwa FutureWarning z Pandas 2.x
    rets = equity.pct_change(fill_method=None).dropna()
    if not rets.empty and rets.std() != 0:
        sharpe = sqrt(252) * rets.mean() / rets.std()
    else:
        sharpe = 0.0

    return GridResult(p, equity_end, max_dd, cagr, rar, sharpe)


# ---------------- główna funkcja grid search -------------------------
def run_grid(
    df: pd.DataFrame,
    grid: Iterable[dict],
    make_risk: Callable[[], RiskManager] | None = None,
    n_jobs: int = -1,
    export_path: str | Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Uruchamia serię backtestów dla wszystkich kombinacji parametrów podanych w grid.
    Zwraca DataFrame z wynikami (metryki dla każdej kombinacji).
    """
    make_risk = make_risk or (lambda: RiskManager(capital=10_000))

    # Oblicz hash danych + dołącz parametry RiskManager, aby uniknąć kolizji cache
    df_hash = _hash_df(df)
    test_rm = make_risk()
    risk_key = f"{test_rm.capital}_{test_rm.risk_per_trade}_{test_rm.max_drawdown}"
    df_hash = f"{df_hash}_{risk_key}"

    grid_list: List[dict] = list(grid)

    # Funkcja pomocnicza do uruchamiania pojedynczej kombinacji
    def _worker(params: dict) -> GridResult:
        key = tuple(sorted(params.items()))
        if use_cache:
            return _single_run_cached(df_hash, key, df, make_risk)
        # Jeśli cache wyłączony, wywołujemy funkcję bez pamięci podręcznej
        return _single_run_cached.call(df_hash, key, df, make_risk)

    # Uruchom backtesty sekwencyjnie lub równolegle w zależności od n_jobs
    iterator = tqdm(grid_list, desc="ParamGrid", leave=False)
    results = (
        [_worker(p) for p in iterator]
        if n_jobs == 1
        else Parallel(n_jobs=n_jobs)(delayed(_worker)(p) for p in iterator)
    )

    # Konwersja wyników do DataFrame
    out = pd.DataFrame([asdict(r) for r in results])

    # Zapis wyników do pliku (jeśli podano ścieżkę eksportu)
    if export_path:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        if export_path.suffix == ".parquet":
            out.to_parquet(export_path, index=False)
        elif export_path.suffix == ".csv":
            out.to_csv(export_path, index=False)
        else:
            raise ValueError("export_path must end with .parquet or .csv")

    return out

