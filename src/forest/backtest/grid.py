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

# ---------------- persistent cache --------------------
_CACHE_DIR = Path.home() / ".cache" / "forest_grid"
_MEMORY = joblib.Memory(_CACHE_DIR, verbose=0)

# ---------------- wynik pojedynczego przebiegu --------
@dataclass(slots=True)
class GridResult:
    params: Dict[str, Any]
    equity_end: float
    max_dd: float
    cagr: float
    rar: float        # CAGR / max_dd
    sharpe: float     # annualised Sharpe ratio

# ---------------- generator kombinacji ----------------
def param_grid(**param_ranges) -> Iterable[dict]:
    keys, values = zip(*param_ranges.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

# ---------------- hash OHLC ---------------------------
def _hash_df(df: pd.DataFrame) -> str:
    h = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update(",".join(df.columns).encode())
    return h.hexdigest()

# ---------------- base calculation (cached) -----------
@_MEMORY.cache(ignore=["df", "make_risk"])
def _single_run_cached(
    df_hash: str,
    params: Tuple[Tuple[str, Any], ...],
    df: pd.DataFrame,
    make_risk: Callable[[], RiskManager],
) -> GridResult:
    p = dict(params)
    fast, slow = p["fast"], p["slow"]

    rm = make_risk()
    res = run_backtest(df, rm, fast, slow)

    equity = res["equity"]
    equity_end = float(equity.iloc[-1])

    dd = (equity.cummax() - equity) / equity.cummax()
    max_dd = float(dd.max())

    # CAGR
    days = (res.index[-1] - res.index[0]).days or 1
    years = days / 365.25
    cagr = (equity_end / rm.capital) ** (1 / years) - 1 if years > 0 else 0.0
    rar = cagr / max_dd if max_dd > 0 else 0.0

    # Sharpe – dzienne zmiany equity
    rets = equity.pct_change().dropna()
    sharpe = (
        sqrt(252) * rets.mean() / rets.std()
        if not rets.empty and rets.std() != 0
        else 0.0
    )

    return GridResult(p, equity_end, max_dd, cagr, rar, sharpe)

# ---------------- main runner -------------------------
def run_grid(
    df: pd.DataFrame,
    grid: Iterable[dict],
    make_risk: Callable[[], RiskManager] | None = None,
    n_jobs: int = -1,
    export_path: str | Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    make_risk = make_risk or (lambda: RiskManager(capital=10_000))
    df_hash = _hash_df(df)
    grid_list: List[dict] = list(grid)

    def _worker(params: dict):
        key = tuple(sorted(params.items()))
        if use_cache:
            return _single_run_cached(df_hash, key, df, make_risk)
        return _single_run_cached.call(df_hash, key, df, make_risk)

    iterator = tqdm(grid_list, desc="ParamGrid", leave=False)
    results = (
        [_worker(p) for p in iterator]
        if n_jobs == 1
        else Parallel(n_jobs=n_jobs)(delayed(_worker)(p) for p in iterator)
    )

    out = pd.DataFrame([asdict(r) for r in results])

    if export_path:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        if export_path.suffix == ".parquet":
            out.to_parquet(export_path, index=False)
        elif export_path.suffix == ".csv":
            out.to_csv(export_path, index=False)
        else:
            raise ValueError("export_path musi kończyć się .parquet lub .csv")

    return out

