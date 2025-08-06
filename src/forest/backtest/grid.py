from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import joblib
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from forest.backtest.engine import run_backtest
from forest.backtest.risk import RiskManager

# ---------------------------------------------------------------------------#
#  Persistent Joblib cache                                                   #
# ---------------------------------------------------------------------------#
_CACHE_DIR = Path.home() / ".cache" / "forest_grid"
_MEMORY = joblib.Memory(_CACHE_DIR, verbose=0)

# ---------------------------------------------------------------------------#
#  Wynik jednej symulacji                                                    #
# ---------------------------------------------------------------------------#
@dataclass(slots=True)
class GridResult:
    params: Dict[str, Any]
    equity_end: float
    max_dd: float
    cagr: float
    rar: float  # risk‑adjusted return = CAGR / max_dd

# ---------------------------------------------------------------------------#
#  Param-grid generator                                                      #
# ---------------------------------------------------------------------------#
def param_grid(**param_ranges) -> Iterable[dict]:
    keys, values = zip(*param_ranges.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

# ---------------------------------------------------------------------------#
#  Hash danych OHLC → md5                                                    #
# ---------------------------------------------------------------------------#
def _hash_df(df: pd.DataFrame) -> str:
    h = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update(",".join(df.columns).encode())
    return h.hexdigest()

# ---------------------------------------------------------------------------#
#  Pojedynczy przebieg (cached)                                              #
# ---------------------------------------------------------------------------#
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

    equity_end = float(res["equity"].iloc[-1])
    dd = (res["equity"].cummax() - res["equity"]) / res["equity"].cummax()
    max_dd = float(dd.max())

    days = (res.index[-1] - res.index[0]).days or 1
    years = days / 365.25
    cagr = (equity_end / rm.capital) ** (1 / years) - 1 if years > 0 else 0.0
    rar = cagr / max_dd if max_dd > 0 else 0.0

    return GridResult(p, equity_end, max_dd, cagr, rar)

# ---------------------------------------------------------------------------#
#  Główny runner                                                             #
# ---------------------------------------------------------------------------#
def run_grid(
    df: pd.DataFrame,
    grid: Iterable[dict],
    make_risk: Callable[[], RiskManager] | None = None,
    n_jobs: int = -1,
    export_path: str | Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Batch back‑test z multiprocessing i persistent cache."""
    make_risk = make_risk or (lambda: RiskManager(capital=10_000))
    grid_list: List[dict] = list(grid)
    df_hash = _hash_df(df)

    # funkcja robocza (def zamiast lambda – Ruff E731)
    def _worker(params: dict) -> GridResult:
        key = tuple(sorted(params.items()))
        if use_cache:
            return _single_run_cached(df_hash, key, df, make_risk)
        # .call() → pomija cache
        return _single_run_cached.call(df_hash, key, df, make_risk)

    iterator = tqdm(grid_list, desc="ParamGrid", leave=False)

    if n_jobs == 1:
        results = [_worker(p) for p in iterator]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_worker)(p) for p in iterator
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
            raise ValueError("export_path musi mieć rozszerzenie .parquet lub .csv")

    return out

