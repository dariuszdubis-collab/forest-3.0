from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from forest.backtest.engine import run_backtest
from forest.backtest.risk import RiskManager


@dataclass(slots=True)
class GridResult:
    params: Dict[str, Any]
    equity_end: float
    max_dd: float


def param_grid(**param_ranges) -> Iterable[dict]:
    keys, values = zip(*param_ranges.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# ---------------------------------------------------------------------------#
#  Wewnętrzna pojedyncza symulacja                                           #
# ---------------------------------------------------------------------------#
def _single_run(df: pd.DataFrame, params: dict, make_risk: Callable[[], RiskManager]) -> GridResult:
    fast, slow = params["fast"], params["slow"]
    res = run_backtest(df, make_risk(), fast, slow)

    equity_end = res["equity"].iloc[-1]
    dd = (res["equity"].cummax() - res["equity"]) / res["equity"].cummax()
    return GridResult(params, float(equity_end), float(dd.max()))


# ---------------------------------------------------------------------------#
#  Publiczna funkcja run_grid                                                #
# ---------------------------------------------------------------------------#
def run_grid(
    df: pd.DataFrame,
    grid: Iterable[dict],
    make_risk: Callable[[], RiskManager] | None = None,
    n_jobs: int | None = -1,
    export_path: str | Path | None = None,          # NEW
) -> pd.DataFrame:
    """Uruchamia batch back‑test i (opcjonalnie) eksportuje wyniki.

    Parameters
    ----------
    df : DataFrame
    grid : iterable of dict
    make_risk : callable -> RiskManager
    n_jobs : int
        ‑1 = wszystkie CPU, 1 = bez multiprocessing.
    export_path : str | Path | None
        Jeśli podano – zapisuje wyniki:
            *.parquet → df.to_parquet()
            *.csv     → df.to_csv()
    """
    make_risk = make_risk or (lambda: RiskManager(capital=10_000))
    grid_list: List[dict] = list(grid)

    if n_jobs == 1:
        results = [_single_run(df, p, make_risk) for p in tqdm(grid_list, desc="Param grid")]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_single_run)(df, params, make_risk) for params in tqdm(grid_list, desc="Param grid")
        )

    out = pd.DataFrame([asdict(r) for r in results])

    # ------------------- eksport ----------------------------------------- #
    if export_path:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        if export_path.suffix == ".parquet":
            out.to_parquet(export_path, index=False)
        elif export_path.suffix == ".csv":
            out.to_csv(export_path, index=False)
        else:
            raise ValueError("export_path musi mieć rozszerzenie .parquet albo .csv")
    # -------------------------------------------------------------------- #

    return out

