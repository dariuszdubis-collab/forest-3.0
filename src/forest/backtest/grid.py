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


# ---------------------------------------------------------------------------#
#  Struktura wyniku jednego przebiegu                                         #
# ---------------------------------------------------------------------------#
@dataclass(slots=True)
class GridResult:
    params: Dict[str, Any]
    equity_end: float
    max_dd: float
    cagr: float            # ⮕ roczna stopa zwrotu
    rar: float             # ⮕ risk‑adjusted return = CAGR / max_dd


# ---------------------------------------------------------------------------#
#  Generator kombinacji parametrów                                            #
# ---------------------------------------------------------------------------#
def param_grid(**param_ranges) -> Iterable[dict]:
    keys, values = zip(*param_ranges.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# ---------------------------------------------------------------------------#
#  Jedna symulacja                                                            #
# ---------------------------------------------------------------------------#
def _single_run(
    df: pd.DataFrame,
    params: dict,
    make_risk: Callable[[], RiskManager],
) -> GridResult:
    fast, slow = params["fast"], params["slow"]
    rm = make_risk()
    res = run_backtest(df, rm, fast, slow)

    equity_end = float(res["equity"].iloc[-1])
    dd = (res["equity"].cummax() - res["equity"]) / res["equity"].cummax()
    max_dd = float(dd.max())

    # CAGR
    days = (res.index[-1] - res.index[0]).days or 1
    years = days / 365.25
    cagr = (equity_end / rm.capital) ** (1 / years) - 1 if years > 0 else 0.0

    rar = cagr / max_dd if max_dd > 0 else 0.0
    return GridResult(params, equity_end, max_dd, cagr, rar)


# ---------------------------------------------------------------------------#
#  Główny runner                                                              #
# ---------------------------------------------------------------------------#
def run_grid(
    df: pd.DataFrame,
    grid: Iterable[dict],
    make_risk: Callable[[], RiskManager] | None = None,
    n_jobs: int = -1,
    export_path: str | Path | None = None,
) -> pd.DataFrame:
    """Wykonuje równoległy back‑test dla każdej kombinacji parametrów."""
    make_risk = make_risk or (lambda: RiskManager(capital=10_000))
    grid_list: List[dict] = list(grid)

    runner = (
        [_single_run(df, p, make_risk) for p in tqdm(grid_list, desc="Param grid")]
        if n_jobs == 1
        else Parallel(n_jobs=n_jobs)(
            delayed(_single_run)(df, p, make_risk) for p in tqdm(grid_list, desc="Param grid")
        )
    )

    out = pd.DataFrame([asdict(r) for r in runner])

    # --- eksport opcjonalny ------------------------------------------------
    if export_path:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        if export_path.suffix == ".parquet":
            out.to_parquet(export_path, index=False)
        elif export_path.suffix == ".csv":
            out.to_csv(export_path, index=False)
        else:
            raise ValueError("export_path musi mieć .parquet lub .csv")
    # ---------------------------------------------------------------------- #

    return out

