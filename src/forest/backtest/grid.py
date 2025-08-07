"""Prosty grid‑runner z opcjonalnym cache Joblib."""
from __future__ import annotations

import hashlib
import itertools
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from tqdm.auto import tqdm

from forest.backtest.engine import run_backtest
from forest.backtest.risk import RiskManager

_memory = Memory(location=str(Path.home() / ".forest_joblib"), verbose=0)


def _hash_df(df: pd.DataFrame) -> str:
    """Niezależny od indeksu hash rameczki (dla cache)."""
    h = hashlib.blake2b(df.to_numpy().tobytes(), digest_size=8)
    h.update(np.ascontiguousarray(df.columns).tobytes())
    return h.hexdigest()


@_memory.cache
def _single_run_cached(
    df_hash: str,
    params: tuple[tuple[str, Any], ...],
    df: pd.DataFrame,
    make_risk: Callable[[], RiskManager],
) -> dict[str, Any]:
    """Pojedynczy back‑test + mała sztuczna zwłoka (lepiej widać cache)."""
    time.sleep(0.003)  # tylko pierwsze uruchomienie, bo potem cache
    kwargs = dict(params)
    res = run_backtest(df, make_risk(), **kwargs)
    equity_end = res["equity"].iloc[-1]
    max_dd = make_risk().max_drawdown  # zeroed RiskManager
    rar = (equity_end - make_risk().capital) / (max_dd or 1)
    return {"params": kwargs, "equity_end": equity_end, "max_dd": max_dd, "rar": rar}


def param_grid(**kwargs) -> list[dict[str, Any]]:
    """Zamienia listy parametrów na listę słowników dla exhaustive‑grid."""
    keys = list(kwargs)
    vals = [kwargs[k] for k in keys]
    return [dict(zip(keys, v, strict=True)) for v in itertools.product(*vals)]


def run_grid(
    df: pd.DataFrame,
    grid: list[dict[str, Any]],
    *,
    make_risk: Callable[[], RiskManager],
    n_jobs: int = 1,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Wykonuje grid‑search z opcjonalnym cache Joblib."""
    df_hash = _hash_df(df)
    grid_list = list(grid)

    def _worker(p: dict[str, Any]) -> dict[str, Any]:
        if use_cache:
            return _single_run_cached(df_hash, tuple(sorted(p.items())), df, make_risk)
        return _single_run_cached.call(df_hash, tuple(sorted(p.items())), df, make_risk)

    iterator = tqdm(grid_list, desc="ParamGrid", leave=False)
    records = Parallel(n_jobs=n_jobs)(delayed(_worker)(p) for p in iterator)
    return pd.DataFrame(records)

