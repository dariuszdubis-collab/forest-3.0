"""Streamlit dashboard – pierwsza, minimalistyczna wersja.

Uruchamianie:
    poetry run forest-dashboard
lub
    streamlit run src/forest/dashboard/app.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from plotly.express import line

from forest.backtest.engine import run_backtest
from forest.backtest.risk import RiskManager
from forest.utils.log import setup_logger

# wyciszamy logi strategii w interfejsie
setup_logger("ERROR")


# ---------------------------------------------------------------------------#
#  Funkcje pomocnicze                                                        #
# ---------------------------------------------------------------------------#
def load_csv(file) -> pd.DataFrame:
    """Ładuje CSV z kolumnami: time, open, high, low, close."""
    df = (
        pd.read_csv(file, parse_dates=["time"])
        .set_index("time")
        .loc[:, ["open", "high", "low", "close"]]
    )
    return df


def show_equity(df: pd.DataFrame) -> None:
    st.plotly_chart(line(df["equity"], title="Equity curve"), use_container_width=True)
    dd = (df["equity"].cummax() - df["equity"]) / df["equity"].cummax()
    st.area_chart(dd, height=160, use_container_width=True)


# ---------------------------------------------------------------------------#
#  Interfejs Streamlit                                                       #
# ---------------------------------------------------------------------------#
def app() -> None:
    st.title("🌲 Forest 3.0 — Dashboard v0")

    file = st.file_uploader("→ Wrzuć CSV z danymi OHLC", type="csv")
    if not file:
        st.info("Oczekuję na plik...")
        return

    df = load_csv(file)
    st.success(f"Wczytano {len(df):,} świec")

    rm = RiskManager(capital=10_000)
    results = run_backtest(df, rm)

    show_equity(results)

    st.subheader("Trades (ostatnie 20)")
    st.dataframe(results[["signal", "equity"]].tail(20), height=200)


# ---------------------------------------------------------------------------#
#  Alias CLI — `poetry run forest-dashboard`                                 #
# ---------------------------------------------------------------------------#
def main() -> None:  # noqa: D401  (używa Poetry scripts)
    """CLI wrapper – odpala Streamlit z tym plikiem."""
    cmd = ["streamlit", "run", os.fspath(Path(__file__).resolve())]
    subprocess.run(cmd + sys.argv[1:], check=False)


if __name__ == "__main__":
    main()

