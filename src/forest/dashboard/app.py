"""Streamlit dashboard – v1 (equity/DD + ParamGrid heat‑map)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from forest.backtest.engine import run_backtest
from forest.backtest.risk import RiskManager
from forest.utils.log import setup_logger

setup_logger("ERROR")  # wycisz silnik w interfejsie


# ---------------------------------------------------------------------------#
#  Pomocnicze                                                                 #
# ---------------------------------------------------------------------------#
def load_csv(file) -> pd.DataFrame:
    df = (
        pd.read_csv(file, parse_dates=["time"])
        .set_index("time")
        .loc[:, ["open", "high", "low", "close"]]
    )
    return df


def synthetic_metrics(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    equity = df["equity"]
    dd = (equity.cummax() - equity) / equity.cummax()
    return equity, dd


def show_equity(df: pd.DataFrame) -> None:
    equity, dd = synthetic_metrics(df)
    st.plotly_chart(px.line(equity, title="Equity curve"), use_container_width=True)
    st.area_chart(dd, height=160, use_container_width=True)


def heatmap_grid(df_grid: pd.DataFrame, dd_limit: float) -> None:
    # rozpakuj słownik params -> kolumny fast/slow
    params_df = df_grid["params"].apply(pd.Series)
    df = pd.concat([params_df, df_grid[["equity_end", "max_dd"]]], axis=1)
    df = df[df["max_dd"] <= dd_limit / 100.0]
    if df.empty:
        st.warning("Brak wyników po filtrze DD.")
        return

    pivot = (
        df.pivot(index="fast", columns="slow", values="equity_end")
        .sort_index(ascending=False)  # od góry rosnące fast
    )
    fig = px.imshow(
        pivot,
        text_auto=".2s",
        color_continuous_scale="YlGnBu",
        title=f"Heat‑mapa Equity (DD ≤ {dd_limit} %)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------#
#  Interfejs Streamlit                                                       #
# ---------------------------------------------------------------------------#
def app() -> None:
    st.set_page_config(layout="wide")
    tab_bt, tab_grid = st.tabs(["📈 Back‑test", "🌡️ Grid Heat‑map"])

    # -------- Back‑test pojedynczy ----------------------------------------
    with tab_bt:
        st.header("📈 Back‑test pojedynczy")
        file = st.file_uploader("→ Wrzuć CSV z danymi OHLC", type="csv")
        if file:
            df = load_csv(file)
            st.success(f"Wczytano {len(df):,} świec")

            fast = st.slider("EMA fast", 5, 50, 10, step=1)
            slow = st.slider("EMA slow", 20, 100, 30, step=5)
            if slow <= fast:
                st.error("slow musi być > fast")
            else:
                if st.button("Uruchom back‑test"):
                    rm = RiskManager(capital=10_000)
                    results = run_backtest(df, rm, fast, slow)
                    show_equity(results)
                    st.download_button(
                        "Pobierz equity CSV",
                        results[["equity"]].to_csv().encode(),
                        file_name="equity.csv",
                        mime="text/csv",
                    )

    # -------- Heat‑mapa ParamGrid ----------------------------------------
    with tab_grid:
        st.header("🌡️ Heat‑mapa wyników Grid‑runnera")
        gfile = st.file_uploader("Wynik grid_results (parquet/csv)", type=["parquet", "csv"])
        if gfile:
            if gfile.name.endswith(".parquet"):
                grid_df = pd.read_parquet(gfile)
            else:
                grid_df = pd.read_csv(gfile)
            dd_lim = st.slider("Max DD % filtrowania", 0, 50, 20, step=1)
            heatmap_grid(grid_df, dd_lim)


# ---------------------------------------------------------------------------#
#  CLI alias                                                                 #
# ---------------------------------------------------------------------------#
def main() -> None:
    """Uruchom Streamlit dla tego pliku (wykorzystuje alias Poetry)."""
    cmd = ["streamlit", "run", os.fspath(Path(__file__).resolve())]
    subprocess.run(cmd + sys.argv[1:], check=False)


if __name__ == "__main__":
    main()

