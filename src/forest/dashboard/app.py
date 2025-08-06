"""Streamlit dashboard – v1.1 (Equity, Max DD, RAR heat‑maps)."""

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

setup_logger("ERROR")  # wyciszamy silnik w UI


# ---------------------------------------------------------------------------#
#  Helpers                                                                   #
# ---------------------------------------------------------------------------#
def load_csv(file) -> pd.DataFrame:
    df = (
        pd.read_csv(file, parse_dates=["time"])
        .set_index("time")
        .loc[:, ["open", "high", "low", "close"]]
    )
    return df


def metrics(equity: pd.Series) -> tuple[pd.Series, pd.Series]:
    drawdown = (equity.cummax() - equity) / equity.cummax()
    return equity, drawdown


def heatmap_grid(df_grid: pd.DataFrame, metric: str, dd_limit: int) -> None:
    """Rysuje heat‑mapę equity_end, max_dd lub RAR."""
    params_df = df_grid["params"].apply(pd.Series)
    df = pd.concat([params_df, df_grid[["equity_end", "max_dd", "rar"]]], axis=1)

    if metric == "equity_end":
        df = df[df["max_dd"] <= dd_limit / 100.0]

    if df.empty:
        st.warning("Brak wyników po filtrze.")
        return

    value_col = {"equity_end": "equity_end", "max_dd": "max_dd", "rar": "rar"}[metric]
    z_text = ".2s" if metric in ("equity_end", "rar") else ".2%"
    cmap = {"equity_end": "YlGnBu", "max_dd": "RdYlGn_r", "rar": "PuBuGn"}[metric]

    pivot = (
        df.pivot(index="fast", columns="slow", values=value_col)
        .sort_index(ascending=False)
    )

    title = {
        "equity_end": f"Equity (DD ≤ {dd_limit} %)",
        "max_dd": "Max DD %",
        "rar": "Risk‑adj. Return (CAGR / DD)",
    }[metric]

    fig = px.imshow(
        pivot, text_auto=z_text, color_continuous_scale=cmap, title=title
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------#
#  UI                                                                        #
# ---------------------------------------------------------------------------#
def app() -> None:
    st.set_page_config(layout="wide")
    tab_bt, tab_grid = st.tabs(["📈 Back‑test", "🌡️ Grid Heat‑map"])

    # -------- Back‑test pojedynczy ----------------------------------------
    with tab_bt:
        st.header("📈 Back‑test pojedynczy")
        file = st.file_uploader("→ Wrzuć CSV z danymi OHLC", type="csv")
        if file:
            df = load_csv(file)
            st.success(f"Wczytano {len(df):,} świec")

            fast = st.slider("EMA fast", 5, 50, 10, 1)
            slow = st.slider("EMA slow", 20, 100, 30, 5)
            if slow <= fast:
                st.error("slow musi być > fast")
            else:
                if st.button("Uruchom back‑test"):
                    rm = RiskManager(capital=10_000)
                    results = run_backtest(df, rm, fast, slow)
                    eq, dd = metrics(results["equity"])
                    st.plotly_chart(px.line(eq, title="Equity curve"), use_container_width=True)
                    st.area_chart(dd, height=160, use_container_width=True)
                    st.download_button(
                        "Pobierz equity CSV",
                        results[["equity"]].to_csv().encode(),
                        file_name="equity.csv",
                        mime="text/csv",
                    )

    # -------- Heat‑mapa ParamGrid ----------------------------------------
    with tab_grid:
        st.header("🌡️ Heat‑mapa wyników ParamGrid")
        gfile = st.file_uploader("Plik grid_results (.parquet / .csv)", type=["parquet", "csv"])
        if gfile:
            grid_df = pd.read_parquet(gfile) if gfile.name.endswith(".parquet") else pd.read_csv(gfile)
            metric = st.radio("Metryka", ["equity_end", "max_dd", "rar"], horizontal=True)
            dd_lim = st.slider("Filtr Max DD %", 0, 50, 20, 1) if metric == "equity_end" else 100
            heatmap_grid(grid_df, metric, dd_lim)


# ---------------------------------------------------------------------------#
#  CLI alias                                                                 #
# ---------------------------------------------------------------------------#
def main() -> None:
    cmd = ["streamlit", "run", os.fspath(Path(__file__).resolve())]
    subprocess.run(cmd + sys.argv[1:], check=False)


if __name__ == "__main__":
    main()

