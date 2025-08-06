"""Streamlit dashboard – v2.1  (Equity / DD / CAGR/DD / Sharpe heat‑map)."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from forest.backtest.engine import run_backtest
from forest.backtest.grid import param_grid, run_grid
from forest.backtest.risk import RiskManager
from forest.utils.log import setup_logger

setup_logger("ERROR")

# ---------- helpers --------------------------------------------------------
def load_csv(file):  # small wrapper
    return (
        pd.read_csv(file, parse_dates=["time"])
        .set_index("time")
        .loc[:, ["open", "high", "low", "close"]]
    )

def metrics(eq: pd.Series):
    dd = (eq.cummax() - eq) / eq.cummax()
    return eq, dd

def heatmap(df_grid: pd.DataFrame, metric: str, dd_lim: int):
    params_df = df_grid["params"].apply(pd.Series)
    df = pd.concat([params_df, df_grid[["equity_end", "max_dd", "rar", "sharpe"]]], axis=1)

    if metric == "equity_end":
        df = df[df["max_dd"] <= dd_lim / 100]

    if df.empty:
        st.warning("Brak danych do wyświetlenia.")
        return

    value = {
        "equity_end": "equity_end",
        "max_dd": "max_dd",
        "rar": "rar",
        "sharpe": "sharpe",
    }[metric]
    cmap = {
        "equity_end": "YlGnBu",
        "max_dd": "RdYlGn_r",
        "rar": "PuBuGn",
        "sharpe": "Blues",
    }[metric]
    txt_fmt = ".2s" if metric in ("equity_end", "rar") else ".2f"

    pivot = (
        df.pivot(index="fast", columns="slow", values=value)
        .sort_index(ascending=False)
    )
    title = {
        "equity_end": f"Equity (DD ≤ {dd_lim} %)",
        "max_dd": "Max DD %",
        "rar": "CAGR / DD",
        "sharpe": "Sharpe ratio",
    }[metric]

    st.plotly_chart(
        px.imshow(pivot, text_auto=txt_fmt, color_continuous_scale=cmap, title=title),
        use_container_width=True,
    )

# ---------- Streamlit UI ---------------------------------------------------
def app() -> None:
    st.set_page_config(layout="wide")
    tab_bt, tab_runner, tab_grid = st.tabs(
        ["📈 Back‑test", "⚙️ Grid Runner", "🌡️ Grid Heat‑map"]
    )

    # 1. pojedynczy back‑test ----------------------------------------------
    with tab_bt:
        st.header("📈 Back‑test pojedynczy")
        f = st.file_uploader("CSV OHLC", type="csv", key="bt")
        if f:
            df = load_csv(f)
            fast = st.slider("EMA fast", 5, 50, 10)
            slow = st.slider("EMA slow", 20, 100, 30, 5)
            if slow <= fast:
                st.error("slow musi być > fast")
            elif st.button("Run back‑test"):
                rm = RiskManager(capital=10_000)
                res = run_backtest(df, rm, fast, slow)
                eq, dd = metrics(res["equity"])
                st.plotly_chart(px.line(eq, title="Equity"), use_container_width=True)
                st.area_chart(dd, height=160, use_container_width=True)

    # 2. Grid Runner --------------------------------------------------------
    with tab_runner:
        st.header("⚙️ Grid Runner")
        gfile = st.file_uploader("CSV OHLC", type="csv", key="runner")
        if gfile:
            df_src = load_csv(gfile)
            c1, c2, c3 = st.columns(3)
            with c1:
                f_min = st.number_input("fast min", 5, 200, 5)
                f_max = st.number_input("fast max", f_min + 1, 400, 25)
                f_step = st.number_input("fast step", 1, 100, 5)
            with c2:
                s_min = st.number_input("slow min", f_min + 1, 400, 20)
                s_max = st.number_input("slow max", s_min + 1, 600, 60)
                s_step = st.number_input("slow step", 1, 100, 10)
            with c3:
                n_jobs = st.number_input("CPU (-1=all)", -1, 32, -1)
                cache_on = st.checkbox("Use cache", True)

            total = ((f_max - f_min) // f_step + 1) * ((s_max - s_min) // s_step + 1)
            st.write(f"**Total combinations: {total}**")

            if st.button("▶ Run grid"):
                with st.spinner("Running grid…"):
                    grid = param_grid(
                        fast=range(f_min, f_max + 1, f_step),
                        slow=range(s_min, s_max + 1, s_step),
                    )
                    res = run_grid(
                        df_src,
                        grid,
                        make_risk=lambda: RiskManager(capital=10_000),
                        n_jobs=n_jobs,
                        use_cache=cache_on,
                    )
                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    out_path = Path("results") / f"grid_{ts}.parquet"
                    out_path.parent.mkdir(exist_ok=True)
                    res.to_parquet(out_path, index=False)
                    st.success(f"Finished ✓  – saved to {out_path}")
                    st.session_state["latest_grid"] = res

            if "latest_grid" in st.session_state:
                st.subheader("Heat‑mapa Equity")
                heatmap(st.session_state["latest_grid"], "equity_end", 20)
                st.download_button(
                    "Download results (parquet)",
                    st.session_state["latest_grid"].to_parquet(),
                    file_name="grid_results.parquet",
                )

    # 3. Heat‑map zakładka --------------------------------------------------
    with tab_grid:
        st.header("🌡️ Wczytaj istniejące wyniki")
        file2 = st.file_uploader("grid_results (.parquet / .csv)", type=["parquet", "csv"])
        if file2:
            gdf = pd.read_parquet(file2) if file2.name.endswith(".parquet") else pd.read_csv(file2)
            metric = st.radio("Metryka", ["equity_end", "max_dd", "rar", "sharpe"], horizontal=True)
            dd_lim = st.slider("Max DD % filter", 0, 50, 20) if metric == "equity_end" else 100
            heatmap(gdf, metric, dd_lim)

# ---------- CLI alias -------------------------------------------------------
def main():
    subprocess.run(["streamlit", "run", os.fspath(Path(__file__).resolve())] + sys.argv[1:], check=False)

if __name__ == "__main__":
    main()

