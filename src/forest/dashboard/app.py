"""Streamlit dashboard – v2
Zakładki:
  • Back‑test pojedynczy
  • Grid Runner (uruchom siatkę)
  • Grid Heat‑map (Equity / DD / RAR)
"""

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

setup_logger("ERROR")  # wyciszamy silnik w UI


# ---------------------------------------------------------------------------#
#  Helpers                                                                   #
# ---------------------------------------------------------------------------#
def load_csv(file) -> pd.DataFrame:
    return (
        pd.read_csv(file, parse_dates=["time"])
        .set_index("time")
        .loc[:, ["open", "high", "low", "close"]]
    )


def metrics(equity: pd.Series) -> tuple[pd.Series, pd.Series]:
    dd = (equity.cummax() - equity) / equity.cummax()
    return equity, dd


def heatmap_grid(df_grid: pd.DataFrame, metric: str, dd_limit: int) -> None:
    params_df = df_grid["params"].apply(pd.Series)
    df = pd.concat([params_df, df_grid[["equity_end", "max_dd", "rar"]]], axis=1)

    if metric == "equity_end":
        df = df[df["max_dd"] <= dd_limit / 100.0]

    if df.empty:
        st.warning("Brak wyników do wyświetlenia.")
        return

    value_col = {"equity_end": "equity_end", "max_dd": "max_dd", "rar": "rar"}[metric]
    cmap = {"equity_end": "YlGnBu", "max_dd": "RdYlGn_r", "rar": "PuBuGn"}[metric]
    z_txt = ".2s" if metric in ("equity_end", "rar") else ".2%"

    pivot = (
        df.pivot(index="fast", columns="slow", values=value_col)
        .sort_index(ascending=False)
    )

    title = {
        "equity_end": f"Equity (DD ≤ {dd_limit} %)",
        "max_dd": "Max DD %",
        "rar": "Risk‑adj. Return",
    }[metric]

    fig = px.imshow(pivot, text_auto=z_txt, color_continuous_scale=cmap, title=title)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------#
#  UI                                                                        #
# ---------------------------------------------------------------------------#
def app() -> None:
    st.set_page_config(layout="wide")
    tab_bt, tab_runner, tab_grid = st.tabs(
        ["📈 Back‑test", "⚙️ Grid Runner", "🌡️ Grid Heat‑map"]
    )

    # ---------------------- Back‑test pojedynczy --------------------------
    with tab_bt:
        st.header("📈 Back‑test pojedynczy")
        file = st.file_uploader("Wrzuć CSV OHLC", type="csv", key="bt_csv")
        if file:
            df = load_csv(file)
            st.success(f"Wczytano {len(df):,} świec")

            fast = st.slider("EMA fast", 5, 50, 10, 1)
            slow = st.slider("EMA slow", 20, 100, 30, 5)
            if slow <= fast:
                st.error("slow musi być > fast")
            elif st.button("Uruchom back‑test"):
                rm = RiskManager(capital=10_000)
                res = run_backtest(df, rm, fast, slow)
                eq, dd = metrics(res["equity"])
                st.plotly_chart(px.line(eq, title="Equity curve"), use_container_width=True)
                st.area_chart(dd, height=180, use_container_width=True)
                st.download_button(
                    "Pobierz equity CSV",
                    res[["equity"]].to_csv().encode(),
                    file_name="equity.csv",
                    mime="text/csv",
                )

    # ---------------------- Grid Runner -----------------------------------
    with tab_runner:
        st.header("⚙️ ParamGrid Runner")
        gfile = st.file_uploader("CSV OHLC (dla Grid)", type="csv", key="runner_csv")
        if gfile:
            df_grid_src = load_csv(gfile)
            st.info(f"Zakres danych: {df_grid_src.index[0]} – {df_grid_src.index[-1]}")

            c1, c2, c3 = st.columns(3)
            with c1:
                fast_min = st.number_input("fast min", 5, 200, 5, 1)
                fast_max = st.number_input("fast max", fast_min + 1, 400, 25, 1)
                fast_step = st.number_input("fast step", 1, 100, 5, 1)
            with c2:
                slow_min = st.number_input("slow min", fast_min + 1, 400, 20, 1)
                slow_max = st.number_input("slow max", slow_min + 1, 600, 60, 1)
                slow_step = st.number_input("slow step", 1, 100, 10, 1)
            with c3:
                n_jobs = st.number_input("CPU (‑1=all)", -1, 32, -1, 1)
                use_cache = st.checkbox("Use cache", True)

            grid_size = (
                (fast_max - fast_min) // fast_step + 1
            ) * (
                (slow_max - slow_min) // slow_step + 1
            )
            st.write(f"**Kombinacji = {grid_size}**")

            if st.button("▶ Run grid"):
                st.session_state["grid_running"] = True
                with st.spinner("Uruchamiam symulację… to może potrwać"):
                    grid = param_grid(
                        fast=range(fast_min, fast_max + 1, fast_step),
                        slow=range(slow_min, slow_max + 1, slow_step),
                    )
                    results = run_grid(
                        df_grid_src,
                        grid,
                        make_risk=lambda: RiskManager(capital=10_000),
                        n_jobs=n_jobs,
                        use_cache=use_cache,
                    )
                    out_dir = Path("results")
                    out_dir.mkdir(exist_ok=True)
                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    out_path = out_dir / f"grid_{ts}.parquet"
                    results.to_parquet(out_path, index=False)
                    st.success(f"Grid DONE ✓ – zapisano {out_path}")
                    st.session_state["last_grid"] = results
                st.session_state["grid_running"] = False

            # Auto‑display heat‑map po zakończeniu
            if "last_grid" in st.session_state and not st.session_state.get(
                "grid_running", False
            ):
                st.subheader("Heat‑mapa (Equity)")
                heatmap_grid(st.session_state["last_grid"], "equity_end", 20)
                st.download_button(
                    "Pobierz wyniki Parquet",
                    st.session_state["last_grid"].to_parquet(),
                    file_name="grid_results.parquet",
                    mime="application/octet-stream",
                )

    # ---------------------- Heat‑mapa ParamGrid ---------------------------
    with tab_grid:
        st.header("🌡️ Heat‑mapa istniejących wyników")
        gfile2 = st.file_uploader("Plik grid_results (.parquet / .csv)", type=["parquet", "csv"])
        if gfile2:
            gdf = pd.read_parquet(gfile2) if gfile2.name.endswith(".parquet") else pd.read_csv(gfile2)
            metric = st.radio("Metryka", ["equity_end", "max_dd", "rar"], horizontal=True)
            dd_lim = st.slider("Filtr Max DD %", 0, 50, 20, 1) if metric == "equity_end" else 100
            heatmap_grid(gdf, metric, dd_lim)


# ---------------------------------------------------------------------------#
#  CLI alias                                                                 #
# ---------------------------------------------------------------------------#
def main() -> None:
    cmd = ["streamlit", "run", os.fspath(Path(__file__).resolve())]
    subprocess.run(cmd + sys.argv[1:], check=False)


if __name__ == "__main__":
    main()

