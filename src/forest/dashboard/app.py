from __future__ import annotations
import io
import pandas as pd
import plotly.express as px
import streamlit as st

from forest.config import BacktestSettings
from forest.backtest.engine import run_backtest
from forest.backtest.grid import run_grid


st.set_page_config(page_title="Forest 4.0 Dashboard", layout="wide")
st.title("🌲 Forest 4.0 — Backtest & Grid")

def load_csv(uploaded) -> pd.DataFrame:
    df = pd.read_csv(uploaded)
    # heurystyka kolumn, oczekujemy: time, open, high, low, close [, volume]
    time_col = "time" if "time" in df.columns else ("Date" if "Date" in df.columns else None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col).sort_index()
    # ujednolicenie małych liter
    ren = {c: c.lower() for c in df.columns}
    df = df.rename(columns=ren)
    return df

def draw_equity(equity: pd.Series):
    fig = px.line(equity.reset_index(), x=equity.index.name or "index", y="equity", title="Equity")
    st.plotly_chart(fig, use_container_width=True)

def draw_drawdown(equity: pd.Series):
    dd = (equity / equity.cummax()) - 1.0
    fig = px.area(dd.reset_index(), x=dd.index.name or "index", y=0, title="Drawdown")
    st.plotly_chart(fig, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Back-test", "Grid Runner", "Grid Heat-map"])

with tab1:
    st.header("Pojedynczy back-test")
    up = st.file_uploader("Wgraj CSV (OHLC)", type=["csv"])
    fast = st.number_input("EMA fast", value=12, min_value=2, max_value=500, step=1)
    slow = st.number_input("EMA slow", value=26, min_value=3, max_value=1000, step=1)
    run = st.button("Run back-test")
    if up and run:
        df = load_csv(up)
        cfg = BacktestSettings()
        cfg.strategy.params = {"fast": int(fast), "slow": int(slow)}
        strat = cfg.build_strategy()
        res = run_backtest(df=df, strategy=strat, symbol=cfg.symbol, price_col=cfg.strategy.price_col,
                           atr_period=cfg.atr_period, atr_multiple=cfg.atr_multiple)
        c1, c2 = st.columns(2)
        with c1:
            draw_equity(res.equity)
        with c2:
            draw_drawdown(res.equity)
        st.subheader("Metryki")
        st.json(res.metrics)
        st.subheader("Transakcje (head)")
        st.dataframe(res.trades.head(20))

with tab2:
    st.header("Grid Runner")
    upg = st.file_uploader("Wgraj CSV (OHLC) do grida", type=["csv"], key="grid_csv")
    c1, c2 = st.columns(2)
    fast_min = c1.number_input("fast min", value=5)
    fast_max = c1.number_input("fast max", value=30)
    fast_step = c1.number_input("fast step", value=5)
    slow_min = c2.number_input("slow min", value=30)
    slow_max = c2.number_input("slow max", value=120)
    slow_step = c2.number_input("slow step", value=10)
    run_g = st.button("Run grid")
    if upg and run_g:
        df = load_csv(upg)
        cfg = BacktestSettings()
        ranges = {
            "fast": range(int(fast_min), int(fast_max) + 1, int(fast_step)),
            "slow": range(int(slow_min), int(slow_max) + 1, int(slow_step)),
        }
        results = run_grid(df, cfg, ranges)
        st.success(f"Grid done: {len(results)} kombinacji.")
        st.dataframe(results.head(50))
        st.session_state["latest_grid"] = results

with tab3:
    st.header("Heat-map")
    src = st.radio("Źródło wyników", options=["Z ostatniego grida", "Wgraj plik CSV"])
    df_res = None
    if src == "Z ostatniego grida" and "latest_grid" in st.session_state:
        df_res = st.session_state["latest_grid"]
    else:
        uph = st.file_uploader("Wgraj wyniki grida (CSV)", type=["csv"], key="heat_csv")
        if uph:
            df_res = pd.read_csv(uph)

    metric = st.selectbox("Metryka", options=["equity_end", "max_dd", "cagr", "rar", "sharpe"])
    if df_res is not None and {"fast", "slow", metric}.issubset(df_res.columns):
        pivot = df_res.pivot_table(index="slow", columns="fast", values=metric, aggfunc="mean")
        fig = px.imshow(pivot, origin="lower", aspect="auto", title=f"Heatmap: {metric}")
        st.plotly_chart(fig, use_container_width=True)
    elif df_res is not None:
        st.warning("Wyniki nie zawierają kolumn fast/slow/wybranej metryki.")

