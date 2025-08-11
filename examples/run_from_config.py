import pandas as pd

from forest.config import BacktestSettings
from forest.backtest.engine import run_backtest
from forest.backtest.risk import RiskManager

# 1) Wczytaj parametry z config.yaml (z głównego katalogu repo)
CFG_PATH = "config.yaml"  # lub "configs/default.yaml"
cfg = BacktestSettings.from_file(CFG_PATH)
strategy = cfg.build_strategy()

# 2) Wczytaj dane z CSV (kolumny: time, open, high, low, close [, volume])
#    Przykładowo oczekujemy pliku: /<REPO_ROOT>/data/prices.csv
DATA_CSV = "data/prices.csv"
df = pd.read_csv(DATA_CSV, parse_dates=["time"]).set_index("time")
df.columns = [c.lower() for c in df.columns]  # ujednolicamy nazwy kolumn

# 3) Zbuduj RiskManager z configu (żeby użyć initial_capital / fee / itp.)
risk = RiskManager(**cfg.risk.model_dump())

# 4) Odpal backtest
res = run_backtest(
    df=df,
    strategy=strategy,
    symbol=cfg.symbol,
    price_col=cfg.strategy.price_col,  # "close", jeśli nie nadpisano w YAML
    atr_period=cfg.atr_period,
    atr_multiple=cfg.atr_multiple,
    risk=risk,
)

print("=== METRYKI ===")
for k, v in res.metrics.items():
    print(f"{k}: {v}")

print("\n=== TRADES (head) ===")
print(res.trades.head())

# zapis equity do CSV (opcjonalnie)
res.equity.to_frame().to_csv("data/equity_out.csv")
print("\nZapisano equity do data/equity_out.csv")

