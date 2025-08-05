import pandas as pd

from forest.backtest.tradebook import Trade, TradeBook


def test_tradebook_pnl_and_dd():
    tb = TradeBook()
    # syntetyczna sekwencja: kupno → sprzedaż → kupno
    tb.add(Trade(pd.Timestamp("2025-08-05 10:00"), 100.0,  1, "LONG"))   # +100
    tb.add(Trade(pd.Timestamp("2025-08-05 12:00"),  95.0,  1, "SHORT"))  # -95  → saldo +5
    tb.add(Trade(pd.Timestamp("2025-08-05 13:00"), 110.0,  2, "LONG"))   # +220 → saldo +225

    eq = tb.equity_curve()
    assert list(eq) == [100.0, 5.0, 225.0]

    dd = tb.max_drawdown()
    assert dd == 95.0            # największe obsunięcie między 100 a 5

