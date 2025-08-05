from forest.backtest.risk import RiskManager


def test_position_size():
    rm = RiskManager(capital=10_000, risk_per_trade=0.02)
    qty = rm.position_size(atr=1.5, atr_multiple=2.0)
    assert round(qty, 4) == round((10_000 * 0.02) / (1.5 * 2), 4)


def test_max_dd_guard():
    rm = RiskManager(capital=10_000, max_drawdown=0.10)

    # seria strat – 3 × −1 000 => equity 7 000 (30 % DD)
    for _ in range(3):
        rm.record_trade(-1_000)

    assert rm.equity == 7_000
    assert rm.exceeded_max_dd() is True
