from forest.backtest.risk import RiskManager


def test_position_cost():
    rm = RiskManager(capital=10_000)
    cost = rm.position_cost(qty=1.0, price=100.0,
                            spread=0.001, commission=0.0, slippage=0.0)
    assert cost == 0.1          # 0.1% * 100 = 0.1

