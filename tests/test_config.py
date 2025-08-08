from __future__ import annotations

from textwrap import dedent

from forest.config import BacktestSettings, RiskSettings, StrategySettings


def test_config_from_yaml(tmp_path):
    yaml_text = dedent(
        """
        symbol: EURUSD
        timeframe: H
        seed: 123
        risk:
          capital: 20000
          risk_per_trade: 0.02
          max_drawdown: 0.15
          fee_perc: 0.0001
          slippage: 0.0002
          atr_multiple: 2.0
        strategy:
          mode: classic
          fast: 10
          slow: 30
          atr_period: 14
        """
    ).strip()

    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml_text, encoding="utf-8")

    cfg = BacktestSettings.from_file(cfg_file)

    assert cfg.symbol == "EURUSD"
    assert cfg.timeframe == "1h"  # 'H' → '1h'
    assert cfg.seed == 123
    assert isinstance(cfg.risk, RiskSettings)
    assert cfg.risk.capital == 20000
    assert cfg.risk.max_drawdown == 0.15
    assert isinstance(cfg.strategy, StrategySettings)
    assert cfg.strategy.fast == 10
    assert cfg.strategy.slow == 30

