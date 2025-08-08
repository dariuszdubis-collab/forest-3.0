from __future__ import annotations

import math

from forest.broker import PaperBroker, Side


def test_paper_broker_open_close_long():
    br = PaperBroker(initial_balance=10_000)
    br.update_price("SYN", 100.0)
    br.market_order("SYN", Side.BUY, qty=1.0)

    # wzrost ceny -> zysk niezrealizowany = +2
    br.update_price("SYN", 102.0)
    assert math.isclose(br.equity(), 10_002.0, rel_tol=1e-9)

    # zamknięcie pozycji -> realizacja zysku
    out = br.close_position("SYN")
    assert math.isclose(out.realized_pnl, 2.0, rel_tol=1e-9)
    assert math.isclose(br.balance(), 10_002.0, rel_tol=1e-9)
    assert br.positions() == {}


def test_paper_broker_avg_price_and_flip():
    br = PaperBroker()
    br.update_price("X", 100.0)
    br.market_order("X", Side.BUY, qty=1.0)

    # dokładamy LONG -> średnia 105
    br.update_price("X", 110.0)
    br.market_order("X", Side.BUY, qty=1.0)
    pos = br.positions()["X"]
    assert math.isclose(pos.entry, 105.0, rel_tol=1e-9)
    assert math.isclose(pos.qty, 2.0, rel_tol=1e-9)

    # flip na SHORT -> zamknięcie LONG po 110, realized = (110-105)*2 = +10
    out = br.market_order("X", Side.SELL, qty=1.0)
    assert math.isclose(out.realized_pnl, 10.0, rel_tol=1e-9)
    assert br.positions()["X"].side == Side.SELL

