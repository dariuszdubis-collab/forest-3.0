from __future__ import annotations

from forest.live import Order, PaperBroker


def test_paper_broker_basic_flow():
    brk = PaperBroker(initial_cash=10_000, fee_perc=0.0)
    brk.connect()
    brk.set_price("SYN", 100.0)

    # BUY 10 @ 100
    res1 = brk.market_order(Order(symbol="SYN", side="BUY", qty=10, price=100.0))
    assert res1.status == "filled"
    assert res1.filled_qty == 10
    assert brk.position_qty("SYN") == 10
    # cash spada do 9 000, wartość pozycji 1 000 -> equity = 10 000
    assert abs(brk.equity() - 10_000.0) < 1e-9

    # Mark-to-market na 110
    brk.set_price("SYN", 110.0)
    # equity: 9 000 + 10 * 110 = 10 100
    assert abs(brk.equity() - 10_100.0) < 1e-9

    # SELL 10 @ 110 => zamknięcie pozycji i zrealizowany zysk 100
    res2 = brk.market_order(Order(symbol="SYN", side="SELL", qty=10, price=110.0))
    assert res2.status == "filled"
    assert brk.position_qty("SYN") == 0
    assert abs(brk.equity() - 10_100.0) < 1e-9  # wszystko w cashu


def test_rejects_without_price_or_connection():
    brk = PaperBroker(initial_cash=1_000)
    # brak connect()
    rej1 = brk.market_order(Order(symbol="SYN", side="BUY", qty=1, price=100.0))
    assert rej1.status == "rejected"

    brk.connect()
    # brak ceny (ani w orderze, ani w last_price)
    rej2 = brk.market_order(Order(symbol="SYN", side="BUY", qty=1))
    assert rej2.status == "rejected"

