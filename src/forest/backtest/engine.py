"""Wektorowy back‑tester strategii EMA‑cross (lub opcjonalnie ML‑modelu)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from forest.backtest.risk import RiskManager
from forest.backtest.trace import DecisionTrace
from forest.backtest.tradebook import Trade, TradeBook
from forest.core.indicators import atr, ema_cross_strategy
from forest.utils.log import log

# --------------------------------------------------------------------------- #
#  Typ opcjonalnego modelu ML – pozwala zachować typowanie bez extras[ml].
# --------------------------------------------------------------------------- #
if TYPE_CHECKING:  # pragma: no cover
    from forest.ml.infer import ONNXModel  # noqa: N811 – import tylko dla Mypy
else:
    ONNXModel = Any  # type: ignore

# --------------------------------------------------------------------------- #
#  Pomocnicza funkcja zamykająca pozycję (DRY).
# --------------------------------------------------------------------------- #


def _close_open_position(
    tb: TradeBook,
    risk: RiskManager,
    when: pd.Timestamp,
    price: float,
    side: int,
    qty: float,
    entry_price: float,
) -> None:
    pnl = (price - entry_price) * side * qty
    cost = risk.position_cost(qty, price)
    risk.record_trade(pnl - cost)
    tb.add(Trade(when, price, qty, "LONG" if side == 1 else "SHORT"))


# --------------------------------------------------------------------------- #
#  Główna pętla back‑testera.
# --------------------------------------------------------------------------- #


def run_backtest(
    df: pd.DataFrame,
    risk: RiskManager,
    model: "ONNXModel | None" = None,  # noqa: N803 – opcjonalny model
    ml_threshold: float = 0.55,
) -> pd.DataFrame:
    """Back‑test strategii: EMA‑cross lub (jeśli podano) modelu ML."""
    out = df.copy()

    # ───────────────── 1) SYGNAŁ STRATEGII ──────────────────────────────────
    if model is not None:
        # lazy‑import, żeby core‑CI działało bez extras[ml]
        if TYPE_CHECKING:  # pragma: no cover
            from forest.strategy.ml_runner import ml_signal
        else:  # pragma: no cover
            from forest.strategy.ml_runner import ml_signal  # type: ignore

        from forest.strategy.features import build_features  # lokalny import

        feats = build_features(df)  # type: ignore[arg-type]
        out["signal"] = ml_signal(model, feats, threshold=ml_threshold)
    else:
        out["signal"] = ema_cross_strategy(df)

    # ───────────────── 2) ATR do sizingu ────────────────────────────────────
    out["atr"] = atr(df["high"], df["low"], df["close"], period=14)

    # ───────────────── 3) Pętla zdarzeń tick‑po‑tick ───────────────────────
    tb = TradeBook()
    position: int | None = None
    entry_price: float | None = None
    entry_qty: float | None = None

    for idx, row in out.iterrows():
        sig = int(row.signal)

        # trailing‑SL ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
        if position is not None:
            risk.update_trailing_sl(row.close, row.atr)
            if risk.hit_trailing_sl(row.close):
                _close_open_position(tb, risk, idx, row.close, position, entry_qty, entry_price)
                log.warning("trailing_sl_hit", time=str(idx), price=row.close)
                position = entry_price = entry_qty = None
                continue  # kontynuujemy test dalej (nie przerywamy)

        # zmiana kierunku / otwarcie pozycji ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
        if sig != 0 and sig != position:
            # zamknięcie starej (jeśli istniała)
            if position is not None:
                _close_open_position(tb, risk, idx, row.close, position, entry_qty, entry_price)

            # otwarcie nowej
            qty = risk.position_size(row.atr)
            if qty == 0:  # ATR zbyt duży ⇒ pomijamy sygnał
                continue

            position = sig
            entry_price = row.close
            entry_qty = qty
            tb.add(Trade(idx, row.close, qty, "LONG" if sig == 1 else "SHORT"))

        # log ścieżki decyzyjnej (debug) ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
        trace = DecisionTrace(
            time=str(idx),
            symbol="SYN",
            filters={"atr_ok": row.atr > 0},
            final={1: "BUY", -1: "SELL"}.get(sig, "WAIT"),
        )
        log.info("decision", **trace.as_dict())  # type: ignore[arg-type]

    # jeżeli coś nadal otwarte – zamyka się na ostatniej świecy
    if position is not None:
        _close_open_position(
            tb, risk, out.index[-1], out.close.iat[-1], position, entry_qty, entry_price
        )

    # ───────────────── 4) Equity curve (bez duplikatów) ─────────────────────
    ec = tb.equity_curve()
    ec = ec[~ec.index.duplicated(keep="last")]
    out["equity"] = ec.reindex(out.index).ffill()

    return out

