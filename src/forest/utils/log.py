"""Centralny, lekki wrapper na `structlog`.

- `setup_logger()`  – konfiguruje format, poziom i opcjonalnie JSON‑formatter
- `log`             – gotowy, globalny logger używany w całym projekcie
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

# ———————————————————————————————————————————————————————————————————— #
# Konfiguracja bazowa
# ———————————————————————————————————————————————————————————————————— #
_LOGGER_NAME = "forest"

def _build_processor_chain(json: bool) -> list[Any]:
    """Zwraca łańcuch processorów w zależności od formatu wyjściowego."""
    common: list[Any] = [
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.add_log_level,
    ]
    if json:
        common.append(structlog.processors.JSONRenderer())
    else:
        common.append(structlog.dev.ConsoleRenderer(colors=True))
    return common


def setup_logger(level: int = logging.INFO, *, json: bool = False) -> structlog.BoundLogger:
    """Inicjalizuje i zwraca logger.

    Parameters
    ----------
    level : int
        Poziom logowania (`logging.INFO`, `logging.DEBUG`, …).
    json : bool, default False
        Jeśli *True* – logi w formacie JSON (np. pod ELK/Grafanę).

    Returns
    -------
    structlog.BoundLogger
        Gotowy logger.
    """
    logging.basicConfig(level=level, format="%(message)s", force=True)

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=_build_processor_chain(json),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    return structlog.get_logger(_LOGGER_NAME)


# Utworzenie globalnego „domyślnego” loggera wykorzystywanego w całym kodzie
log: structlog.BoundLogger = setup_logger()

