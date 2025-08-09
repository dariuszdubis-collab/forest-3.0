from __future__ import annotations

import logging
import sys
from typing import Any, Optional

import structlog

__all__ = ["setup_logger", "get_logger", "log", "logger"]

_CONFIGURED = False


def _to_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    try:
        return getattr(logging, str(level).upper())
    except AttributeError:
        return logging.INFO


def setup_logger(level: str | int = "INFO", json: bool = False):
    """
    Idempotentna konfiguracja structlog + stdlib logging.
    Zwraca BoundLogger; można wywoływać wielokrotnie bez skutków ubocznych.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        # stdlib logging -> minimalne ustawienia na stdout
        logging.basicConfig(
            level=_to_level(level),
            format="%(message)s",
            stream=sys.stdout,
        )

        processors: list[Any] = [
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        if json:
            processors.append(structlog.processors.JSONRenderer())
        else:
            # Czytelny renderer do dev/testów
            processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(_to_level(level)),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        _CONFIGURED = True

    return structlog.get_logger("forest")


def get_logger(name: Optional[str] = None):
    """
    Pobiera BoundLogger. Gwarantuje, że konfiguracja istnieje.
    """
    if not _CONFIGURED:
        setup_logger()
    return structlog.get_logger(name) if name else structlog.get_logger()


# Domyślny logger eksportowany dla wygody importów:
#   from forest.utils.log import log
log = setup_logger()
# Alias zgodny z niektórymi stylami:
logger = log

