from __future__ import annotations

import logging

import structlog
from rich.console import Console
from rich.logging import RichHandler


def setup_logger(level: str = "INFO") -> None:
    """Globalna konfiguracja logera Forest 3.0.

    Po wywołaniu `setup_logger()` wszystkie logi `structlog.get_logger()`
    trafiają na Rich‑konsolę (czytelne w CLI i CI) + format JSON, więc
    łatwo je przekierować do pliku.
    """
    logging.basicConfig(
        level=level,
        handlers=[RichHandler(console=Console(), markup=True)],
        format="%(message)s",
    )
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )

