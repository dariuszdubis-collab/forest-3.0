import logging

import structlog
from rich.console import Console
from rich.logging import RichHandler


def setup_logger(level: str = "INFO") -> None:
    """Konfiguracja globalna logera Forest 3.0."""
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
