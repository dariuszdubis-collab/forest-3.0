from __future__ import annotations


class ForestError(Exception):
    """Bazowy wyjątek dla projektu."""


class DataValidationError(ForestError):
    """Problemy z wejściowymi danymi/ramką."""


class BacktestConfigError(ForestError):
    """Nieprawidłowa konfiguracja backtestu."""

