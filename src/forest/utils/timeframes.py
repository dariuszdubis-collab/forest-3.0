"""
Pomocnicze funkcje do pracy z interwałami czasowymi (timeframes).

Obsługujemy formy:
- '1m', '3m', '5m', '15m', '30m'
- '1h', '4h'
- '1d' / 'd' / 'D'
Dodatkowo normalizujemy wielkość liter i usuwamy spacje.
"""

from __future__ import annotations

from typing import Final

# mapowanie skrótów na minuty
_TF_MINUTES: Final[dict[str, int]] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

_ALIASES: Final[dict[str, str]] = {
    "m": "1m",
    "h": "1h",
    "d": "1d",
    "D": "1d",
    "H": "1h",
    "M": "1m",
}


def normalize_timeframe(tf: str) -> str:
    """Znormalizuj zapis TF do postaci z sufiksem 'm'/'h'/'d' i cyfrą z przodu (np. '1h')."""
    raw = tf.strip()
    if raw in _ALIASES:
        raw = _ALIASES[raw]

    s = raw.lower().replace(" ", "")
    # jeśli użytkownik poda np. "H" albo "D", już zamienione powyżej
    if s in _TF_MINUTES:
        return s

    # warianty typu "60", "240" (minuty)
    if s.isdigit():
        minutes = int(s)
        if minutes in _TF_MINUTES.values():
            # odwrotne mapowanie
            for k, v in _TF_MINUTES.items():
                if v == minutes:
                    return k

    # warianty typu "1H", "15M"
    if s.endswith(("m", "h", "d")):
        num = s[:-1]
        unit = s[-1]
        if num.isdigit():
            candidate = f"{int(num)}{unit}"
            if candidate in _TF_MINUTES:
                return candidate

    raise ValueError(f"Nieobsługiwany timeframe: {tf!r}")


def to_minutes(tf: str) -> int:
    """Zwróć liczbę minut dla danego TF."""
    norm = normalize_timeframe(tf)
    return _TF_MINUTES[norm]

