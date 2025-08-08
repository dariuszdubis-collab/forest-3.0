from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from forest.utils.timeframes import to_minutes


class CSVConfig(BaseModel):
    """Konfiguracja odczytu CSV z danymi OHLC."""

    path: Path = Field(..., description="Ścieżka do pliku CSV.")
    symbol: str = Field("SYN", description="Symbol instrumentu (metadane).")
    timeframe: str = Field("1h", description="TF wejściowych danych (np. '1h', '15m').")
    tz: str | None = Field("UTC", description="Strefa czasowa indeksu po wczytaniu.")
    time_col: str = Field("time", description="Nazwa kolumny z czasem.")
    sep: str = Field(",", description="Separator CSV.")

    @field_validator("timeframe")
    @classmethod
    def _validate_tf(cls, v: str) -> str:
        # Walidacja: rzuci ValueError dla niepoprawnych wartości
        _ = to_minutes(v)
        return v


def _standardize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Sprowadź nazwy kolumn do open/high/low/close (+volume jeżeli jest)."""
    df = df.rename(columns=str.lower)

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns: open, high, low, close "
            f"(got: {sorted(df.columns)})"
        )
    return df


def _parse_time_index(df: pd.DataFrame, time_col: str, tz: str | None) -> pd.DataFrame:
    if time_col not in df.columns:
        raise ValueError(f"CSV missing time column: {time_col!r}")

    # Używamy utc=True, aby zawsze mieć tz-aware indeks
    idx = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.drop(columns=[time_col]).assign(_time=idx).dropna(subset=["_time"])
    df = df.set_index("_time").sort_index()

    if tz and tz.upper() != "UTC":
        # konwersja strefy (indeks jest już w UTC)
        df.index = df.index.tz_convert(tz)

    return df


def _maybe_resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Jeśli trzeba, przelicza do zadanego TF. Używamy simple-OHLC aggregacji."""
    rule = f"{to_minutes(timeframe)}min"  # np. '60min' dla '1h'

    if df.index.inferred_type not in {"datetime64", "datetime64tz"}:
        raise ValueError("Index must be a DateTimeIndex to resample.")

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"

    out = df.resample(rule).agg(agg).dropna(how="any")
    return out


def load_history_csv(cfg: CSVConfig) -> pd.DataFrame:
    """Wczytaj świece z CSV → DataFrame z kolumnami: open, high, low, close, [volume]."""
    df = pd.read_csv(cfg.path, sep=cfg.sep)
    df = _standardize_ohlc(df)
    df = _parse_time_index(df, cfg.time_col, cfg.tz)
    df = _maybe_resample(df, cfg.timeframe)
    return df


def iter_stream(df: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, pd.Series]]:
    """Prosty 'stream': iterator po kolejnych wierszach (czas, rekord)."""
    for ts, row in df.iterrows():
        # Zwracamy (Timestamp, Series) – przyda się pod feed live/symulację
        yield ts, row

