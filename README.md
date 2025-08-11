# 🌲 Forest 4.0.0 — lekki framework do backtestów strategii

Forest to modularny framework do **backtestów** i **optymalizacji** strategii tradingowych w Pythonie.
Wersja **4.0.0** wprowadza ujednolicony **interfejs Strategii**, spójny **BacktestResult** z metrykami,
odświeżony **engine**, prosty **grid search** oraz **dashboard** w Streamlit.

## Spis treści
- [Cechy](#cechy)
- [Wymagania](#wymagania)
- [Instalacja](#instalacja)
- [Szybki start](#szybki-start)
- [Interfejs strategii](#interfejs-strategii)
- [Konfiguracja z pliku](#konfiguracja-z-pliku)
- [Grid search](#grid-search)
- [Dashboard (Streamlit)](#dashboard-streamlit)
- [Testy i CI](#testy-i-ci)
- [Migracja 3.x → 4.0](#migracja-3x--400)
- [Struktura katalogów](#struktura-katalogów)
- [Licencja](#licencja)

## Cechy
- ✅ **Strategy API** — dodawaj strategie jako klasy (`Strategy`), bez grzebania w engine
- ✅ **BacktestResult** — wyniki z equity, listą transakcji i metrykami (`equity_end`, `max_dd`, `cagr`, `rar`, `sharpe`)
- ✅ **RiskManager** — sizing wg ATR, trailing stop, koszty transakcyjne, kontrola max DD
- ✅ **Grid search** — proste przeszukiwanie parametrów + heat‑mapy w dashboardzie
- ✅ **Dashboard** — szybkie uruchomienie backtestu z CSV oraz wizualizacje w przeglądarce

## Wymagania
- Python **3.11**
- Poetry (zalecane) lub Conda
- Dane wejściowe w formacie **OHLC** z kolumnami: `time, open, high, low, close [, volume]`

## Instalacja

### Poetry (zalecane)
```bash
pip install --upgrade pip
pip install poetry
poetry install

