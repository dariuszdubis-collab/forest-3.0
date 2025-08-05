# FOREST 3.0 📈 — algorithmic‑trading research stack

[![CI (build + tests)](https://github.com/dariuszdubis-collab/forest-3.0/actions/workflows/ci.yml/badge.svg)](https://github.com/dariuszdubis-collab/forest-3.0/actions)

**Forest 3.0** to trzecia, od zera przepisana wersja mojego środowiska
back‑test / ML‑driven trading. Celem jest:

* 💡 **przejrzyste logowanie** – każda decyzja sygnału z opisem „why / why not”  
* ⚡ **wektorowy silnik back‑testu** (NumPy + Numba) – setki parametrów / min  
* 🧩 **architektura pluginów** – nowe strategie i indykatory bez zmian core  
* 🔒 **Risk & Money Management** – ATR, trailing SL, max‑DD guard  
* 📈 łatwy eksport equity / metrics do Dash / Streamlit

---

## Szybki start (💻 Ubuntu + Conda)

```bash
# 1. Klon repo
git clone https://github.com/dariuszdubis-collab/forest-3.0.git
cd forest-3.0

# 2. Jednorazowo – środowisko
conda create -n forest3 python=3.11 poetry -c conda-forge
conda activate forest3
poetry config virtualenvs.create false --local
poetry install --with dev         # ~30 s

# 3. Testy & lint lokalnie
poetry run ruff check src tests --fix
poetry run pytest -q              # 4 passed

# 4. Commit → push → GitHub Actions (zielony badge)

