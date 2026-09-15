# NBA Predictor

NBA game outcome, spread/total, and player-prop predictions with a
conference-standings projection frontend. Sibling project to PL_Predictor,
NFL_Predictor, and CFB_Predictor — see
`docs/superpowers/specs/2026-09-15-nba-predictor-design.md` for the full
design.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

## Run the API

```bash
uvicorn nba_predictor.api.app:app --reload
```
