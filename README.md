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

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The dev server proxies `/teams`, `/games`, `/hub`, `/manifest`,
`/retrain`, and `/refresh-odds` to `http://127.0.0.1:8020` (see
`frontend/vite.config.ts`) — run the API on port 8020 alongside it for a
full local setup, or change the proxy target.

## Deployment

NBA_Predictor deploys the same way as PL_Predictor, F1_Predictor, and the
NFL/CFB predictor: GitHub Actions builds the Docker image, pushes it to
GHCR, and updates an Azure Container App in the shared `predictor-hub-rg`
resource group. See `.github/workflows/deploy-azure.yml`.

### One-time setup (run once, by a human, before the first automated deploy)

1. Find the existing Container Apps environment the other predictors share:
   ```bash
   az containerapp env list --resource-group predictor-hub-rg --query "[].name" -o tsv
   ```
2. Create the `nba-predictor` Container App in that environment, pointing at
   the image the first CI run will have pushed to GHCR:
   ```bash
   az containerapp create \
     --name nba-predictor \
     --resource-group predictor-hub-rg \
     --environment <environment-name-from-step-1> \
     --image ghcr.io/kevocado/nba-predictor:latest \
     --target-port 8000 \
     --ingress external \
     --registry-server ghcr.io \
     --registry-username <github-username> \
     --registry-password <GHCR_PAT>
   ```
3. Once live, update the "NBA Predictor" card in
   `predictor-hub/index.html` from "In development" to a live link pointing
   at the new Container App's URL (printed by the `create` command above,
   or found via `az containerapp show --name nba-predictor --resource-group
   predictor-hub-rg --query properties.configuration.ingress.fqdn`).

After this one-time setup, every push to `main` redeploys automatically via
`.github/workflows/deploy-azure.yml`.

### Public snapshot

`data/public_snapshot.json` bundles the current schedule, hub aggregates,
track record, and model manifest into one file, refreshed every 6 hours by
`.github/workflows/refresh-public-snapshot.yml`. Regenerate it manually with:

```bash
python -m nba_predictor.public_snapshot
```
