"""The whole daily refresh in one command: ingest (fetch, retrain on frozen history + window, score), then the
odds-derived markets pass. Run by the VPS timer: `python -u -m nba_predictor.pipeline.refresh_full`.

POST /admin/refresh-full does the same from inside the API process with a hard-coded window; this uses the ingest
defaults (previous season's Oct 1 to 14 days out), so it keeps working when the season rolls over. A markets fault
(no odds key, API down) is reported in the summary and never fails the run: the picks were already scored.
"""
from __future__ import annotations

import json

from nba_predictor import config
from nba_predictor.pipeline.ingest import default_window_end, default_window_start, run_ingest
from nba_predictor.pipeline.refresh_kalshi import refresh_kalshi_markets
from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
from nba_predictor.services.schedule_repository import load_schedule


def run_full_refresh(log=print) -> dict:
    summary = run_ingest(default_window_start(), default_window_end(), log=log)
    # Kalshi moneylines first: free, public, keyless. The sportsbook pass below is a paid, rate-limited feed.
    try:
        schedule = load_schedule(config.DATA_DIR / "cache" / "schedule" / "games.json")
        summary["kalshi_market_rows"] = refresh_kalshi_markets(schedule, config.TRACKING_DB_PATH)
        log(f"  stored {summary['kalshi_market_rows']} Kalshi market rows")
    except Exception as exc:  # noqa: BLE001
        summary["kalshi_error"] = str(exc)
        log(f"  Kalshi market pass failed (the picks are already scored): {exc}")
    try:
        from nba_predictor.api.routes import _market_stds_from_manifest

        schedule = load_schedule(config.DATA_DIR / "cache" / "schedule" / "games.json")
        margin_std, total_std = _market_stds_from_manifest(config.PROJECT_ROOT / "models")
        summary["market_predictions_stored"] = refresh_market_predictions(
            schedule, config.TRACKING_DB_PATH, margin_std=margin_std, total_std=total_std
        )
        log(f"  stored {summary['market_predictions_stored']} market predictions")
    except Exception as exc:  # noqa: BLE001 - see module docstring
        summary["market_error"] = str(exc)
        log(f"  market pass failed (the picks are already scored): {exc}")
    return summary


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in run_full_refresh().items() if not isinstance(v, (dict, list))}, default=str))
