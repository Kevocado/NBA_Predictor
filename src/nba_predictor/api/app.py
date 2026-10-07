import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Scope

from nba_predictor import config
from nba_predictor.api.deps import get_models_dir, get_schedule_path
from nba_predictor.api.explain import router as explain_router
from nba_predictor.api.facts import router as facts_router
from nba_predictor.api.signals import router as signals_router
from nba_predictor.api.routes import router, start_mae_warmer, _market_stds_from_manifest
from nba_predictor.services.hub_service import warm_player_props_cache
from nba_predictor.pipeline.odds_refresher import start_odds_refresher
from nba_predictor.tracking.store import init_db


class SPAStaticFiles(StaticFiles):
    """Falls back to index.html for any path that isn't a real static
    asset, so client-side routes (/hub, /model, /calibration, ...) work on
    a direct navigation or page refresh, not just via in-app SPA
    navigation. Plain StaticFiles(html=True) only serves index.html for
    "/" and 404s everything else that isn't a literal file on disk —
    confirmed live: curl /hub against the built image returned a real 404,
    not the app shell. StaticFiles.get_response *raises* HTTPException(404)
    rather than returning a 404 response, so that's what has to be caught."""

    async def get_response(self, path: str, scope: Scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the player-props MAE cache off the startup path, and keep the odds
    market predictions fresh.

    The first GET /games/{game_id}/players after any deploy or database change
    otherwise costs the visitor about a second of bulk reads that belong to the
    process, not to them. The warmer is a daemon thread started here and its
    failures are logged inside warm_mae_cache, so a database that cannot be read
    yet at boot delays nothing and takes nothing down.

    The track-record player-props aggregate gets the same treatment, once, here.
    It is a 1.5s aggregate over 270,660 snapshots and 124,000 outcomes that
    `GET /hub/track-record` needs on every request, and a cold first visitor
    should not pay for it. No second thread: that aggregate's bound is a 300s
    TTL rather than a database-state change, so past the TTL a request simply
    recomputes it, which is the correct answer and needs no polling to get it.

    The odds refresher is the same shape and for the same reason.
    `POST /refresh-odds` is `Depends(require_admin)`, so before this it ran only
    when somebody called it by hand: the deployed container held
    `SPORTSBOOK_API_KEY` and its `cache/sportsbook/` stayed empty, which is what
    "wired up but never switched on" looks like. NBA's games are spread across
    the week, so the refresh is a loop rather than a once-a-day step, and its
    cadence is the sportsbook cache TTL — a faster tick spends no requests,
    because the responses are already cached, and achieves nothing.
    """
    try:
        warm_player_props_cache(config.TRACKING_DB_PATH)
    except Exception:  # noqa: BLE001 - a warm that cannot start must not stop the app
        logging.getLogger(__name__).exception("player-props aggregate warm could not start")
    try:
        start_mae_warmer(config.TRACKING_DB_PATH, get_schedule_path())
    except Exception:  # noqa: BLE001 - a warm that cannot start must not stop the app
        logging.getLogger(__name__).exception("MAE cache warmer could not start")
    try:
        margin_std, total_std = _market_stds_from_manifest(get_models_dir())
        start_odds_refresher(
            config.TRACKING_DB_PATH,
            get_schedule_path(),
            margin_std=margin_std,
            total_std=total_std,
        )
    except Exception:  # noqa: BLE001 - odds that cannot refresh must not stop the app
        logging.getLogger(__name__).exception("odds refresher could not start")
    yield


def create_app() -> FastAPI:
    init_db(config.TRACKING_DB_PATH)

    app = FastAPI(title="NBA Predictor API", lifespan=lifespan)
    app.include_router(router)
    # The explainer service calls {SPORT_API}/facts/{id} on the API root, so
    # this router carries no /api prefix. Registered before the SPA mount
    # below so /facts/* is never swallowed by the static-file fallback.
    # /facts/upcoming is declared before /facts/{game_id} inside facts.py.
    app.include_router(facts_router)
    # The browser's route to the plain-English summary. Caddy only
    # reverse-proxies this app, so the explainer is reached through here rather
    # than directly. Registered before the SPA mount below so /api/explain/* is
    # never swallowed by the static-file fallback.
    app.include_router(explain_router)
    # GET /api/signals/{game_id} -- spec §3's per-fixture signal payloads, rendered
    # by the shared predictor-ui component. A game with nothing to say returns an
    # empty list rather than an empty state.
    app.include_router(signals_router)

    frontend_dist = config.PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", SPAStaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


app = create_app()
