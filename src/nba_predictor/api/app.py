from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Scope

from nba_predictor import config
from nba_predictor.api.facts import router as facts_router
from nba_predictor.api.routes import router
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


def create_app() -> FastAPI:
    init_db(config.TRACKING_DB_PATH)

    app = FastAPI(title="NBA Predictor API")
    app.include_router(router)
    # The explainer service calls {SPORT_API}/facts/{id} on the API root, so
    # this router carries no /api prefix. Registered before the SPA mount
    # below so /facts/* is never swallowed by the static-file fallback.
    # /facts/upcoming is declared before /facts/{game_id} inside facts.py.
    app.include_router(facts_router)

    frontend_dist = config.PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", SPAStaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


app = create_app()
