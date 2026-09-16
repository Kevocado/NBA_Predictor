from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from nba_predictor import config
from nba_predictor.api.routes import router
from nba_predictor.tracking.store import init_db


def create_app() -> FastAPI:
    init_db(config.TRACKING_DB_PATH)

    app = FastAPI(title="NBA Predictor API")
    app.include_router(router)

    frontend_dist = config.PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


app = create_app()
