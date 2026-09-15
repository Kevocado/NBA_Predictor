from fastapi import FastAPI

from nba_predictor.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="NBA Predictor API")
    app.include_router(router)
    return app


app = create_app()
