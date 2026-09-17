FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim AS runtime
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY models/ ./models/
COPY data/public_snapshot.json ./data/public_snapshot.json
COPY data/cache/schedule/ ./data/cache/schedule/
COPY data/cache/hub/ ./data/cache/hub/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

ENV PORT=8000
ENV PROJECT_ROOT=/app
EXPOSE 8000

CMD ["uvicorn", "nba_predictor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
