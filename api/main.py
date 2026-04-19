from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

from api.routes.data import router as data_router
from api.routes.dashboard import router as dashboard_router
from api.routes.execution import router as execution_router
from api.routes.health import router as health_router
from api.routes.model import router as model_router
from api.routes.options import router as options_router
from api.routes.predict import router as predict_router
from api.routes.signal import router as signal_router
from observability.middleware import PrometheusMiddleware
from utils.logger import setup_logging

setup_logging()

app = FastAPI(title="Automated AI Trading Platform API", version="0.1.0")

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(health_router)
app.include_router(data_router)
app.include_router(dashboard_router)
app.include_router(predict_router)
app.include_router(model_router)
app.include_router(signal_router)
app.include_router(options_router)
app.include_router(execution_router)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/dashboard", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        # Serve a favicon if available; otherwise return 204 to avoid noisy 404s
        favicon_path = WEB_DIR / "favicon.ico"
        if favicon_path.exists():
            return FileResponse(favicon_path)
        from fastapi.responses import Response

        return Response(status_code=204)
