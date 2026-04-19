"""Model training and inference tasks for Celery distributed processing."""
from celery import Task
from celery.utils.log import get_task_logger

from task_queue.celery_app import app

logger = get_task_logger(__name__)


class DatabaseTask(Task):
    """Base task with database session management."""

    _db = None

    @property
    def db(self):
        if self._db is None:
            from db.connection import SessionLocal

            self._db = SessionLocal()
        return self._db

    def after_return(self, *args, **kwargs):
        if self._db is not None:
            self._db.close()
            self._db = None


@app.task(bind=True, base=DatabaseTask, name="task_queue.tasks.model_tasks.build_features_task")
def build_features_task(self, symbols: list[str] | None = None):
    """Build features for all symbols."""
    from feature_engine.feature_builder import FeatureBuilder
    from utils.config import get_settings

    settings = get_settings()
    builder = FeatureBuilder(session_factory=lambda: self.db)

    if symbols is None:
        symbols = [key.split("|")[1] if "|" in key else key for key in settings.instrument_keys]

    try:
        results = {}
        for symbol in symbols:
            count = builder.build_features_for_symbol(symbol)
            results[symbol] = count
            logger.info(f"Built {count} feature rows for {symbol}")

        return {"status": "success", "results": results}
    except Exception as e:
        logger.exception("Failed to build features")
        return {"status": "error", "error": str(e)}


@app.task(bind=True, base=DatabaseTask, name="task_queue.tasks.model_tasks.train_models_task")
def train_models_task(self, symbols: list[str] | None = None, model_version: str = "v3"):
    """Train models for all symbols."""
    from models.v3_pipeline import build_point_in_time_dataset
    from models.xgboost.train import train_xgboost_models
    from utils.config import get_settings

    settings = get_settings()

    if symbols is None:
        symbols = [key.split("|")[1] if "|" in key else key for key in settings.instrument_keys]

    try:
        results = {}
        for symbol in symbols:
            dataset, quality = build_point_in_time_dataset(self.db, symbol)
            if not quality.passed:
                logger.warning(f"Quality check failed for {symbol}: {quality.details}")
                results[symbol] = {"status": "skipped", "reason": "quality_check_failed"}
                continue

            metrics = train_xgboost_models(self.db, symbol, dataset)
            results[symbol] = {"status": "success", "metrics": metrics}
            logger.info(f"Trained models for {symbol}: {metrics}")

        return {"status": "success", "results": results}
    except Exception as e:
        logger.exception("Failed to train models")
        return {"status": "error", "error": str(e)}


@app.task(bind=True, base=DatabaseTask, name="task_queue.tasks.model_tasks.run_inference_task")
def run_inference_task(self, symbols: list[str] | None = None):
    """Run inference for all symbols."""
    from prediction_engine.orchestrator import PredictionOrchestrator
    from utils.config import get_settings

    settings = get_settings()
    orchestrator = PredictionOrchestrator(session_factory=lambda: self.db)

    if symbols is None:
        symbols = [key.split("|")[1] if "|" in key else key for key in settings.instrument_keys]

    try:
        results = {}
        for symbol in symbols:
            prediction = orchestrator.predict_next_day(symbol)
            results[symbol] = prediction
            logger.info(f"Generated prediction for {symbol}: {prediction}")

        return {"status": "success", "results": results}
    except Exception as e:
        logger.exception("Failed to run inference")
        return {"status": "error", "error": str(e)}


@app.task(
    bind=True,
    base=DatabaseTask,
    name="task_queue.tasks.model_tasks.train_single_model_task",
)
def train_single_model_task(self, symbol: str, model_type: str = "xgboost"):
    """Train a single model for A/B testing."""
    from models.v3_pipeline import build_point_in_time_dataset
    from utils.config import get_settings

    settings = get_settings()

    try:
        dataset, quality = build_point_in_time_dataset(self.db, symbol)
        if not quality.passed:
            return {"status": "skipped", "reason": "quality_check_failed", "quality": quality.details}

        if model_type == "xgboost":
            from models.xgboost.train import train_xgboost_models

            metrics = train_xgboost_models(self.db, symbol, dataset)
        elif model_type == "lightgbm":
            from models.lgbm_v3 import train_lgbm_models

            metrics = train_lgbm_models(self.db, symbol, dataset)
        elif model_type == "catboost":
            from models.catboost_v3 import train_catboost_models

            metrics = train_catboost_models(self.db, symbol, dataset)
        else:
            return {"status": "error", "error": f"Unknown model_type: {model_type}"}

        logger.info(f"Trained {model_type} for {symbol}: {metrics}")
        return {"status": "success", "model_type": model_type, "metrics": metrics}
    except Exception as e:
        logger.exception(f"Failed to train {model_type} for {symbol}")
        return {"status": "error", "error": str(e)}
