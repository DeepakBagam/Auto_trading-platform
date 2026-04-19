"""
Celery distributed task queue configuration.
Enables async processing of data ingestion, feature building, and model training.
"""
from celery import Celery
from celery.schedules import crontab

from utils.config import get_settings

settings = get_settings()

# Celery app initialization
app = Celery(
    "trading_platform",
    broker=f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
    backend=f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
    include=[
        "task_queue.tasks.data_tasks",
        "task_queue.tasks.model_tasks",
        "task_queue.tasks.execution_tasks",
    ],
)

# Celery configuration
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3300,  # 55 min soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    result_expires=86400,  # 24 hours
    broker_connection_retry_on_startup=True,
)

# Scheduled tasks (replaces APScheduler)
app.conf.beat_schedule = {
    "ingest-news-every-10min": {
        "task": "task_queue.tasks.data_tasks.ingest_news_task",
        "schedule": crontab(minute="*/10"),
    },
    "ingest-candles-market-hours": {
        "task": "task_queue.tasks.data_tasks.ingest_candles_task",
        "schedule": crontab(minute="*/1", hour="9-15", day_of_week="mon-fri"),
    },
    "build-features-daily": {
        "task": "task_queue.tasks.model_tasks.build_features_task",
        "schedule": crontab(hour=16, minute=30, day_of_week="mon-fri"),
    },
    "train-models-weekly": {
        "task": "task_queue.tasks.model_tasks.train_models_task",
        "schedule": crontab(hour=18, minute=0, day_of_week="saturday"),
    },
    "run-inference-daily": {
        "task": "task_queue.tasks.model_tasks.run_inference_task",
        "schedule": crontab(hour=8, minute=0, day_of_week="mon-fri"),
    },
}

if __name__ == "__main__":
    app.start()
