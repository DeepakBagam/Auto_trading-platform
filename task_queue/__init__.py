"""Task queue package for distributed processing."""
from task_queue.celery_app import app

__all__ = ["app"]
