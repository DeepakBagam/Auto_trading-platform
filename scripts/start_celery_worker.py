"""Start Celery worker for distributed task processing."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    from task_queue.celery_app import app

    # Start worker
    # celery -A task_queue.celery_app worker --loglevel=info --concurrency=4
    app.worker_main(
        argv=[
            "worker",
            "--loglevel=info",
            "--concurrency=4",
            "--max-tasks-per-child=100",
        ]
    )
