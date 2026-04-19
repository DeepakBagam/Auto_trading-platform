"""Start Celery beat scheduler for periodic tasks."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    from task_queue.celery_app import app

    # Start beat scheduler
    # celery -A task_queue.celery_app beat --loglevel=info
    app.worker_main(
        argv=[
            "beat",
            "--loglevel=info",
        ]
    )
