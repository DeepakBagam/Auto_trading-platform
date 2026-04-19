try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

from data_layer.scheduler.scheduler import start_scheduler
from utils.logger import setup_logging


def main() -> None:
    setup_logging("scheduler")
    start_scheduler()


if __name__ == "__main__":
    main()
