try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

from data_layer.collectors.news_collector import NewsCollector
from db.connection import SessionLocal
from utils.logger import setup_logging


def main() -> None:
    setup_logging()
    db = SessionLocal()
    try:
        inserted = NewsCollector().run_once(db)
        print(f"inserted_news={inserted}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
