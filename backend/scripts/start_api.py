try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import uvicorn

from backend.db.init_db import init_db
from backend.utils.config import get_settings
from backend.utils.logger import setup_logging


def main() -> None:
    setup_logging("api")
    init_db()
    settings = get_settings()
    uvicorn.run("backend.api.main:app", host=settings.api_host, port=settings.api_port, reload=False)


if __name__ == "__main__":
    main()
