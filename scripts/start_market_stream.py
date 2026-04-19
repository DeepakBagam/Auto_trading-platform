try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

from data_layer.streamers.upstox_market_stream import UpstoxMarketStream
from utils.logger import setup_logging


def main() -> None:
    setup_logging("market_stream")
    UpstoxMarketStream().run_forever()


if __name__ == "__main__":
    main()
