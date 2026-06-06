from zoneinfo import ZoneInfo

IST_ZONE = ZoneInfo("Asia/Kolkata")

MINUTE_INTERVALS = tuple(f"{value}minute" for value in range(1, 301))
HOUR_INTERVALS = tuple(f"{value}hour" for value in range(1, 6))
DAILY_INTERVALS = ("day", "week", "month")
SUPPORTED_INTERVALS = (*MINUTE_INTERVALS, *HOUR_INTERVALS, *DAILY_INTERVALS)
NEWS_RSS_FEEDS = (
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/latestnews.xml",
)

DIRECTION_BUY = "BUY"
DIRECTION_SELL = "SELL"
DIRECTION_HOLD = "HOLD"
