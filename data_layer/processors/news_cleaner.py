import re
from typing import Iterable, List


def clean_text(value: str) -> str:
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def deduplicate_news(items: Iterable[dict]) -> List[dict]:
    seen = set()
    output = []
    for item in items:
        key = (item.get("source"), item.get("url"), clean_text(item.get("title", "")).lower())
        if key in seen:
            continue
        seen.add(key)
        item = dict(item)
        item["title"] = clean_text(item.get("title", ""))
        item["content"] = clean_text(item.get("content", ""))
        output.append(item)
    return output
