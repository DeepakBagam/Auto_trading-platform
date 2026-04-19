from data_layer.processors.news_cleaner import deduplicate_news


def test_news_dedup():
    items = [
        {"source": "A", "url": "u1", "title": "Title 1", "content": "x"},
        {"source": "A", "url": "u1", "title": "Title 1", "content": "x"},
        {"source": "B", "url": "u2", "title": "Title 2", "content": "y"},
    ]
    out = deduplicate_news(items)
    assert len(out) == 2
