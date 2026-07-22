from datetime import date

import pandas as pd
import pytest

from data import rss_news

SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>Fonte Teste</title>
<item>
<title>Dolar sobe apos decisao do Copom</title>
<link>https://example.com/noticia-1</link>
<pubDate>Wed, 22 Jul 2026 13:28:04 +0000</pubDate>
<description><![CDATA[<p>O <b>dolar</b> subiu 1% hoje.</p>]]></description>
</item>
<item>
<title>Bolsa fecha em alta</title>
<link>https://example.com/noticia-2</link>
<pubDate>Wed, 22 Jul 2026 12:00:00 +0000</pubDate>
<description><![CDATA[Ibovespa fechou em alta de 2%.]]></description>
</item>
</channel>
</rss>
"""


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(rss_news, "RAW_DIR", tmp_path / "raw" / "rss_news")
    monkeypatch.setattr(rss_news, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(rss_news, "PROCESSED_PATH", tmp_path / "processed" / "rss_news.parquet")
    yield


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


def test_fetch_rss_raw_caches_by_day(monkeypatch):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return _FakeResponse(SAMPLE_RSS.encode("utf-8"))

    monkeypatch.setattr(rss_news.requests, "get", fake_get)

    today = date(2026, 7, 22)
    path = rss_news.fetch_rss_raw("infomoney", as_of=today)
    assert path.exists()
    assert len(calls) == 1
    assert calls[0] == rss_news.FEEDS["infomoney"]

    path_again = rss_news.fetch_rss_raw("infomoney", as_of=today)
    assert path_again == path
    assert len(calls) == 1


def test_parse_rss_raw_builds_expected_dataframe(tmp_path):
    path = tmp_path / "sample.xml"
    path.write_text(SAMPLE_RSS, encoding="utf-8")

    df = rss_news._parse_rss_raw(path, "infomoney")

    assert list(df.columns) == ["date", "title", "link", "description", "source"]
    assert len(df) == 2
    assert str(df["date"].dt.tz) == rss_news.TIMEZONE
    assert df["description"].iloc[0] == "O dolar subiu 1% hoje."
    assert df["source"].unique().tolist() == ["infomoney"]


def test_load_rss_processed_upserts_history(monkeypatch):
    monkeypatch.setattr(
        rss_news.requests, "get", lambda url, headers=None, timeout=None: _FakeResponse(
            SAMPLE_RSS.encode("utf-8")
        )
    )

    df = rss_news.load_rss_processed(sources=("infomoney",), as_of=date(2026, 7, 22))

    assert rss_news.PROCESSED_PATH.exists()
    assert len(df) == 2

    df_again = rss_news.load_rss_processed(sources=("infomoney",), as_of=date(2026, 7, 22))
    assert len(df_again) == 2  # dedup por (source, link)

    reloaded = pd.read_parquet(rss_news.PROCESSED_PATH)
    assert len(reloaded) == len(df_again)
