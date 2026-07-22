import json
from datetime import date

import pandas as pd
import pytest

from data import gdelt_news

QUERY = "Brazil real currency"

TONE_SAMPLE = {
    "query_details": {"title": QUERY, "date_resolution": "day"},
    "timeline": [
        {
            "series": "Average Tone",
            "data": [
                {"date": "20260601T000000Z", "value": 0.3433},
                {"date": "20260602T000000Z", "value": -1.437},
            ],
        }
    ],
}

ARTLIST_SAMPLE = {
    "articles": [
        {
            "url": "https://example.com/a",
            "url_mobile": "",
            "title": "Brazil currency moves on rate decision",
            "seendate": "20260601T120000Z",
            "socialimage": "",
            "domain": "example.com",
            "language": "English",
            "sourcecountry": "United States",
        },
        {
            "url": "https://example.com/b",
            "url_mobile": "",
            "title": "Real weakens against dollar",
            "seendate": "20260602T090000Z",
            "socialimage": "",
            "domain": "example.com",
            "language": "English",
            "sourcecountry": "United States",
        },
    ]
}


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(gdelt_news, "RAW_DIR", tmp_path / "raw" / "gdelt")
    monkeypatch.setattr(gdelt_news, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(gdelt_news, "TONE_PROCESSED_PATH", tmp_path / "processed" / "gdelt_tone.parquet")
    monkeypatch.setattr(
        gdelt_news, "HEADLINES_PROCESSED_PATH", tmp_path / "processed" / "gdelt_headlines.parquet"
    )
    yield


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self):
        pass


def test_fetch_gdelt_tone_raw_calls_api_and_caches(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        return _FakeResponse(TONE_SAMPLE)

    monkeypatch.setattr(gdelt_news.requests, "get", fake_get)

    start, end = date(2026, 6, 1), date(2026, 6, 2)
    path = gdelt_news.fetch_gdelt_tone_raw(start, end, query=QUERY)

    assert path.exists()
    assert len(calls) == 1
    assert calls[0]["mode"] == "timelinetone"
    assert calls[0]["query"] == QUERY

    path_again = gdelt_news.fetch_gdelt_tone_raw(start, end, query=QUERY)
    assert path_again == path
    assert len(calls) == 1  # cache hit, sem novo request


def test_parse_tone_raw_builds_expected_dataframe(tmp_path):
    path = tmp_path / "tone.json"
    path.write_text(json.dumps(TONE_SAMPLE), encoding="utf-8")

    df = gdelt_news._parse_tone_raw(path, QUERY)

    assert list(df.columns) == ["date", "tone", "query"]
    assert len(df) == 2
    assert str(df["date"].dt.tz) == gdelt_news.TIMEZONE
    assert df["tone"].tolist() == [0.3433, -1.437]
    assert df["query"].unique().tolist() == [QUERY]


def test_parse_headlines_raw_builds_expected_dataframe(tmp_path):
    path = tmp_path / "artlist.json"
    path.write_text(json.dumps(ARTLIST_SAMPLE), encoding="utf-8")

    df = gdelt_news._parse_headlines_raw(path, QUERY)

    assert list(df.columns) == [
        "date",
        "title",
        "url",
        "domain",
        "language",
        "sourcecountry",
        "query",
    ]
    assert len(df) == 2
    assert str(df["date"].dt.tz) == gdelt_news.TIMEZONE


def test_load_gdelt_tone_processed_upserts_history(monkeypatch):
    monkeypatch.setattr(gdelt_news.requests, "get", lambda *a, **k: _FakeResponse(TONE_SAMPLE))

    df = gdelt_news.load_gdelt_tone_processed(date(2026, 6, 1), date(2026, 6, 2), query=QUERY)

    assert gdelt_news.TONE_PROCESSED_PATH.exists()
    assert len(df) == 2

    reloaded = pd.read_parquet(gdelt_news.TONE_PROCESSED_PATH)
    assert len(reloaded) == len(df)

    # Reprocessar a mesma janela nao deve duplicar linhas (upsert por query+date).
    df_again = gdelt_news.load_gdelt_tone_processed(date(2026, 6, 1), date(2026, 6, 2), query=QUERY)
    assert len(df_again) == 2


def test_load_gdelt_headlines_processed_upserts_history(monkeypatch):
    monkeypatch.setattr(gdelt_news.requests, "get", lambda *a, **k: _FakeResponse(ARTLIST_SAMPLE))

    df = gdelt_news.load_gdelt_headlines_processed(date(2026, 6, 1), date(2026, 6, 2), query=QUERY)

    assert gdelt_news.HEADLINES_PROCESSED_PATH.exists()
    assert len(df) == 2

    df_again = gdelt_news.load_gdelt_headlines_processed(
        date(2026, 6, 1), date(2026, 6, 2), query=QUERY
    )
    assert len(df_again) == 2  # dedup por (query, url)
