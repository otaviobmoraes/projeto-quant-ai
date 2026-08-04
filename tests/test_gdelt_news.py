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

VOLUME_SAMPLE = {
    "query_details": {"title": QUERY, "date_resolution": "day"},
    "timeline": [
        {
            "series": "Article Count",
            "data": [
                {"date": "20260601T000000Z", "value": 3, "norm": 100000},
                {"date": "20260602T000000Z", "value": 6, "norm": 120000},
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
    monkeypatch.setattr(
        gdelt_news, "VOLUME_PROCESSED_PATH", tmp_path / "processed" / "gdelt_volume.parquet"
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


def test_parse_volume_raw_computes_share_pct(tmp_path):
    path = tmp_path / "volume.json"
    path.write_text(json.dumps(VOLUME_SAMPLE), encoding="utf-8")

    df = gdelt_news._parse_volume_raw(path, QUERY)

    assert list(df.columns) == ["date", "article_count", "total_monitored", "share_pct", "query"]
    assert len(df) == 2
    assert df["share_pct"].tolist() == pytest.approx([3 / 100000 * 100, 6 / 120000 * 100])


def test_fetch_gdelt_volume_raw_chunks_and_caches(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        return _FakeResponse(VOLUME_SAMPLE)

    monkeypatch.setattr(gdelt_news.requests, "get", fake_get)

    start, end = date(2023, 1, 1), date(2023, 6, 1)  # > 100 dias -> mais de 1 janela
    paths = gdelt_news.fetch_gdelt_volume_raw(start, end, query=QUERY, chunk_days=100)

    assert len(paths) == 2  # jan-abr, abr-jun
    assert len(calls) == 2

    # cache: rodar de novo nao dispara novo request.
    paths_again = gdelt_news.fetch_gdelt_volume_raw(start, end, query=QUERY, chunk_days=100)
    assert paths_again == paths
    assert len(calls) == 2


def test_fetch_raw_chunked_retries_on_failure_then_succeeds(monkeypatch):
    calls = []

    def flaky_get(url, params=None, timeout=None):
        calls.append(params)
        if len(calls) == 1:
            raise gdelt_news.requests.exceptions.RequestException("429")
        return _FakeResponse(VOLUME_SAMPLE)

    monkeypatch.setattr(gdelt_news.requests, "get", flaky_get)
    monkeypatch.setattr(gdelt_news.time_module, "sleep", lambda seconds: None)

    paths = gdelt_news.fetch_gdelt_volume_raw(date(2026, 6, 1), date(2026, 6, 2), query=QUERY)

    assert len(paths) == 1
    assert paths[0].exists()
    assert len(calls) == 2  # falhou uma vez, teve sucesso na segunda


def test_fetch_raw_chunked_raises_after_max_retries(monkeypatch):
    def always_fails(url, params=None, timeout=None):
        raise gdelt_news.requests.exceptions.RequestException("429")

    monkeypatch.setattr(gdelt_news.requests, "get", always_fails)
    monkeypatch.setattr(gdelt_news.time_module, "sleep", lambda seconds: None)

    with pytest.raises(gdelt_news.requests.exceptions.RequestException):
        gdelt_news.fetch_gdelt_volume_raw(date(2026, 6, 1), date(2026, 6, 2), query=QUERY)


def test_load_gdelt_volume_processed_upserts_history(monkeypatch):
    monkeypatch.setattr(gdelt_news.requests, "get", lambda *a, **k: _FakeResponse(VOLUME_SAMPLE))

    df = gdelt_news.load_gdelt_volume_processed(date(2026, 6, 1), date(2026, 6, 2), query=QUERY)

    assert gdelt_news.VOLUME_PROCESSED_PATH.exists()
    assert len(df) == 2

    df_again = gdelt_news.load_gdelt_volume_processed(date(2026, 6, 1), date(2026, 6, 2), query=QUERY)
    assert len(df_again) == 2  # dedup por (query, date)


def test_load_fiscal_risk_series_raises_when_not_collected():
    with pytest.raises(FileNotFoundError):
        gdelt_news.load_fiscal_risk_series(query=QUERY)


def test_load_fiscal_risk_series_returns_share_pct_indexed_by_date(monkeypatch):
    monkeypatch.setattr(gdelt_news.requests, "get", lambda *a, **k: _FakeResponse(VOLUME_SAMPLE))
    gdelt_news.load_gdelt_volume_processed(date(2026, 6, 1), date(2026, 6, 2), query=QUERY)

    series = gdelt_news.load_fiscal_risk_series(query=QUERY)

    assert series.name == "share_pct"
    assert len(series) == 2
    assert series.index.is_monotonic_increasing
    assert series.tolist() == pytest.approx([3 / 100000 * 100, 6 / 120000 * 100])
