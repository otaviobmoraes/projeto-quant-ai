import json
from datetime import date

import numpy as np
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


def test_fetch_gdelt_headlines_raw_chunked_chunks_and_caches(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        return _FakeResponse(ARTLIST_SAMPLE)

    monkeypatch.setattr(gdelt_news.requests, "get", fake_get)
    monkeypatch.setattr(gdelt_news.time_module, "sleep", lambda seconds: None)

    start, end = date(2023, 1, 1), date(2023, 1, 20)  # 19 dias, chunk_days=7 -> 3 janelas
    paths = gdelt_news.fetch_gdelt_headlines_raw_chunked(start, end, query=QUERY, chunk_days=7)

    assert len(paths) == 3
    assert len(calls) == 3
    assert all(c["mode"] == "artlist" for c in calls)

    # cache: rodar de novo nao dispara novo request.
    paths_again = gdelt_news.fetch_gdelt_headlines_raw_chunked(start, end, query=QUERY, chunk_days=7)
    assert paths_again == paths
    assert len(calls) == 3


def test_load_gdelt_headlines_processed_chunked_upserts_history(monkeypatch):
    monkeypatch.setattr(gdelt_news.requests, "get", lambda *a, **k: _FakeResponse(ARTLIST_SAMPLE))
    monkeypatch.setattr(gdelt_news.time_module, "sleep", lambda seconds: None)

    df = gdelt_news.load_gdelt_headlines_processed_chunked(
        date(2023, 1, 1), date(2023, 1, 20), query=QUERY, chunk_days=7
    )

    assert gdelt_news.HEADLINES_PROCESSED_PATH.exists()
    # 2 artigos por janela x 3 janelas, mas dedup por (query, url) -> so 2 unicos
    assert len(df) == 2

    df_again = gdelt_news.load_gdelt_headlines_processed_chunked(
        date(2023, 1, 1), date(2023, 1, 20), query=QUERY, chunk_days=7
    )
    assert len(df_again) == 2


def test_select_surprise_spike_windows_picks_top_n_non_clustered():
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    surprise = pd.Series(np.nan, index=idx)  # NaN = sem baseline confiavel ainda (fora do ranking)
    # 3 picos, dois deles bem proximos um do outro (deveriam contar como 1 evento)
    surprise.iloc[50] = 5.0
    surprise.iloc[52] = 4.5  # a 2 dias do pico anterior -- clusterizado
    surprise.iloc[150] = 3.0

    windows = gdelt_news.select_surprise_spike_windows(
        surprise, top_n=15, window_days=3, min_gap_days=10
    )

    assert len(windows) == 2  # so 2 eventos distintos, nao 3
    for start, end in windows:
        assert start < end


def test_select_surprise_spike_windows_respects_top_n():
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    rng = np.random.default_rng(3)
    surprise = pd.Series(rng.normal(0, 1, 200), index=idx)

    windows = gdelt_news.select_surprise_spike_windows(
        surprise, top_n=5, window_days=2, min_gap_days=1
    )

    assert len(windows) == 5


def test_fetch_gdelt_headlines_raw_for_windows_one_request_per_window(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        return _FakeResponse(ARTLIST_SAMPLE)

    monkeypatch.setattr(gdelt_news.requests, "get", fake_get)
    monkeypatch.setattr(gdelt_news.time_module, "sleep", lambda seconds: None)

    windows = [
        (date(2024, 11, 15), date(2024, 11, 22)),
        (date(2025, 3, 1), date(2025, 3, 8)),
    ]
    paths = gdelt_news.fetch_gdelt_headlines_raw_for_windows(windows, query=QUERY)

    assert len(paths) == 2
    assert len(calls) == 2


def test_load_gdelt_headlines_processed_for_windows_upserts_history(monkeypatch):
    monkeypatch.setattr(gdelt_news.requests, "get", lambda *a, **k: _FakeResponse(ARTLIST_SAMPLE))
    monkeypatch.setattr(gdelt_news.time_module, "sleep", lambda seconds: None)

    windows = [(date(2024, 11, 15), date(2024, 11, 22)), (date(2025, 3, 1), date(2025, 3, 8))]
    df = gdelt_news.load_gdelt_headlines_processed_for_windows(windows, query=QUERY)

    assert gdelt_news.HEADLINES_PROCESSED_PATH.exists()
    assert len(df) == 2  # 2 artigos por janela, mas dedup por (query, url) -> 2 unicos


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
    monkeypatch.setattr(gdelt_news.time_module, "sleep", lambda seconds: None)

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


def test_fetch_raw_chunked_uses_longer_backoff_for_429(monkeypatch):
    calls = []
    sleeps = []

    class _FakeResponse429:
        status_code = 429

        def raise_for_status(self):
            raise gdelt_news.requests.exceptions.HTTPError(response=self)

    def flaky_get(url, params=None, timeout=None):
        calls.append(params)
        if len(calls) == 1:
            return _FakeResponse429()
        return _FakeResponse(VOLUME_SAMPLE)

    monkeypatch.setattr(gdelt_news.requests, "get", flaky_get)
    monkeypatch.setattr(gdelt_news.time_module, "sleep", lambda seconds: sleeps.append(seconds))

    paths = gdelt_news.fetch_gdelt_volume_raw(date(2026, 6, 1), date(2026, 6, 2), query=QUERY)

    assert len(paths) == 1
    assert sleeps[0] == 90  # backoff de 429 (90s), nao o generico (15s)


def test_fetch_raw_chunked_raises_after_max_retries(monkeypatch):
    def always_fails(url, params=None, timeout=None):
        raise gdelt_news.requests.exceptions.RequestException("429")

    monkeypatch.setattr(gdelt_news.requests, "get", always_fails)
    monkeypatch.setattr(gdelt_news.time_module, "sleep", lambda seconds: None)

    with pytest.raises(gdelt_news.requests.exceptions.RequestException):
        gdelt_news.fetch_gdelt_volume_raw(date(2026, 6, 1), date(2026, 6, 2), query=QUERY)


def test_load_gdelt_volume_processed_upserts_history(monkeypatch):
    monkeypatch.setattr(gdelt_news.requests, "get", lambda *a, **k: _FakeResponse(VOLUME_SAMPLE))
    monkeypatch.setattr(gdelt_news.time_module, "sleep", lambda seconds: None)

    df = gdelt_news.load_gdelt_volume_processed(date(2026, 6, 1), date(2026, 6, 2), query=QUERY)

    assert gdelt_news.VOLUME_PROCESSED_PATH.exists()
    assert len(df) == 2

    df_again = gdelt_news.load_gdelt_volume_processed(date(2026, 6, 1), date(2026, 6, 2), query=QUERY)
    assert len(df_again) == 2  # dedup por (query, date)


def test_fiscal_risk_surprise_zero_when_series_flat():
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    series = pd.Series(0.1, index=idx)  # sem variacao nenhuma

    surprise = gdelt_news.fiscal_risk_surprise(series, window=63, min_periods=21)

    # desvio-padrao 0 -> divisao vira NaN (evita divisao por zero), nao 0/0 explodindo
    assert surprise.dropna().eq(0).all() or surprise.isna().all()


def test_fiscal_risk_surprise_flags_spike_above_recent_baseline():
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    series = pd.Series(0.1 + rng.normal(0, 0.01, 100), index=idx)
    series.iloc[90] = 1.0  # pico bem acima do normal recente

    surprise = gdelt_news.fiscal_risk_surprise(series, window=63, min_periods=21)

    assert surprise.iloc[90] > 3  # pico vira surpresa grande em desvios-padrao
    assert surprise.iloc[:20].isna().all()  # antes de min_periods, sem baseline confiavel


def test_fiscal_risk_surprise_uses_only_past_data_no_lookahead():
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    rng = np.random.default_rng(1)
    series = pd.Series(0.1 + rng.normal(0, 0.01, 100), index=idx)

    surprise_full = gdelt_news.fiscal_risk_surprise(series, window=63, min_periods=21)
    # trunca a serie ANTES do pico: se o calculo em t=50 usasse dado futuro,
    # truncar em t=60 mudaria o valor em t=50.
    surprise_truncated = gdelt_news.fiscal_risk_surprise(series.iloc[:60], window=63, min_periods=21)

    pd.testing.assert_series_equal(
        surprise_full.iloc[:60], surprise_truncated, check_names=False
    )


def test_load_fiscal_risk_series_raises_when_not_collected():
    with pytest.raises(FileNotFoundError):
        gdelt_news.load_fiscal_risk_series(query=QUERY)


def test_load_fiscal_risk_series_returns_share_pct_indexed_by_date(monkeypatch):
    monkeypatch.setattr(gdelt_news.requests, "get", lambda *a, **k: _FakeResponse(VOLUME_SAMPLE))
    monkeypatch.setattr(gdelt_news.time_module, "sleep", lambda seconds: None)
    gdelt_news.load_gdelt_volume_processed(date(2026, 6, 1), date(2026, 6, 2), query=QUERY)

    series = gdelt_news.load_fiscal_risk_series(query=QUERY)

    assert series.name == "share_pct"
    assert len(series) == 2
    assert series.index.is_monotonic_increasing
    assert series.tolist() == pytest.approx([3 / 100000 * 100, 6 / 120000 * 100])
