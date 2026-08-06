from datetime import date

import pandas as pd
import pytest

from data import global_risk

SAMPLE_INDEX = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
SAMPLE_DF = pd.DataFrame(
    {
        ("Close", "^VIX"): [13.5, 14.2],
        ("Close", "DX-Y.NYB"): [102.1, 101.8],
        ("Open", "^VIX"): [13.8, 14.0],
        ("Open", "DX-Y.NYB"): [102.3, 101.9],
        ("Volume", "^VIX"): [0, 0],
        ("Volume", "DX-Y.NYB"): [0, 0],
    },
    index=SAMPLE_INDEX,
)
SAMPLE_DF.columns = pd.MultiIndex.from_tuples(SAMPLE_DF.columns, names=["Price", "Ticker"])


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(global_risk, "RAW_DIR", tmp_path / "raw" / "global_risk")
    monkeypatch.setattr(global_risk, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(global_risk, "PROCESSED_PATH", tmp_path / "processed" / "global_risk.parquet")
    yield


def test_fetch_global_risk_raw_calls_yfinance_and_caches(monkeypatch):
    calls = []

    def fake_download(tickers, **kwargs):
        calls.append((tickers, kwargs))
        return SAMPLE_DF.copy()

    monkeypatch.setattr(global_risk.yf, "download", fake_download)

    start, end = date(2024, 1, 2), date(2024, 1, 3)
    path = global_risk.fetch_global_risk_raw(start, end)

    assert path.exists()
    assert len(calls) == 1
    assert set(calls[0][0]) == {"^VIX", "DX-Y.NYB"}

    path_again = global_risk.fetch_global_risk_raw(start, end)
    assert path_again == path
    assert len(calls) == 1  # cache hit, sem novo download


def test_parse_raw_file_builds_expected_dataframe(tmp_path):
    path = tmp_path / "sample.parquet"
    SAMPLE_DF.to_parquet(path)

    df = global_risk._parse_raw_file(path)

    assert list(df.columns) == ["date", "vix_close", "dxy_close"]
    assert len(df) == 2
    assert str(df["date"].dt.tz) == global_risk.TIMEZONE
    assert df["vix_close"].tolist() == [13.5, 14.2]
    assert df["dxy_close"].tolist() == [102.1, 101.8]


def test_load_global_risk_processed_writes_parquet(monkeypatch):
    monkeypatch.setattr(global_risk.yf, "download", lambda tickers, **kwargs: SAMPLE_DF.copy())

    df = global_risk.load_global_risk_processed(date(2024, 1, 2), date(2024, 1, 3))

    assert global_risk.PROCESSED_PATH.exists()
    assert len(df) == 2

    reloaded = pd.read_parquet(global_risk.PROCESSED_PATH)
    assert len(reloaded) == len(df)
