from datetime import date

import pandas as pd
import pytest

from data import fx_spot

SAMPLE_INDEX = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
SAMPLE_DF = pd.DataFrame(
    {
        "Open": [4.85, 4.90],
        "High": [4.88, 4.93],
        "Low": [4.83, 4.89],
        "Close": [4.8520, 4.9010],
        "Adj Close": [4.8520, 4.9010],
        "Volume": [0, 0],
    },
    index=SAMPLE_INDEX,
)


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(fx_spot, "RAW_DIR", tmp_path / "raw" / "fx_spot")
    monkeypatch.setattr(fx_spot, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(fx_spot, "PROCESSED_PATH", tmp_path / "processed" / "fx_spot.parquet")
    yield


def test_fetch_fx_spot_raw_calls_yfinance_and_caches(monkeypatch):
    calls = []

    def fake_download(ticker, **kwargs):
        calls.append((ticker, kwargs))
        return SAMPLE_DF.copy()

    monkeypatch.setattr(fx_spot.yf, "download", fake_download)

    start, end = date(2024, 1, 2), date(2024, 1, 3)
    path = fx_spot.fetch_fx_spot_raw(start, end)

    assert path.exists()
    assert len(calls) == 1
    assert calls[0][0] == "BRL=X"

    # Segunda chamada com mesmo intervalo deve usar o cache, sem novo download.
    path_again = fx_spot.fetch_fx_spot_raw(start, end)
    assert path_again == path
    assert len(calls) == 1


def test_parse_raw_file_builds_expected_dataframe(tmp_path):
    path = tmp_path / "sample.parquet"
    SAMPLE_DF.to_parquet(path)

    df = fx_spot._parse_raw_file(path)

    assert list(df.columns) == ["date", "close"]
    assert len(df) == 2
    assert str(df["date"].dt.tz) == fx_spot.TIMEZONE
    assert df["close"].tolist() == [4.8520, 4.9010]


def test_load_fx_spot_processed_writes_parquet(monkeypatch):
    def fake_download(ticker, **kwargs):
        return SAMPLE_DF.copy()

    monkeypatch.setattr(fx_spot.yf, "download", fake_download)

    df = fx_spot.load_fx_spot_processed(date(2024, 1, 2), date(2024, 1, 3))

    assert fx_spot.PROCESSED_PATH.exists()
    assert len(df) == 2

    reloaded = pd.read_parquet(fx_spot.PROCESSED_PATH)
    assert len(reloaded) == len(df)
