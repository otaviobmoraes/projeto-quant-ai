import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from credibility import data_gap

IBCBR_SAMPLE = [
    {"data": "01/01/2020", "valor": "100.0"},
    {"data": "01/02/2020", "valor": "101.0"},
    {"data": "01/03/2020", "valor": "90.0"},
    {"data": "01/04/2020", "valor": "80.0"},
]


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(data_gap, "RAW_DIR", tmp_path / "raw" / "ibcbr")
    monkeypatch.setattr(data_gap, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(data_gap, "PROCESSED_PATH", tmp_path / "processed" / "output_gap.parquet")
    yield


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_ibcbr_raw_caches_by_window(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        return _FakeResponse(IBCBR_SAMPLE)

    monkeypatch.setattr(data_gap.requests, "get", fake_get)

    start, end = date(2020, 1, 1), date(2020, 4, 30)
    paths = data_gap.fetch_ibcbr_raw(start, end)

    assert len(paths) == 1
    assert paths[0].exists()
    assert len(calls) == 1

    paths_again = data_gap.fetch_ibcbr_raw(start, end)
    assert paths_again == paths
    assert len(calls) == 1


def test_parse_ibcbr_raw_computes_release_date(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(IBCBR_SAMPLE), encoding="utf-8")

    df = data_gap._parse_ibcbr_raw([path])

    assert len(df) == 4
    assert str(df["reference_date"].dt.tz) == data_gap.TIMEZONE
    # release_date sempre POSTERIOR a reference_date (defasagem de divulgacao)
    assert (df["release_date"] > df["reference_date"]).all()
    lag_days = (df["release_date"] - df["reference_date"]).dt.days
    assert (lag_days == data_gap.PUBLICATION_LAG_DAYS).all()


def test_expanding_hp_gap_only_uses_past_data():
    rng = np.random.default_rng(0)
    log_value = pd.Series(np.log(100 + np.cumsum(rng.normal(0, 1, 60))))

    gap_short = data_gap.expanding_hp_gap(log_value.iloc[:40], min_window=24)
    gap_full = data_gap.expanding_hp_gap(log_value, min_window=24)

    # O gap no ponto 35, calculado so com dados ate 40, tem que ser IGUAL ao
    # calculado com a serie inteira ate 60 -- se dependesse do futuro, os
    # dois teriam valores diferentes no mesmo ponto.
    assert gap_short.iloc[35] == pytest.approx(gap_full.iloc[35])


def test_expanding_hp_gap_nan_before_min_window():
    log_value = pd.Series(np.log(np.full(30, 100.0)))
    gap = data_gap.expanding_hp_gap(log_value, min_window=24)

    assert gap.iloc[:24].isna().all()
    assert gap.iloc[24:].notna().all()


def test_load_output_gap_processed_filters_to_requested_range(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(IBCBR_SAMPLE)

    monkeypatch.setattr(data_gap.requests, "get", fake_get)

    df = data_gap.load_output_gap_processed(date(2020, 2, 1), date(2020, 3, 31))

    assert data_gap.PROCESSED_PATH.exists()
    assert len(df) == 2
    assert df["reference_date"].min() >= pd.Timestamp("2020-02-01", tz=data_gap.TIMEZONE)

    full_history = pd.read_parquet(data_gap.PROCESSED_PATH)
    assert len(full_history) == 4  # parquet salvo guarda o historico completo
