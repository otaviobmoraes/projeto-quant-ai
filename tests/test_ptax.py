import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data import ptax

SGS_SAMPLE = [
    {"data": "02/01/2024", "valor": "4.8520"},
    {"data": "03/01/2024", "valor": "4.9010"},
]


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    """Redireciona RAW_DIR/PROCESSED_DIR pro tmp_path pra nao sujar data/ real."""
    monkeypatch.setattr(ptax, "RAW_DIR", tmp_path / "raw" / "ptax")
    monkeypatch.setattr(ptax, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(ptax, "PROCESSED_PATH", tmp_path / "processed" / "ptax.parquet")
    yield


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_ptax_raw_calls_api_and_caches(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        return _FakeResponse(SGS_SAMPLE)

    monkeypatch.setattr(ptax.requests, "get", fake_get)

    start, end = date(2024, 1, 1), date(2024, 1, 3)
    paths = ptax.fetch_ptax_raw(start, end, series="venda")

    assert len(paths) == 1
    assert paths[0].exists()
    assert json.loads(paths[0].read_text()) == SGS_SAMPLE
    assert len(calls) == 1
    assert "bcdata.sgs.1" in calls[0][0]

    # Segunda chamada com mesmo intervalo deve usar o cache, sem novo request.
    paths_again = ptax.fetch_ptax_raw(start, end, series="venda")
    assert paths_again == paths
    assert len(calls) == 1


def test_fetch_ptax_raw_uses_compra_series_code(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return _FakeResponse(SGS_SAMPLE)

    monkeypatch.setattr(ptax.requests, "get", fake_get)

    ptax.fetch_ptax_raw(date(2024, 1, 1), date(2024, 1, 3), series="compra")
    assert "bcdata.sgs.10813" in calls[0]


def test_parse_raw_file_builds_expected_dataframe(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(SGS_SAMPLE), encoding="utf-8")

    df = ptax._parse_raw_file(path, series="venda")

    assert list(df.columns) == ["date", "value", "tipo"]
    assert len(df) == 2
    assert df["tipo"].unique().tolist() == ["venda"]
    assert str(df["date"].dt.tz) == ptax.TIMEZONE
    assert df["value"].tolist() == [4.8520, 4.9010]


def test_load_ptax_processed_writes_parquet(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(SGS_SAMPLE)

    monkeypatch.setattr(ptax.requests, "get", fake_get)

    df = ptax.load_ptax_processed(date(2024, 1, 1), date(2024, 1, 3), series=("venda", "compra"))

    assert ptax.PROCESSED_PATH.exists()
    assert set(df["tipo"].unique()) == {"venda", "compra"}
    assert len(df) == 4  # 2 datas x 2 series

    reloaded = pd.read_parquet(ptax.PROCESSED_PATH)
    assert len(reloaded) == len(df)


def test_load_ptax_processed_deduplicates(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(SGS_SAMPLE)

    monkeypatch.setattr(ptax.requests, "get", fake_get)

    # Duas janelas sobrepostas (mesmo range chamado duas vezes) nao devem duplicar linhas.
    ptax.fetch_ptax_raw(date(2024, 1, 1), date(2024, 1, 3), series="venda")
    df = ptax.load_ptax_processed(date(2024, 1, 1), date(2024, 1, 3), series=("venda",))

    assert len(df) == 2
