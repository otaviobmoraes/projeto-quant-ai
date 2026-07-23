import json
from datetime import date

import pandas as pd
import pytest

from credibility import data_focus

FOCUS_SAMPLE = {
    "value": [
        {
            "Data": "2023-07-03",
            "Suavizada": "N",
            "Media": 3.9829,
            "Mediana": 3.9272,
            "DesvioPadrao": 0.4690,
            "numeroRespondentes": 137,
            "baseCalculo": 0,
        },
        {
            "Data": "2023-07-03",
            "Suavizada": "S",
            "Media": 4.2210,
            "Mediana": 4.2035,
            "DesvioPadrao": 0.4554,
            "numeroRespondentes": 137,
            "baseCalculo": 0,
        },
        {
            "Data": "2023-07-03",
            "Suavizada": "N",
            "Media": 3.8925,
            "Mediana": 3.8803,
            "DesvioPadrao": 0.4254,
            "numeroRespondentes": 76,
            "baseCalculo": 1,
        },
        {
            "Data": "2023-07-04",
            "Suavizada": "N",
            "Media": 3.9769,
            "Mediana": 3.9169,
            "DesvioPadrao": 0.4694,
            "numeroRespondentes": 137,
            "baseCalculo": 0,
        },
    ]
}

META_SAMPLE = [
    {"data": "01/01/2023", "valor": "3.25"},
    {"data": "01/01/2024", "valor": "3.00"},
]


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(data_focus, "RAW_DIR", tmp_path / "raw" / "focus")
    monkeypatch.setattr(data_focus, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(data_focus, "PROCESSED_PATH", tmp_path / "processed" / "focus.parquet")
    yield


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return json.loads(self.text)


def test_fetch_focus_raw_calls_api_and_caches(monkeypatch):
    calls = []

    def fake_get(url, timeout=None):
        calls.append(url)
        return _FakeResponse(json.dumps(FOCUS_SAMPLE))

    monkeypatch.setattr(data_focus.requests, "get", fake_get)

    start, end = date(2023, 7, 1), date(2023, 7, 10)
    path = data_focus.fetch_focus_raw(start, end)

    assert path.exists()
    assert len(calls) == 1
    # Filtro tem que ir com espaco %20 (nao '+'), senao a API do BCB devolve
    # 400 -- ver comentario em fetch_focus_raw.
    assert "Indicador%20eq%20%27IPCA%27" in calls[0]
    assert "+" not in calls[0]

    path_again = data_focus.fetch_focus_raw(start, end)
    assert path_again == path
    assert len(calls) == 1


def test_parse_focus_raw_filters_baseCalculo_and_suavizada(tmp_path):
    path = tmp_path / "focus.json"
    path.write_text(json.dumps(FOCUS_SAMPLE), encoding="utf-8")

    df = data_focus._parse_focus_raw(path)

    # So devem sobrar as linhas Suavizada=N e baseCalculo=0 (2 das 4 amostras)
    assert len(df) == 2
    assert str(df["date"].dt.tz) == data_focus.TIMEZONE
    assert df["pi_expectation"].tolist() == [3.9829, 3.9769]


def test_parse_meta_raw_builds_year_dict(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(META_SAMPLE), encoding="utf-8")

    meta = data_focus._parse_meta_raw(path)

    assert meta == {2023: 3.25, 2024: 3.00}


def test_blended_meta_weights_two_years_correctly():
    # Horizonte comecando bem perto do fim do ano corrente: quase todo o
    # horizonte de 12 meses cai no ano seguinte.
    d = pd.Timestamp("2023-12-20", tz=data_focus.TIMEZONE)
    meta = data_focus._blended_meta(d, {2023: 4.0, 2024: 3.0})

    assert 3.0 < meta < 3.2  # pesado pro ano seguinte (2024)


def test_blended_meta_weights_mostly_current_year_near_start():
    # Horizonte comecando logo no inicio do ano: quase todo o horizonte de
    # 12 meses cai no ano corrente.
    d = pd.Timestamp("2023-01-05", tz=data_focus.TIMEZONE)
    meta = data_focus._blended_meta(d, {2023: 4.0, 2024: 3.0})

    assert 3.8 < meta < 4.0


def test_blended_meta_falls_back_to_current_year_if_next_missing():
    d = pd.Timestamp("2023-12-20", tz=data_focus.TIMEZONE)
    meta = data_focus._blended_meta(d, {2023: 4.0})
    assert meta == pytest.approx(4.0)


def test_load_focus_processed_computes_gap(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if "Expectativas" in url:
            return _FakeResponse(json.dumps(FOCUS_SAMPLE))
        return _FakeResponse(json.dumps(META_SAMPLE))

    monkeypatch.setattr(data_focus.requests, "get", fake_get)

    df = data_focus.load_focus_processed(date(2023, 7, 1), date(2023, 7, 10))

    assert data_focus.PROCESSED_PATH.exists()
    assert "gap" in df.columns
    assert (df["gap"] == (df["pi_expectation"] - df["meta"])).all()

    reloaded = pd.read_parquet(data_focus.PROCESSED_PATH)
    assert len(reloaded) == len(df)
