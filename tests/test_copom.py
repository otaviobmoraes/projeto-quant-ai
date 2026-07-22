import json
from datetime import date

import pandas as pd
import pytest

from data import copom

ATAS_LIST_SAMPLE = {
    "conteudo": [
        {
            "nroReuniao": 279,
            "dataReferencia": "2026-06-17",
            "dataPublicacao": "2026-06-23",
            "titulo": "279ª Reunião - 16-17 junho, 2026",
        }
    ]
}

ATA_DETAIL_SAMPLE = {
    "conteudo": [
        {
            "nroReuniao": 279,
            "dataReferencia": "2026-06-17",
            "dataPublicacao": "2026-06-23",
            "titulo": "279ª Reunião - 16-17 junho, 2026",
            "urlPdfAta": "https://www.bcb.gov.br/content/copom/atascopom/Copom279.pdf",
            "textoAta": "<div><p class=\"paragrafo\">O Comite decidiu manter a taxa Selic.</p></div>",
        }
    ]
}

COMUNICADOS_LIST_SAMPLE = {
    "conteudo": [
        {
            "nro_reuniao": 279,
            "dataReferencia": "2026-06-17",
            "titulo": "279ª reunião - Copom reduz a taxa Selic para 14,25% a.a.",
        }
    ]
}

COMUNICADO_DETAIL_SAMPLE = {
    "conteudo": [
        {
            "nro_reuniao": 279,
            "dataReferencia": "2026-06-17",
            "titulo": "Copom reduz a taxa Selic para 14,25% a.a.",
            "textoComunicado": "<p>O ambiente externo &eacute; incerto.</p>",
        }
    ]
}


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(copom, "RAW_DIR", tmp_path / "raw" / "copom")
    monkeypatch.setattr(copom, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(copom, "PROCESSED_PATH", tmp_path / "processed" / "copom.parquet")
    yield


class _FakeResponse:
    def __init__(self, payload):
        self.text = json.dumps(payload)

    def raise_for_status(self):
        pass


def _fake_get(payload_by_path):
    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(payload_by_path[url])

    return fake_get


def test_strip_html_removes_tags_and_unescapes_entities():
    text = copom._strip_html("<p>O ambiente externo &eacute; incerto.</p>")
    assert text == "O ambiente externo é incerto."


def test_fetch_copom_list_raw_caches_by_day(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return _FakeResponse(ATAS_LIST_SAMPLE)

    monkeypatch.setattr(copom.requests, "get", fake_get)

    today = date(2026, 6, 24)
    path = copom.fetch_copom_list_raw("atas", quantidade=10, as_of=today)
    assert path.exists()
    assert len(calls) == 1

    path_again = copom.fetch_copom_list_raw("atas", quantidade=10, as_of=today)
    assert path_again == path
    assert len(calls) == 1  # mesmo dia -> cache


def test_fetch_copom_detail_raw_caches_forever(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return _FakeResponse(ATA_DETAIL_SAMPLE)

    monkeypatch.setattr(copom.requests, "get", fake_get)

    path = copom.fetch_copom_detail_raw("atas", 279)
    path_again = copom.fetch_copom_detail_raw("atas", 279)
    assert path == path_again
    assert len(calls) == 1


def test_parse_detail_raw_ata_and_comunicado(tmp_path):
    ata_path = tmp_path / "ata.json"
    ata_path.write_text(json.dumps(ATA_DETAIL_SAMPLE), encoding="utf-8")
    record = copom._parse_detail_raw(ata_path, "atas")
    assert record["nro_reuniao"] == 279
    assert record["tipo"] == "ata"
    assert record["texto"] == "O Comite decidiu manter a taxa Selic."
    assert record["url_pdf"].endswith(".pdf")

    com_path = tmp_path / "comunicado.json"
    com_path.write_text(json.dumps(COMUNICADO_DETAIL_SAMPLE), encoding="utf-8")
    record2 = copom._parse_detail_raw(com_path, "comunicados")
    assert record2["tipo"] == "comunicado"
    assert record2["texto"] == "O ambiente externo é incerto."
    assert record2["url_pdf"] is None


def test_load_copom_processed_builds_corpus(monkeypatch):
    urls = {
        f"{copom.BASE_URL}/atas": ATAS_LIST_SAMPLE,
        f"{copom.BASE_URL}/atas_detalhes": ATA_DETAIL_SAMPLE,
        f"{copom.BASE_URL}/comunicados": COMUNICADOS_LIST_SAMPLE,
        f"{copom.BASE_URL}/comunicados_detalhes": COMUNICADO_DETAIL_SAMPLE,
    }
    monkeypatch.setattr(copom.requests, "get", _fake_get(urls))

    df = copom.load_copom_processed(quantidade=5)

    assert copom.PROCESSED_PATH.exists()
    assert len(df) == 2
    assert set(df["tipo"]) == {"ata", "comunicado"}
    assert str(df["data_referencia"].dt.tz) == copom.TIMEZONE

    reloaded = pd.read_parquet(copom.PROCESSED_PATH)
    assert len(reloaded) == 2

    # Rodar de novo (mesmo dia, mesmas reunioes) nao deve duplicar linhas.
    df_again = copom.load_copom_processed(quantidade=5)
    assert len(df_again) == 2
