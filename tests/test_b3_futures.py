import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from data import b3_futures


def _pric_rpt(ticker: str, **fields) -> str:
    body = "".join(f'<{k} Ccy="BRL">{v}</{k}>' for k, v in fields.items())
    return (
        "<PricRpt>"
        f"<TradDt><Dt>2026-08-04</Dt></TradDt>"
        f"<SctyId><TckrSymb>{ticker}</TckrSymb></SctyId>"
        f"<FinInstrmAttrbts>{body}</FinInstrmAttrbts>"
        "</PricRpt>"
    )


def _bundle(*records: str) -> bytes:
    """Monta um pacote no mesmo formato da B3: zip -> zip -> XMLs."""
    xml = ('<?xml version="1.0"?><Document>' + "".join(records) + "</Document>").encode()

    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w") as inner:
        # 2 snapshots: o primeiro incompleto, o ultimo (ordem alfabetica) completo
        inner.writestr("BVBG.086.01_BV0001.xml", b"<Document/>")
        inner.writestr("BVBG.086.01_BV0002.xml", xml)

    outer_buf = io.BytesIO()
    with zipfile.ZipFile(outer_buf, "w") as outer:
        outer.writestr("PR260804.zip", inner_buf.getvalue())
    return outer_buf.getvalue()


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(b3_futures, "RAW_DIR", tmp_path / "raw" / "b3_futures")
    monkeypatch.setattr(b3_futures, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(b3_futures, "PROCESSED_PATH", tmp_path / "processed" / "b3_dol_futures.parquet")
    yield


def test_parse_dol_futures_extracts_settlement_and_range():
    payload = _bundle(
        _pric_rpt("DOLU26", AdjstdQt=5171.47, MinPric=5105, MaxPric=5181, OpnIntrst=623043, TradQty=19927)
    )

    df = b3_futures.parse_dol_futures(payload, date(2026, 8, 4))

    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "DOLU26"
    assert row["settlement"] == pytest.approx(5171.47)
    assert row["low"] == pytest.approx(5105)
    assert row["high"] == pytest.approx(5181)
    assert str(df["date"].dt.tz) == b3_futures.TIMEZONE


def test_parse_dol_futures_extracts_full_ohlc():
    payload = _bundle(
        _pric_rpt("DOLU26", AdjstdQt=5171.47, FrstPric=5109.5, LastPric=5157.5, MinPric=5105, MaxPric=5181)
    )

    row = b3_futures.parse_dol_futures(payload, date(2026, 8, 4)).iloc[0]

    assert row["open"] == pytest.approx(5109.5)   # primeiro NEGOCIO do pregao
    assert row["last"] == pytest.approx(5157.5)   # ultimo NEGOCIO
    assert row["settlement"] == pytest.approx(5171.47)  # ajuste (nao e o ultimo negocio)


def test_parse_dol_futures_ignores_options():
    # Opcoes (sufixo C/P + strike) nao devem entrar -- so o futuro puro.
    payload = _bundle(
        _pric_rpt("DOLU26", AdjstdQt=5171.47),
        _pric_rpt("DOLU26C005600", AdjstdQt=42.0),
        _pric_rpt("DOLU26P005600", AdjstdQt=13.0),
    )

    df = b3_futures.parse_dol_futures(payload, date(2026, 8, 4))

    assert df["ticker"].tolist() == ["DOLU26"]


def test_parse_dol_futures_skips_contracts_without_settlement():
    payload = _bundle(
        _pric_rpt("DOLU26", AdjstdQt=5171.47),
        _pric_rpt("DOLF27", OpnIntrst=100),  # vencimento sem ajuste publicado
    )

    df = b3_futures.parse_dol_futures(payload, date(2026, 8, 4))

    assert df["ticker"].tolist() == ["DOLU26"]


def test_parse_dol_futures_reads_last_snapshot_not_first():
    payload = _bundle(_pric_rpt("DOLU26", AdjstdQt=5171.47))
    # o 1o XML do bundle e vazio de proposito; se o parser lesse ele, daria 0 linhas
    assert len(b3_futures.parse_dol_futures(payload, date(2026, 8, 4))) == 1


def test_front_month_series_picks_most_liquid_contract():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-04", tz=b3_futures.TIMEZONE)] * 2,
            "ticker": ["DOLU26", "DOLV26"],
            "settlement": [5171.47, 5200.0],
            "open": [5109.5, 5190.0],
            "last": [5157.5, 5199.0],
            "low": [5105.0, 5150.0],
            "high": [5181.0, 5210.0],
            "open_interest": [623043.0, 1000.0],
            "trades": [19927.0, 12.0],
        }
    )

    front = b3_futures.front_month_series(df)

    assert len(front) == 1
    assert front.iloc[0]["ticker"] == "DOLU26"  # o de maior numero de negocios


def test_front_month_series_flags_contract_change_on_roll():
    days = pd.date_range("2026-08-04", periods=3, freq="B", tz=b3_futures.TIMEZONE)
    df = pd.DataFrame(
        {
            "date": list(days),
            "ticker": ["DOLU26", "DOLU26", "DOLV26"],  # rola no 3o dia
            "settlement": [5171.0, 5180.0, 5250.0],
            "open": [5150.0, 5175.0, 5240.0],
            "last": [5170.0, 5179.0, 5249.0],
            "low": [5100.0, 5150.0, 5200.0],
            "high": [5200.0, 5210.0, 5300.0],
            "open_interest": [1.0, 1.0, 1.0],
            "trades": [10.0, 10.0, 10.0],
        }
    )

    front = b3_futures.front_month_series(df)

    assert front["contract_changed"].tolist() == [False, False, True]


def test_front_month_series_empty_input():
    result = b3_futures.front_month_series(pd.DataFrame())
    assert result.empty


def test_fetch_pr_raw_caches_and_skips_second_request(monkeypatch):
    calls = []
    payload = _bundle(_pric_rpt("DOLU26", AdjstdQt=5171.47))

    class _Response:
        content = payload

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=None):
        calls.append(url)
        return _Response()

    monkeypatch.setattr(b3_futures.requests, "get", fake_get)

    path = b3_futures.fetch_pr_raw(date(2026, 8, 4))
    assert path is not None and path.exists()
    assert len(calls) == 1

    b3_futures.fetch_pr_raw(date(2026, 8, 4))
    assert len(calls) == 1  # cache hit


def test_fetch_pr_raw_returns_none_on_holiday(monkeypatch):
    class _EmptyResponse:
        content = b"x" * 22  # a B3 devolve 200 com zip vazio em dia sem pregao

        def raise_for_status(self):
            pass

    monkeypatch.setattr(b3_futures.requests, "get", lambda url, timeout=None: _EmptyResponse())

    assert b3_futures.fetch_pr_raw(date(2026, 8, 4)) is None


def test_fetch_pr_bytes_retries_transient_error_then_succeeds(monkeypatch):
    calls = []
    payload = _bundle(_pric_rpt("DOLU26", AdjstdQt=5171.47))

    class _Response:
        content = payload

        def raise_for_status(self):
            pass

    def flaky_get(url, timeout=None):
        calls.append(url)
        if len(calls) == 1:
            raise b3_futures.requests.exceptions.ConnectionError("boom")
        return _Response()

    monkeypatch.setattr(b3_futures.requests, "get", flaky_get)
    monkeypatch.setattr(b3_futures.time, "sleep", lambda s: None)

    assert b3_futures.fetch_pr_bytes(date(2026, 8, 4)) == payload
    assert len(calls) == 2  # falhou uma vez, sucesso na segunda


def test_fetch_pr_bytes_raises_after_max_retries(monkeypatch):
    def always_fails(url, timeout=None):
        raise b3_futures.requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(b3_futures.requests, "get", always_fails)
    monkeypatch.setattr(b3_futures.time, "sleep", lambda s: None)

    with pytest.raises(b3_futures.requests.exceptions.RequestException):
        b3_futures.fetch_pr_bytes(date(2026, 8, 4))


def test_load_dol_futures_processed_keep_raw_false_writes_no_raw_files(monkeypatch):
    payload = _bundle(_pric_rpt("DOLU26", AdjstdQt=5171.47, TradQty=100))

    class _Response:
        content = payload

        def raise_for_status(self):
            pass

    monkeypatch.setattr(b3_futures.requests, "get", lambda url, timeout=None: _Response())

    df = b3_futures.load_dol_futures_processed(date(2026, 8, 6), date(2026, 8, 7), keep_raw=False)

    assert len(df) == 2
    assert b3_futures.PROCESSED_PATH.exists()
    # nenhum zip de ~10 MB deixado pra tras (o ponto de keep_raw=False)
    assert not b3_futures.RAW_DIR.exists() or not list(b3_futures.RAW_DIR.glob("*.zip"))


def test_load_dol_futures_processed_skips_holiday_without_crashing(monkeypatch):
    class _EmptyResponse:
        content = b"x" * 22

        def raise_for_status(self):
            pass

    monkeypatch.setattr(b3_futures.requests, "get", lambda url, timeout=None: _EmptyResponse())

    df = b3_futures.load_dol_futures_processed(date(2026, 8, 6), date(2026, 8, 7), keep_raw=True)

    assert df.empty  # nenhum pregao valido, mas sem excecao


def test_load_dol_futures_processed_writes_parquet_and_skips_weekend(monkeypatch):
    requested = []
    payload = _bundle(_pric_rpt("DOLU26", AdjstdQt=5171.47, MinPric=5105, MaxPric=5181, TradQty=100))

    class _Response:
        content = payload

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=None):
        requested.append(url)
        return _Response()

    monkeypatch.setattr(b3_futures.requests, "get", fake_get)

    # 2026-08-08 e sabado, 2026-08-09 domingo
    df = b3_futures.load_dol_futures_processed(date(2026, 8, 6), date(2026, 8, 10))

    assert b3_futures.PROCESSED_PATH.exists()
    assert len(requested) == 3  # qui, sex, seg -- fim de semana nem tenta a rede
    assert len(df) == 3
