import io
import zipfile
from datetime import date, datetime

import openpyxl
import pandas as pd
import pytest

from data import iv_surface


def _build_sample_zip(refdate: date) -> bytes:
    """Monta um .zip com uma xlsx no mesmo layout publicado pela B3 (celulas
    de data como datetime, igual ao arquivo real)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append([datetime.combine(refdate, datetime.min.time()), 1, 5, 50, 95, 99])
    ws.append([datetime(2026, 8, 3), 13.44, 12.49, 9.16, 8.58, 9.09])
    ws.append([datetime(2026, 9, 1), 14.35, 13.90, 9.55, 9.02, 9.04])

    xlsx_buf = io.BytesIO()
    wb.save(xlsx_buf)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("Superficie_Vol_VTC_dummy.xlsx", xlsx_buf.getvalue())
    return zip_buf.getvalue()


SAMPLE_ZIP = _build_sample_zip(date(2026, 7, 21))


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(iv_surface, "RAW_DIR", tmp_path / "raw" / "iv_surface")
    monkeypatch.setattr(iv_surface, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(iv_surface, "PROCESSED_PATH", tmp_path / "processed" / "iv_surface.parquet")
    yield


class _FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


def test_parse_workbook_reads_refdate_and_long_format():
    xlsx_bytes = iv_surface._extract_xlsx_bytes(SAMPLE_ZIP)
    refdate, df = iv_surface._parse_workbook(xlsx_bytes)

    assert refdate == date(2026, 7, 21)
    assert list(df.columns) == ["maturity_date", "delta_pct", "iv_pct"]
    assert len(df) == 10  # 2 vencimentos x 5 deltas
    assert df["maturity_date"].unique().tolist() == [date(2026, 8, 3), date(2026, 9, 1)]
    assert set(df["delta_pct"]) == {1.0, 5.0, 50.0, 95.0, 99.0}

    atm = df[(df["maturity_date"] == date(2026, 8, 3)) & (df["delta_pct"] == 50.0)]
    assert atm["iv_pct"].iloc[0] == 9.16


def test_fetch_iv_surface_raw_caches_by_refdate(monkeypatch):
    calls = []

    def fake_get(url, timeout=None):
        calls.append(url)
        return _FakeResponse(SAMPLE_ZIP)

    monkeypatch.setattr(iv_surface.requests, "get", fake_get)

    refdate, path = iv_surface.fetch_iv_surface_raw()
    assert refdate == date(2026, 7, 21)
    assert path.exists()
    assert len(calls) == 1

    # B3 sempre serve o snapshot mais recente na mesma URL; se a data nao mudou,
    # nao devemos sobrescrever o raw file (mas o request em si sempre acontece,
    # pois nao ha como saber a data sem baixar).
    refdate_again, path_again = iv_surface.fetch_iv_surface_raw()
    assert refdate_again == refdate
    assert path_again == path
    assert len(calls) == 2


def test_load_iv_surface_processed_upserts_history(monkeypatch):
    responses = iter([SAMPLE_ZIP, _build_sample_zip(date(2026, 7, 22))])
    monkeypatch.setattr(iv_surface.requests, "get", lambda url, timeout=None: _FakeResponse(next(responses)))

    df_day1 = iv_surface.load_iv_surface_processed()
    assert set(df_day1["refdate"].dt.date) == {date(2026, 7, 21)}
    assert str(df_day1["refdate"].dt.tz) == iv_surface.TIMEZONE

    df_day2 = iv_surface.load_iv_surface_processed()
    assert set(df_day2["refdate"].dt.date) == {date(2026, 7, 21), date(2026, 7, 22)}

    reloaded = pd.read_parquet(iv_surface.PROCESSED_PATH)
    assert len(reloaded) == len(df_day2)
