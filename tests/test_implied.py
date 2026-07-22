from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from vol import implied


def _smile(refdate: date, points: list[tuple[int, float]]) -> pd.DataFrame:
    """points: lista de (dias corridos ate o vencimento, iv_pct)."""
    return pd.DataFrame(
        {
            "maturity_date": [refdate + timedelta(days=d) for d, _ in points],
            "iv_pct": [iv for _, iv in points],
        }
    )


def test_interpolate_atm_iv_exact_match_on_existing_maturity():
    refdate = date(2026, 7, 21)
    smile = _smile(refdate, [(13, 9.16), (42, 10.15), (72, 10.95)])

    assert implied.interpolate_atm_iv(smile, refdate, 42) == pytest.approx(10.15)


def test_interpolate_atm_iv_between_two_maturities_uses_variance_interpolation():
    refdate = date(2026, 7, 21)
    smile = _smile(refdate, [(10, 10.0), (30, 20.0)])

    target_days = 20
    result = implied.interpolate_atm_iv(smile, refdate, target_days)

    var1 = (10.0 / 100) ** 2 * 10
    var2 = (20.0 / 100) ** 2 * 30
    frac = (20 - 10) / (30 - 10)
    expected_var = var1 + frac * (var2 - var1)
    expected_iv = np.sqrt(expected_var / target_days) * 100

    assert result == pytest.approx(expected_iv)
    # Interpolar em variancia nao e o mesmo que interpolar a vol linearmente.
    assert result != pytest.approx((10.0 + 20.0) / 2)


def test_interpolate_atm_iv_flat_extrapolates_beyond_range():
    refdate = date(2026, 7, 21)
    smile = _smile(refdate, [(13, 9.16), (42, 10.15)])

    assert implied.interpolate_atm_iv(smile, refdate, 5) == pytest.approx(9.16)
    assert implied.interpolate_atm_iv(smile, refdate, 200) == pytest.approx(10.15)


def test_atm_iv_series_filters_delta_50_and_groups_by_refdate():
    refdate1 = pd.Timestamp("2026-07-21", tz="America/Sao_Paulo")
    refdate2 = pd.Timestamp("2026-07-22", tz="America/Sao_Paulo")
    df = pd.DataFrame(
        {
            "refdate": [refdate1, refdate1, refdate1, refdate2, refdate2],
            "maturity_date": [
                date(2026, 8, 3),
                date(2026, 8, 3),
                date(2026, 9, 1),
                date(2026, 8, 4),
                date(2026, 9, 2),
            ],
            "delta_pct": [50.0, 25.0, 50.0, 50.0, 50.0],
            "iv_pct": [9.16, 10.5, 10.15, 9.20, 10.20],
        }
    )

    result = implied.atm_iv_series(df, target_days=21)

    assert len(result) == 2
    assert list(result.columns) == ["refdate", "target_date", "iv_atm_pct"]
    assert result["refdate"].tolist() == [refdate1, refdate2]


def test_load_iv_atm_processed_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(implied, "IV_PROCESSED_PATH", tmp_path / "missing.parquet")

    with pytest.raises(FileNotFoundError):
        implied.load_iv_atm_processed()


def test_load_iv_atm_processed_reads_parquet(tmp_path, monkeypatch):
    refdate = pd.Timestamp("2026-07-21", tz="America/Sao_Paulo")
    df = pd.DataFrame(
        {
            "refdate": [refdate, refdate],
            "maturity_date": [date(2026, 8, 3), date(2026, 9, 1)],
            "delta_pct": [50.0, 50.0],
            "iv_pct": [9.16, 10.15],
        }
    )
    path = tmp_path / "iv_surface.parquet"
    df.to_parquet(path)
    monkeypatch.setattr(implied, "IV_PROCESSED_PATH", path)

    result = implied.load_iv_atm_processed(target_days=13)
    assert len(result) == 1
    assert result["iv_atm_pct"].iloc[0] == pytest.approx(9.16)
