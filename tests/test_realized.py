import numpy as np
import pandas as pd
import pytest

from vol import realized


def test_log_returns_matches_manual_calc():
    prices = pd.Series([5.0, 5.1, 5.0, 5.2])
    r = realized.log_returns(prices)

    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(5.1 / 5.0))
    assert r.iloc[2] == pytest.approx(np.log(5.0 / 5.1))
    assert r.iloc[3] == pytest.approx(np.log(5.2 / 5.0))


def test_realized_vol_constant_returns_is_zero():
    # Retornos log constantes -> desvio padrao zero -> RV zero (a partir da janela).
    prices = pd.Series([5.0 * (1.01**i) for i in range(10)])
    rv = realized.realized_vol(prices, window=3)

    assert rv.iloc[3:].abs().max() < 1e-8


def test_realized_vol_scales_with_annualization_factor():
    rng = np.random.default_rng(42)
    prices = pd.Series(5.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))))

    rv_252 = realized.realized_vol(prices, window=21, annualization_factor=252)
    rv_1 = realized.realized_vol(prices, window=21, annualization_factor=1)

    ratio = (rv_252 / rv_1).dropna()
    assert ratio.apply(lambda x: x == pytest.approx(np.sqrt(252), rel=1e-6)).all()


def test_load_ptax_realized_vol_reads_processed_parquet(tmp_path, monkeypatch):
    dates = pd.date_range("2026-01-01", periods=30, freq="B", tz="America/Sao_Paulo")
    rng = np.random.default_rng(0)
    prices = 5.0 * np.exp(np.cumsum(rng.normal(0, 0.005, len(dates))))
    df = pd.DataFrame(
        {
            "date": list(dates) + list(dates),
            "value": list(prices) + list(prices * 1.001),
            "tipo": ["venda"] * len(dates) + ["compra"] * len(dates),
        }
    )
    path = tmp_path / "ptax.parquet"
    df.to_parquet(path)
    monkeypatch.setattr(realized, "PTAX_PROCESSED_PATH", path)

    rv = realized.load_ptax_realized_vol(window=5, tipo="venda")

    assert list(rv.columns) == ["rv_pct"]
    assert len(rv) == len(dates)
    assert rv["rv_pct"].iloc[:5].isna().all() or rv["rv_pct"].isna().sum() >= 4


def test_load_ptax_realized_vol_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(realized, "PTAX_PROCESSED_PATH", tmp_path / "missing.parquet")

    with pytest.raises(FileNotFoundError):
        realized.load_ptax_realized_vol()
