import pandas as pd

from strategy import signal


def test_generate_signal_buys_vol_when_rv_above_iv():
    assert signal.generate_signal(rv_forecast_pct=12.0, iv_pct=9.0) == signal.LONG_VOL


def test_generate_signal_sells_vol_when_rv_below_iv():
    assert signal.generate_signal(rv_forecast_pct=7.0, iv_pct=9.0) == signal.SHORT_VOL


def test_generate_signal_no_trade_within_band():
    assert signal.generate_signal(rv_forecast_pct=9.5, iv_pct=9.0, band_pct=1.0) == signal.NO_TRADE
    assert signal.generate_signal(rv_forecast_pct=8.5, iv_pct=9.0, band_pct=1.0) == signal.NO_TRADE


def test_generate_signal_band_boundary_is_no_trade():
    # Exatamente na borda da banda -> ainda dentro (nao operar).
    assert signal.generate_signal(rv_forecast_pct=10.0, iv_pct=9.0, band_pct=1.0) == signal.NO_TRADE


def test_generate_signal_beyond_band_trades():
    assert signal.generate_signal(rv_forecast_pct=10.01, iv_pct=9.0, band_pct=1.0) == signal.LONG_VOL
    assert signal.generate_signal(rv_forecast_pct=7.99, iv_pct=9.0, band_pct=1.0) == signal.SHORT_VOL


def test_signal_series_aligns_by_date_and_applies_rule():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    rv = pd.Series([12.0, 7.0, 9.0, 9.0], index=idx)
    iv = pd.Series([9.0, 9.0, 9.0], index=idx[:3])  # falta um dia de proposito

    result = signal.signal_series(rv, iv)

    assert len(result) == 3  # join interno: so onde ha as duas series
    assert result.tolist() == [signal.LONG_VOL, signal.SHORT_VOL, signal.NO_TRADE]
