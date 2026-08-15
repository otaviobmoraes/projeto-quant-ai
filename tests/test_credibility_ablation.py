import numpy as np
import pandas as pd
import pytest

from credibility import ablation


def _price_series(n: int, seed: int = 0, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="America/Sao_Paulo")
    prices = 5.0 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    return pd.Series(prices, index=idx)


def _credibility_df(idx: pd.DatetimeIndex, theta, dispersion) -> pd.DataFrame:
    return pd.DataFrame({"date": idx, "gap": 0.0, "dispersion": dispersion, "theta_baseline": theta})


def test_build_dataset_with_credibility_aligns_by_ffill():
    prices = _price_series(60)
    cred_idx = prices.index[::3]  # credibilidade so em dias alternados
    credibility_df = _credibility_df(cred_idx, theta=0.5, dispersion=0.4)

    dataset = ablation.build_dataset_with_credibility(prices, credibility_df, horizon=5)

    assert "theta_baseline" in dataset.columns
    assert "dispersion" in dataset.columns
    assert not dataset.isna().any().any()


def test_run_credibility_ablation_structure():
    prices = _price_series(400, seed=1)
    idx = prices.index
    credibility_df = _credibility_df(idx, theta=0.5, dispersion=0.4)

    result = ablation.run_credibility_ablation(prices, credibility_df, horizon=21, n_splits=4, embargo_days=5)

    assert "per_fold" in result
    assert set(result["baseline"]) == {"rmse", "mae", "r2_oos", "vies_pct"}
    assert set(result["com_credibilidade"]) == {"rmse", "mae", "r2_oos", "vies_pct"}
    assert set(result["baseline_pooled"]) == {"rmse", "mae", "r2_oos", "n_obs"}
    assert "com_credibilidade_pooled" in result


def test_run_credibility_ablation_recovers_signal_when_theta_is_informative():
    prices = _price_series(400, seed=1)
    from vol.forecast import build_dataset

    dataset_baseline = build_dataset(prices, horizon=21)
    rng = np.random.default_rng(2)
    informative_theta = dataset_baseline["target"] + rng.normal(0, 0.01, len(dataset_baseline))
    theta_series = informative_theta.reindex(prices.index)

    credibility_df = pd.DataFrame(
        {
            "date": prices.index,
            "gap": 0.0,
            "dispersion": 0.4,
            "theta_baseline": theta_series.to_numpy(),
        }
    )

    result = ablation.run_credibility_ablation(prices, credibility_df, horizon=21, n_splits=4, embargo_days=5)

    assert result["baseline"]["r2_oos"] < result["com_credibilidade"]["r2_oos"]


def test_run_credibility_ablation_raises_when_dataset_too_small():
    prices = _price_series(40, seed=3)
    credibility_df = _credibility_df(prices.index, theta=0.5, dispersion=0.4)

    with pytest.raises(ValueError):
        ablation.run_credibility_ablation(prices, credibility_df, horizon=21, n_splits=10, embargo_days=5)


def test_load_and_run_credibility_ablation_reads_processed_parquets(tmp_path, monkeypatch):
    prices = _price_series(400, seed=4)
    ptax_df = pd.DataFrame({"date": prices.index, "value": prices.to_numpy(), "tipo": "venda"})
    ptax_path = tmp_path / "ptax.parquet"
    ptax_df.to_parquet(ptax_path)

    credibility_df = _credibility_df(prices.index, theta=0.5, dispersion=0.4)
    credibility_path = tmp_path / "credibility.parquet"
    credibility_df.to_parquet(credibility_path)

    import vol.realized as realized_module
    monkeypatch.setattr(realized_module, "PTAX_PROCESSED_PATH", ptax_path)
    monkeypatch.setattr(ablation, "CREDIBILITY_PROCESSED_PATH", credibility_path)

    result = ablation.load_and_run_credibility_ablation(
        horizon=21, n_splits=3, embargo_days=5, source="ptax"
    )

    assert "baseline" in result and "com_credibilidade" in result


def test_load_and_run_credibility_ablation_source_yfinance_reads_fx_spot(tmp_path, monkeypatch):
    prices = _price_series(400, seed=6)
    credibility_df = _credibility_df(prices.index, theta=0.5, dispersion=0.4)
    credibility_path = tmp_path / "credibility.parquet"
    credibility_df.to_parquet(credibility_path)

    rng = np.random.default_rng(10)
    noise = np.abs(rng.normal(0, 0.003, len(prices)))
    fx_df = pd.DataFrame(
        {
            "date": prices.index,
            "open": prices.to_numpy(),
            "high": prices.to_numpy() * (1 + noise),
            "low": prices.to_numpy() * (1 - noise),
            "close": prices.to_numpy(),
        }
    )
    fx_path = tmp_path / "fx_spot.parquet"
    fx_df.to_parquet(fx_path)

    import vol.realized as realized_module

    monkeypatch.setattr(ablation, "CREDIBILITY_PROCESSED_PATH", credibility_path)
    monkeypatch.setattr(realized_module, "FX_SPOT_PROCESSED_PATH", fx_path)

    result = ablation.load_and_run_credibility_ablation(
        horizon=21, n_splits=3, embargo_days=5, source="yfinance"
    )

    assert "baseline" in result and "com_credibilidade" in result


def test_load_and_run_credibility_ablation_missing_files_raise(tmp_path, monkeypatch):
    import vol.realized as realized_module
    monkeypatch.setattr(realized_module, "PTAX_PROCESSED_PATH", tmp_path / "missing_ptax.parquet")
    monkeypatch.setattr(ablation, "CREDIBILITY_PROCESSED_PATH", tmp_path / "missing_cred.parquet")

    with pytest.raises(FileNotFoundError):
        ablation.load_and_run_credibility_ablation(source="ptax")
