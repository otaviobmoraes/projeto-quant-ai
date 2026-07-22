import numpy as np
import pandas as pd
import pytest

from backtest import ablation


def _price_series(n: int, seed: int = 0, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="America/Sao_Paulo")
    prices = 5.0 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    return pd.Series(prices, index=idx)


def test_run_purged_ablation_structure():
    prices = _price_series(400, seed=1)
    from vol.forecast import build_dataset

    dataset_baseline = build_dataset(prices, horizon=21)
    rng = np.random.default_rng(2)
    informative_news = dataset_baseline["target"] + rng.normal(0, 0.01, len(dataset_baseline))
    news_series = informative_news.reindex(prices.index)

    result = ablation.run_purged_ablation(prices, news_series, horizon=21, n_splits=4, embargo_days=5)

    assert "per_fold" in result
    assert result["n_splits"] <= 4
    assert set(result["baseline"]) == {"rmse", "mae", "r2_oos"}


def test_run_purged_ablation_recovers_signal_when_news_is_informative():
    prices = _price_series(400, seed=1)
    from vol.forecast import build_dataset

    dataset_baseline = build_dataset(prices, horizon=21)
    rng = np.random.default_rng(2)
    informative_news = dataset_baseline["target"] + rng.normal(0, 0.01, len(dataset_baseline))
    news_series = informative_news.reindex(prices.index)

    result = ablation.run_purged_ablation(prices, news_series, horizon=21, n_splits=4, embargo_days=5)

    assert result["baseline"]["r2_oos"] < result["com_noticia"]["r2_oos"]


def test_run_purged_ablation_raises_when_dataset_too_small():
    prices = _price_series(40, seed=3)
    news = pd.Series(0.0, index=prices.index)
    with pytest.raises(ValueError):
        ablation.run_purged_ablation(prices, news, horizon=21, n_splits=10, embargo_days=5)


def test_load_and_run_purged_ablation_reads_processed_parquets(tmp_path, monkeypatch):
    prices = _price_series(400, seed=4)
    ptax_df = pd.DataFrame({"date": prices.index, "value": prices.to_numpy(), "tipo": "venda"})
    ptax_path = tmp_path / "ptax.parquet"
    ptax_df.to_parquet(ptax_path)

    tone_df = pd.DataFrame(
        {
            "date": prices.index,
            "tone": np.random.default_rng(5).normal(0, 1, len(prices)),
            "query": "Brazil",
        }
    )
    tone_path = tmp_path / "gdelt_tone.parquet"
    tone_df.to_parquet(tone_path)

    import vol.forecast as forecast_module

    monkeypatch.setattr(forecast_module, "PTAX_PROCESSED_PATH", ptax_path)
    monkeypatch.setattr(forecast_module, "TONE_PROCESSED_PATH", tone_path)

    result = ablation.load_and_run_purged_ablation(horizon=21, n_splits=3, embargo_days=5, query="Brazil")

    assert "baseline" in result and "com_noticia" in result
