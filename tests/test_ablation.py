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


def test_run_purged_ablation_uses_provided_folds_instead_of_n_splits():
    from backtest.walk_forward import purged_walk_forward_splits_by_step
    from vol.forecast import build_dataset

    prices = _price_series(700, seed=6)
    news = pd.Series(0.0, index=prices.index)

    dataset = build_dataset(prices, news=news, horizon=21, news_smooth_window=21)
    custom_folds = purged_walk_forward_splits_by_step(
        dataset, min_train_size=252, step_size=21, horizon=21, embargo_days=5
    )

    result = ablation.run_purged_ablation(prices, news, horizon=21, embargo_days=5, folds=custom_folds)

    assert result["n_splits"] == len(custom_folds)
    assert result["n_splits"] > 5  # bem mais folds que o esquema de 5 fixos


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


def test_load_and_run_fiscal_risk_ablation_reads_processed_parquets(tmp_path, monkeypatch):
    prices = _price_series(400, seed=11)
    ptax_df = pd.DataFrame({"date": prices.index, "value": prices.to_numpy(), "tipo": "venda"})
    ptax_path = tmp_path / "ptax.parquet"
    ptax_df.to_parquet(ptax_path)

    import data.gdelt_news as gdelt_news_module

    volume_df = pd.DataFrame(
        {
            "date": prices.index,
            "share_pct": np.abs(np.random.default_rng(12).normal(0.1, 0.02, len(prices))),
            "query": gdelt_news_module.FISCAL_RISK_QUERY,
        }
    )
    volume_path = tmp_path / "gdelt_volume.parquet"
    volume_df.to_parquet(volume_path)

    monkeypatch.setattr(ablation, "PTAX_PROCESSED_PATH", ptax_path)
    monkeypatch.setattr(gdelt_news_module, "VOLUME_PROCESSED_PATH", volume_path)

    result = ablation.load_and_run_fiscal_risk_ablation(horizon=21, n_splits=3, embargo_days=5)

    assert "baseline" in result and "com_noticia" in result


def test_load_and_run_fiscal_risk_ablation_use_surprise_transforms_news(tmp_path, monkeypatch):
    prices = _price_series(400, seed=13)
    ptax_df = pd.DataFrame({"date": prices.index, "value": prices.to_numpy(), "tipo": "venda"})
    ptax_path = tmp_path / "ptax.parquet"
    ptax_df.to_parquet(ptax_path)

    import data.gdelt_news as gdelt_news_module

    volume_df = pd.DataFrame(
        {
            "date": prices.index,
            "share_pct": np.abs(np.random.default_rng(14).normal(0.1, 0.02, len(prices))),
            "query": gdelt_news_module.FISCAL_RISK_QUERY,
        }
    )
    volume_path = tmp_path / "gdelt_volume.parquet"
    volume_df.to_parquet(volume_path)

    monkeypatch.setattr(ablation, "PTAX_PROCESSED_PATH", ptax_path)
    monkeypatch.setattr(gdelt_news_module, "VOLUME_PROCESSED_PATH", volume_path)

    result = ablation.load_and_run_fiscal_risk_ablation(
        horizon=21, n_splits=3, embargo_days=5, use_surprise=True, surprise_window=63
    )

    assert "baseline" in result and "com_noticia" in result


def test_load_and_run_purged_ablation_use_parkinson_reads_fx_spot(tmp_path, monkeypatch):
    prices = _price_series(400, seed=7)
    ptax_df = pd.DataFrame({"date": prices.index, "value": prices.to_numpy(), "tipo": "venda"})
    ptax_path = tmp_path / "ptax.parquet"
    ptax_df.to_parquet(ptax_path)

    tone_df = pd.DataFrame(
        {"date": prices.index, "tone": np.random.default_rng(8).normal(0, 1, len(prices)), "query": "Brazil"}
    )
    tone_path = tmp_path / "gdelt_tone.parquet"
    tone_df.to_parquet(tone_path)

    rng = np.random.default_rng(9)
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

    import vol.forecast as forecast_module
    import vol.realized as realized_module

    monkeypatch.setattr(forecast_module, "PTAX_PROCESSED_PATH", ptax_path)
    monkeypatch.setattr(forecast_module, "TONE_PROCESSED_PATH", tone_path)
    monkeypatch.setattr(realized_module, "FX_SPOT_PROCESSED_PATH", fx_path)

    result = ablation.load_and_run_purged_ablation(
        horizon=21, n_splits=3, embargo_days=5, query="Brazil", use_parkinson=True
    )

    assert "baseline" in result and "com_noticia" in result


def test_build_dataset_with_fiscal_risk_aligns_both_layers_no_lookahead():
    prices = _price_series(300, seed=20)

    surprise = pd.Series(1.0, index=prices.index)
    sentiment = pd.Series(0.5, index=prices.index)

    dataset = ablation.build_dataset_with_fiscal_risk(prices, surprise, sentiment, horizon=21)

    assert list(dataset.columns) == ["rv_d", "rv_w", "rv_m", "target", "fiscal_surprise", "fiscal_sentiment"]
    assert (dataset["fiscal_surprise"] == 1.0).all()
    assert (dataset["fiscal_sentiment"] == 0.5).all()


def test_build_dataset_with_fiscal_risk_aligns_date_only_sentiment_index():
    # sentiment.daily_index.daily_sentiment_index guarda "date" como
    # datetime.date PURO (sem tz) -- prices/fiscal_surprise sao tz-aware
    # (America/Sao_Paulo). Sem normalizar, o reindex casaria zero labels.
    prices = _price_series(300, seed=26)
    surprise = pd.Series(1.0, index=prices.index)
    sentiment = pd.Series(0.5, index=prices.index.date)  # indice date-only, sem tz

    dataset = ablation.build_dataset_with_fiscal_risk(
        prices, surprise, sentiment, horizon=21, sentiment_smooth_window=None
    )

    assert len(dataset) > 0
    assert (dataset["fiscal_sentiment"] == 0.5).all()


def test_run_fiscal_risk_ablation_v2_recovers_signal_when_informative():
    prices = _price_series(400, seed=21)
    from vol.forecast import build_dataset

    dataset_baseline = build_dataset(prices, horizon=21)
    rng = np.random.default_rng(22)
    informative = dataset_baseline["target"] + rng.normal(0, 0.01, len(dataset_baseline))
    surprise = informative.reindex(prices.index)
    sentiment = pd.Series(0.0, index=prices.index)

    result = ablation.run_fiscal_risk_ablation_v2(
        prices, surprise, sentiment, horizon=21, n_splits=4, embargo_days=5
    )

    assert result["baseline"]["r2_oos"] < result["com_risco_fiscal_v2"]["r2_oos"]


def test_load_and_run_fiscal_risk_ablation_v2_raises_when_sentiment_not_collected(tmp_path, monkeypatch):
    import sentiment.daily_index as daily_index_module

    monkeypatch.setattr(daily_index_module, "FISCAL_SENTIMENT_PROCESSED_PATH", tmp_path / "missing.parquet")

    with pytest.raises(FileNotFoundError):
        ablation.load_and_run_fiscal_risk_ablation_v2()


def test_load_and_run_fiscal_risk_ablation_v2_reads_processed_parquets(tmp_path, monkeypatch):
    prices = _price_series(400, seed=23)
    ptax_df = pd.DataFrame({"date": prices.index, "value": prices.to_numpy(), "tipo": "venda"})
    ptax_path = tmp_path / "ptax.parquet"
    ptax_df.to_parquet(ptax_path)

    import data.gdelt_news as gdelt_news_module
    import sentiment.daily_index as daily_index_module

    volume_df = pd.DataFrame(
        {
            "date": prices.index,
            "share_pct": np.abs(np.random.default_rng(24).normal(0.1, 0.02, len(prices))),
            "query": gdelt_news_module.FISCAL_RISK_QUERY,
        }
    )
    volume_path = tmp_path / "gdelt_volume.parquet"
    volume_df.to_parquet(volume_path)

    sentiment_df = pd.DataFrame(
        {"date": prices.index.date, "sentiment_mean": np.random.default_rng(25).normal(0, 0.1, len(prices))}
    )
    sentiment_path = tmp_path / "fiscal_sentiment_index.parquet"
    sentiment_df.to_parquet(sentiment_path)

    monkeypatch.setattr(ablation, "PTAX_PROCESSED_PATH", ptax_path)
    monkeypatch.setattr(gdelt_news_module, "VOLUME_PROCESSED_PATH", volume_path)
    monkeypatch.setattr(daily_index_module, "FISCAL_SENTIMENT_PROCESSED_PATH", sentiment_path)

    result = ablation.load_and_run_fiscal_risk_ablation_v2(horizon=21, n_splits=3, embargo_days=5)

    assert "baseline" in result and "com_risco_fiscal_v2" in result
