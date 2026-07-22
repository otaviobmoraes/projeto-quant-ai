import numpy as np
import pandas as pd
import pytest

from vol import forecast


def _price_series(n: int, seed: int = 0, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="America/Sao_Paulo")
    prices = 5.0 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    return pd.Series(prices, index=idx)


def test_har_features_manual_calc():
    prices = pd.Series([5.0, 5.1, 5.0, 5.2, 5.1, 5.3, 5.2])
    df = forecast.har_features(prices)

    r2 = np.log(prices).diff() ** 2
    assert df["rv_d"].iloc[3] == pytest.approx(r2.iloc[3])
    # rv_w em t=5 usa os retornos de t=1..5 (r2[0] e NaN, entao a janela
    # completa e valida so a partir daqui).
    assert df["rv_w"].iloc[5] == pytest.approx(r2.iloc[1:6].mean())


def test_forward_target_has_no_lookahead():
    prices = _price_series(60)
    target = forecast.forward_target(prices, horizon=5)

    # target em t so deve depender dos retornos estritamente APOS t.
    r2 = np.log(prices).diff() ** 2
    manual = np.sqrt(r2.iloc[11:16].mean() * forecast.TRADING_DAYS_PER_YEAR) * 100
    assert target.iloc[10] == pytest.approx(manual)
    # As ultimas `horizon` linhas nao tem alvo (olham alem do fim da serie).
    assert target.iloc[-1:].isna().all()


def test_build_dataset_aligns_news_with_ffill_and_drops_na():
    prices = _price_series(50)
    news = pd.Series(1.0, index=prices.index[::2])  # noticia so em dias alternados

    dataset = forecast.build_dataset(prices, news=news, horizon=5)

    assert "news" in dataset.columns
    assert dataset["news"].notna().all()
    assert not dataset.isna().any().any()


def test_build_dataset_without_news_has_no_news_column():
    prices = _price_series(40)
    dataset = forecast.build_dataset(prices, horizon=5)
    assert "news" not in dataset.columns


def test_chronological_split_preserves_time_order():
    prices = _price_series(80)
    dataset = forecast.build_dataset(prices, horizon=5)

    train, test = forecast.chronological_split(dataset, test_size=0.25)

    assert len(train) + len(test) == len(dataset)
    assert train.index.max() < test.index.min()


def test_run_ablation_recovers_signal_when_news_is_informative():
    prices = _price_series(300, seed=1)
    dataset_baseline = forecast.build_dataset(prices, horizon=21)

    rng = np.random.default_rng(2)
    # Noticia perfeitamente correlacionada ao alvo (+ ruido pequeno) --
    # cenario onde ela DEVE ajudar bastante o modelo fora da amostra.
    informative_news = dataset_baseline["target"] + rng.normal(0, 0.01, len(dataset_baseline))
    news_series = informative_news.reindex(prices.index)

    result = forecast.run_ablation(prices, news_series, horizon=21, test_size=0.2)

    assert result["baseline"]["r2_oos"] < result["com_noticia"]["r2_oos"]
    assert result["com_noticia"]["rmse"] < result["baseline"]["rmse"]
    assert result["n_train"] > 0 and result["n_test"] > 0


def test_evaluate_perfect_predictions_give_r2_one():
    test = pd.DataFrame({"rv_d": [1.0, 2.0], "target": [10.0, 20.0]})

    class _PerfectModel:
        def predict(self, X):
            return X["rv_d"] * 10

    result = forecast.evaluate(_PerfectModel(), test, ["rv_d"])
    assert result["r2_oos"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(0.0)


def test_load_and_run_ablation_missing_files_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(forecast, "PTAX_PROCESSED_PATH", tmp_path / "missing_ptax.parquet")
    monkeypatch.setattr(forecast, "TONE_PROCESSED_PATH", tmp_path / "missing_tone.parquet")

    with pytest.raises(FileNotFoundError):
        forecast.load_and_run_ablation()


def test_load_and_run_ablation_reads_processed_parquets(tmp_path, monkeypatch):
    prices = _price_series(300, seed=3)
    ptax_df = pd.DataFrame(
        {"date": prices.index, "value": prices.to_numpy(), "tipo": "venda"}
    )
    ptax_path = tmp_path / "ptax.parquet"
    ptax_df.to_parquet(ptax_path)

    tone_df = pd.DataFrame(
        {
            "date": prices.index,
            "tone": np.random.default_rng(4).normal(0, 1, len(prices)),
            "query": "Brazil",
        }
    )
    tone_path = tmp_path / "gdelt_tone.parquet"
    tone_df.to_parquet(tone_path)

    monkeypatch.setattr(forecast, "PTAX_PROCESSED_PATH", ptax_path)
    monkeypatch.setattr(forecast, "TONE_PROCESSED_PATH", tone_path)

    result = forecast.load_and_run_ablation(horizon=21, query="Brazil")

    assert "baseline" in result and "com_noticia" in result
    assert result["n_train"] > 0 and result["n_test"] > 0
