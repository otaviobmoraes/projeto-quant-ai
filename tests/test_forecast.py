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


def test_build_dataset_daily_variance_override_replaces_close_to_close():
    prices = _price_series(60, seed=9)
    default_dataset = forecast.build_dataset(prices, horizon=5)

    # Variancia diaria diferente (ex.: Parkinson) -> features/target tem
    # que refletir a serie fornecida, nao o retorno de fechamento.
    custom_variance = pd.Series(0.0001, index=prices.index)
    custom_dataset = forecast.build_dataset(prices, horizon=5, daily_variance=custom_variance)

    assert not custom_dataset["rv_d"].equals(default_dataset["rv_d"])
    assert (custom_dataset["rv_d"] == 0.0001).all()


def test_har_features_from_variance_matches_har_features():
    prices = _price_series(60, seed=11)
    variance = forecast.log_returns(prices) ** 2

    assert forecast.har_features(prices).equals(forecast.har_features_from_variance(variance))


def test_overnight_variance_manual_calc():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    open_prices = pd.Series([5.0, 5.2, 5.15], index=idx)
    close = pd.Series([5.05, 5.18, 5.20], index=idx)
    prev_close = close.shift(1)

    result = forecast.overnight_variance(open_prices, prev_close)

    assert pd.isna(result.iloc[0])  # sem close anterior no 1o dia
    expected_day2 = np.log(5.2 / 5.05) ** 2
    assert result.iloc[1] == pytest.approx(expected_day2)


def test_overnight_variance_zero_when_open_equals_prev_close():
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    open_prices = pd.Series([5.0, 5.05], index=idx)
    prev_close = pd.Series([np.nan, 5.05], index=idx)

    result = forecast.overnight_variance(open_prices, prev_close)

    assert result.iloc[1] == pytest.approx(0.0)


def test_semivariance_features_splits_by_return_sign():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    daily_variance = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    returns = pd.Series([0.01, -0.02, 0.03, -0.04], index=idx)

    result = forecast.semivariance_features(daily_variance, returns)

    assert list(result.columns) == ["rv_d_pos", "rv_d_neg"]
    assert result["rv_d_pos"].tolist() == [1.0, 0.0, 3.0, 0.0]
    assert result["rv_d_neg"].tolist() == [0.0, 2.0, 0.0, 4.0]


def test_semivariance_features_sum_equals_daily_variance():
    idx = pd.date_range("2024-01-01", periods=50, freq="B")
    rng = np.random.default_rng(5)
    daily_variance = pd.Series(np.abs(rng.normal(1, 0.3, 50)), index=idx)
    returns = pd.Series(rng.normal(0, 0.01, 50), index=idx)

    result = forecast.semivariance_features(daily_variance, returns)

    assert (result["rv_d_pos"] + result["rv_d_neg"]).equals(daily_variance)


def test_build_dataset_extended_has_expected_columns():
    prices = _price_series(100, seed=13)
    open_ = prices * 1.001  # abertura levemente diferente do fechamento

    dataset = forecast.build_dataset_extended(prices, open_, horizon=21)

    assert set(dataset.columns) == {
        "rv_d", "rv_w", "rv_m", "target", "rv_d_pos", "rv_d_neg", "overnight",
    }
    assert len(dataset) > 0
    assert (dataset["rv_d_pos"] + dataset["rv_d_neg"] - dataset["rv_d"]).abs().max() < 1e-12


def test_build_dataset_extended_overnight_has_no_lookahead():
    prices = _price_series(100, seed=14)
    open_ = prices * 1.001

    full = forecast.build_dataset_extended(prices, open_, horizon=21)
    truncated = forecast.build_dataset_extended(prices.iloc[:60], open_.iloc[:60], horizon=21)

    common_idx = full.index.intersection(truncated.index)
    assert len(common_idx) > 0
    pd.testing.assert_series_equal(
        full.loc[common_idx, "overnight"], truncated.loc[common_idx, "overnight"]
    )


def test_forward_target_from_variance_matches_forward_target():
    prices = _price_series(60, seed=12)
    variance = forecast.log_returns(prices) ** 2

    left = forecast.forward_target(prices, horizon=10)
    right = forecast.forward_target_from_variance(variance, horizon=10)
    pd.testing.assert_series_equal(left, right)


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


def test_predict_matches_evaluate_internals():
    prices = _price_series(200, seed=20)
    dataset = forecast.build_dataset(prices, horizon=21)
    train, test = forecast.chronological_split(dataset, test_size=0.2)

    model = forecast.fit_har(train, forecast.BASELINE_FEATURES, log_target=True)
    pred = forecast.predict(model, test, forecast.BASELINE_FEATURES, log_target=True)

    # evaluate() usa a mesma previsao internamente -- reconstruindo o RMSE a
    # partir de predict() ele bate com o RMSE reportado por evaluate().
    result = forecast.evaluate(model, test, forecast.BASELINE_FEATURES, log_target=True)
    manual_rmse = float(np.sqrt(((test["target"] - pred) ** 2).mean()))
    assert manual_rmse == pytest.approx(result["rmse"])
    assert pred.index.equals(test.index)


def test_persistence_forecast_matches_manual_calc():
    dataset = pd.DataFrame({"rv_m": [0.0001, 0.0004], "target": [10.0, 20.0]})
    pred = forecast.persistence_forecast(dataset)

    expected = np.sqrt(dataset["rv_m"] * forecast.TRADING_DAYS_PER_YEAR) * 100
    assert pred.tolist() == pytest.approx(expected.tolist())


def test_evaluate_persistence_perfect_when_rv_m_matches_target_exactly():
    # rv_m (em variancia diaria) escolhido pra que sqrt(rv_m*252)*100 bata
    # exatamente com o target -- persistencia "perfeita" por construcao.
    target = np.array([10.0, 15.0, 8.0])
    rv_m = (target / 100) ** 2 / forecast.TRADING_DAYS_PER_YEAR
    dataset = pd.DataFrame({"rv_m": rv_m, "target": target})

    result = forecast.evaluate_persistence(dataset)
    assert result["r2_oos"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(0.0, abs=1e-8)


def test_evaluate_persistence_worse_than_mean_gives_negative_r2():
    # rv_m sistematicamente longe do target -> pior que prever a media.
    dataset = pd.DataFrame({"rv_m": [0.01, 0.01, 0.01], "target": [5.0, 5.1, 4.9]})
    result = forecast.evaluate_persistence(dataset)
    assert result["r2_oos"] < 0


def test_evaluate_perfect_predictions_give_r2_one():
    test = pd.DataFrame({"rv_d": [1.0, 2.0], "target": [10.0, 20.0]})

    class _PerfectModel:
        def predict(self, X):
            return X["rv_d"] * 10

    result = forecast.evaluate(_PerfectModel(), test, ["rv_d"])
    assert result["r2_oos"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(0.0)


def test_fit_har_log_target_recovers_log_linear_relationship():
    rng = np.random.default_rng(7)
    rv_d = rng.uniform(0.1, 2.0, 200)
    # Alvo log-linear em rv_d (+ ruido pequeno em log-escala) -- cenario onde
    # ajustar em log(RV) deve bater bem melhor do que em nivel.
    target = np.exp(1.0 + 0.5 * rv_d + rng.normal(0, 0.01, 200))
    train = pd.DataFrame({"rv_d": rv_d, "target": target})

    model_log = forecast.fit_har(train, ["rv_d"], log_target=True)
    result_log = forecast.evaluate(model_log, train, ["rv_d"], log_target=True)

    model_level = forecast.fit_har(train, ["rv_d"], log_target=False)
    result_level = forecast.evaluate(model_level, train, ["rv_d"], log_target=False)

    assert result_log["r2_oos"] > result_level["r2_oos"]
    assert result_log["r2_oos"] > 0.9


def test_build_dataset_news_smooth_window_reduces_noise():
    prices = _price_series(80)
    rng = np.random.default_rng(5)
    noisy_news = pd.Series(rng.normal(0, 1, len(prices)), index=prices.index)

    raw = forecast.build_dataset(prices, news=noisy_news, horizon=5)
    smoothed = forecast.build_dataset(prices, news=noisy_news, horizon=5, news_smooth_window=21)

    assert smoothed["news"].std() < raw["news"].std()


def test_expanding_window_splits_are_chronological_and_non_overlapping():
    prices = _price_series(150)
    dataset = forecast.build_dataset(prices, horizon=5)

    folds = forecast.expanding_window_splits(dataset, n_splits=4)

    assert len(folds) == 4
    prev_test_end = None
    for train, test in folds:
        assert train.index.max() < test.index.min()
        if prev_test_end is not None:
            assert test.index.min() > prev_test_end
        prev_test_end = test.index.max()


def test_run_ablation_cv_recovers_signal_when_news_is_informative():
    prices = _price_series(400, seed=8)
    dataset_baseline = forecast.build_dataset(prices, horizon=21)

    rng = np.random.default_rng(9)
    informative_news = dataset_baseline["target"] + rng.normal(0, 0.01, len(dataset_baseline))
    news_series = informative_news.reindex(prices.index)

    result = forecast.run_ablation_cv(prices, news_series, horizon=21, n_splits=4)

    assert "per_fold" in result and result["n_splits"] == 4
    assert result["baseline"]["r2_oos"] < result["com_noticia"]["r2_oos"]
    assert len(result["per_fold"]["baseline"]) == 4


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


def test_load_and_run_ablation_cv_reads_processed_parquets(tmp_path, monkeypatch):
    prices = _price_series(400, seed=6)
    ptax_df = pd.DataFrame({"date": prices.index, "value": prices.to_numpy(), "tipo": "venda"})
    ptax_path = tmp_path / "ptax.parquet"
    ptax_df.to_parquet(ptax_path)

    tone_df = pd.DataFrame(
        {
            "date": prices.index,
            "tone": np.random.default_rng(10).normal(0, 1, len(prices)),
            "query": "Brazil",
        }
    )
    tone_path = tmp_path / "gdelt_tone.parquet"
    tone_df.to_parquet(tone_path)

    monkeypatch.setattr(forecast, "PTAX_PROCESSED_PATH", ptax_path)
    monkeypatch.setattr(forecast, "TONE_PROCESSED_PATH", tone_path)

    result = forecast.load_and_run_ablation_cv(horizon=21, n_splits=3, query="Brazil")

    assert result["n_splits"] == 3
    assert "baseline" in result and "com_noticia" in result


def test_load_and_run_ablation_cv_use_parkinson_reads_fx_spot(tmp_path, monkeypatch):
    prices = _price_series(400, seed=13)
    ptax_df = pd.DataFrame({"date": prices.index, "value": prices.to_numpy(), "tipo": "venda"})
    ptax_path = tmp_path / "ptax.parquet"
    ptax_df.to_parquet(ptax_path)

    tone_df = pd.DataFrame(
        {"date": prices.index, "tone": np.random.default_rng(14).normal(0, 1, len(prices)), "query": "Brazil"}
    )
    tone_path = tmp_path / "gdelt_tone.parquet"
    tone_df.to_parquet(tone_path)

    rng = np.random.default_rng(15)
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

    monkeypatch.setattr(forecast, "PTAX_PROCESSED_PATH", ptax_path)
    monkeypatch.setattr(forecast, "TONE_PROCESSED_PATH", tone_path)
    import vol.realized as realized_module

    monkeypatch.setattr(realized_module, "FX_SPOT_PROCESSED_PATH", fx_path)

    result = forecast.load_and_run_ablation_cv(horizon=21, n_splits=3, query="Brazil", use_parkinson=True)

    assert "baseline" in result and "com_noticia" in result
