from report import summary


def test_format_metrics_contains_all_fields():
    text = summary.format_metrics({"rmse": 1.234, "mae": 0.987, "r2_oos": -0.5})
    assert "RMSE=1.234" in text
    assert "MAE=0.987" in text
    assert "R2_oos=-0.500" in text


def test_ablation_summary_md_lists_configs_and_metrics():
    result = {
        "baseline": {"rmse": 2.0, "mae": 1.5, "r2_oos": -0.1},
        "com_noticia": {"rmse": 1.9, "mae": 1.4, "r2_oos": -0.05},
        "n_splits": 5,
    }
    configs = ["log_target=True", "news_smooth_window=21", "horizon=21"]

    text = summary.ablation_summary_md(result, configs)

    assert "Baseline" in text
    assert "Com noticia" in text
    assert "Folds (walk-forward purgado): 5" in text
    assert f"Configuracoes testadas ({len(configs)})" in text
    for c in configs:
        assert c in text


def test_ablation_summary_md_includes_dsr_when_provided():
    result = {"baseline": {"rmse": 2.0, "mae": 1.5, "r2_oos": -0.1}, "com_noticia": {"rmse": 1.9, "mae": 1.4, "r2_oos": -0.05}, "n_splits": 5}
    text = summary.ablation_summary_md(result, ["cfg1"], dsr=0.42)
    assert "Deflated Sharpe Ratio" in text
    assert "0.420" in text


def test_ablation_summary_md_omits_dsr_when_not_provided():
    result = {"baseline": {"rmse": 2.0, "mae": 1.5, "r2_oos": -0.1}, "com_noticia": {"rmse": 1.9, "mae": 1.4, "r2_oos": -0.05}, "n_splits": 5}
    text = summary.ablation_summary_md(result, ["cfg1"])
    assert "Deflated Sharpe Ratio" not in text
