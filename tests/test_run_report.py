from report.run_report import _fold_comparison, build_verdict


def _result(baseline_r2, news_r2, per_fold_baseline, per_fold_news):
    return {
        "baseline": {"rmse": 1.0, "mae": 1.0, "r2_oos": baseline_r2},
        "com_noticia": {"rmse": 1.0, "mae": 1.0, "r2_oos": news_r2},
        "per_fold": {"baseline": per_fold_baseline, "com_noticia": per_fold_news},
    }


def test_fold_comparison_counts_improvements_and_regressions():
    baseline = [{"r2_oos": -0.1}, {"r2_oos": -0.2}, {"r2_oos": -0.3}]
    news = [{"r2_oos": -0.05}, {"r2_oos": -0.25}, {"r2_oos": -0.3}]

    melhorou, piorou = _fold_comparison(baseline, news)
    assert melhorou == 1
    assert piorou == 1


def test_build_verdict_flags_negative_r2_as_no_predictive_power():
    result = _result(-0.85, -0.75, [{"r2_oos": -0.85}], [{"r2_oos": -0.75}])
    verdict = build_verdict(result)
    assert "NAO" in verdict
    assert "-0.850" in verdict


def test_build_verdict_flags_inconsistent_news_effect():
    per_fold_baseline = [{"r2_oos": -0.1}, {"r2_oos": -0.8}, {"r2_oos": -0.2}, {"r2_oos": -2.2}, {"r2_oos": -0.2}]
    per_fold_news = [{"r2_oos": -0.2}, {"r2_oos": -0.8}, {"r2_oos": -0.3}, {"r2_oos": -1.8}, {"r2_oos": -0.15}]
    result = _result(-0.7, -0.65, per_fold_baseline, per_fold_news)

    verdict = build_verdict(result)
    assert "INCONSISTENTE" in verdict


def test_build_verdict_flags_consistent_news_improvement():
    per_fold_baseline = [{"r2_oos": -0.5}] * 5
    per_fold_news = [{"r2_oos": -0.3}] * 5
    result = _result(-0.5, -0.3, per_fold_baseline, per_fold_news)

    verdict = build_verdict(result)
    assert "consistente" in verdict.lower()


def test_build_verdict_flags_reasonable_forecast_when_r2_positive():
    result = _result(0.25, 0.26, [{"r2_oos": 0.25}], [{"r2_oos": 0.26}])
    verdict = build_verdict(result)
    assert "RAZOAVEL" in verdict
