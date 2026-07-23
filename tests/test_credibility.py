from datetime import date

import numpy as np
import pandas as pd
import pytest

from credibility import credibility


def test_theta_baseline_is_one_when_gap_is_zero():
    gap = pd.Series([0.0, 0.0, 0.0])
    theta = credibility.theta_baseline(gap)
    assert (theta == 1.0).all()


def test_theta_baseline_bounded_between_zero_and_one():
    rng = np.random.default_rng(0)
    gap = pd.Series(rng.normal(0, 1.5, 200))
    theta = credibility.theta_baseline(gap)

    assert (theta > 0).all()
    assert (theta <= 1).all()


def test_theta_baseline_decreases_as_gap_magnitude_grows():
    theta_small = credibility.theta_baseline(pd.Series([0.1]))
    theta_large = credibility.theta_baseline(pd.Series([5.0]))

    assert theta_large.iloc[0] < theta_small.iloc[0]


def test_theta_baseline_scale_controls_decay_rate():
    gap = pd.Series([2.0])
    theta_tight = credibility.theta_baseline(gap, scale=1.0)
    theta_loose = credibility.theta_baseline(gap, scale=4.0)

    # Escala maior = mais tolerante (theta cai mais devagar com o mesmo gap).
    assert theta_loose.iloc[0] > theta_tight.iloc[0]


def test_theta_baseline_no_lookahead_pure_pointwise_transform():
    # theta_t so pode depender do gap NAQUELE t -- diferente da versao
    # anterior (desvio expansivo), aqui e uma transformacao ponto a ponto,
    # entao adicionar dados no futuro nao pode mudar valores passados.
    gap_short = pd.Series([0.5, 1.0, 1.5])
    gap_long = pd.Series([0.5, 1.0, 1.5, 3.0, 0.2])

    theta_short = credibility.theta_baseline(gap_short)
    theta_long = credibility.theta_baseline(gap_long)

    assert theta_short.tolist() == pytest.approx(theta_long.iloc[:3].tolist())


def test_load_credibility_processed_reads_focus_and_computes_theta(monkeypatch):
    idx = pd.date_range("2023-01-01", periods=60, tz="America/Sao_Paulo")
    rng = np.random.default_rng(3)
    fake_focus = pd.DataFrame(
        {
            "date": idx,
            "pi_expectation": 4.0 + rng.normal(0, 0.5, 60),
            "pi_median": 4.0,
            "dispersion": rng.uniform(0.2, 0.6, 60),
            "n_respondentes": 100,
            "meta": 3.0,
            "gap": rng.normal(1.0, 0.5, 60),
        }
    )

    def fake_load_focus_processed(start, end):
        return fake_focus

    monkeypatch.setattr(credibility, "load_focus_processed", fake_load_focus_processed)
    monkeypatch.setattr(credibility, "PROCESSED_DIR", credibility.PROCESSED_DIR)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        monkeypatch.setattr(credibility, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(credibility, "PROCESSED_PATH", tmp_path / "credibility.parquet")

        df = credibility.load_credibility_processed(date(2023, 1, 1), date(2023, 3, 1))

        assert list(df.columns) == ["date", "gap", "dispersion", "theta_baseline"]
        assert credibility.PROCESSED_PATH.exists()
        reloaded = pd.read_parquet(credibility.PROCESSED_PATH)
        assert len(reloaded) == len(df)
