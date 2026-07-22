import pandas as pd
import pytest

from sentiment import finbert

_LABELS = {
    "bom": {"label": "POSITIVE", "score": 0.9},
    "ruim": {"label": "NEGATIVE", "score": 0.8},
    "neutro": {"label": "NEUTRAL", "score": 0.6},
}


def fake_pipeline(texts: list[str]) -> list[dict]:
    return [_LABELS[t] for t in texts]


def test_score_texts_computes_signed_score():
    texts = ["bom", "ruim", "neutro"]
    df = finbert.score_texts(texts, pipeline_fn=fake_pipeline)

    assert list(df["text"]) == texts
    assert list(df["label"]) == ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    assert df["signed_score"].iloc[0] == pytest.approx(0.9)
    assert df["signed_score"].iloc[1] == pytest.approx(-0.8)
    assert df["signed_score"].iloc[2] == pytest.approx(0.0)
