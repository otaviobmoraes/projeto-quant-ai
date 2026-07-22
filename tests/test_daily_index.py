from datetime import date

import pandas as pd
import pytest

from sentiment import daily_index

_LABELS = {
    "bom": {"label": "POSITIVE", "score": 0.9},
    "ruim": {"label": "NEGATIVE", "score": 0.8},
    "neutro": {"label": "NEUTRAL", "score": 0.6},
}


def fake_pipeline(texts: list[str]) -> list[dict]:
    return [_LABELS[t] for t in texts]


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_index, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(daily_index, "PROCESSED_PATH", tmp_path / "processed" / "sentiment_index.parquet")
    yield


def test_daily_sentiment_index_aggregates_by_day():
    df = pd.DataFrame(
        {
            "date": [date(2026, 7, 1), date(2026, 7, 1), date(2026, 7, 2)],
            "text": ["bom", "ruim", "neutro"],
        }
    )

    result = daily_index.daily_sentiment_index(
        df, date_col="date", text_col="text", pipeline_fn=fake_pipeline
    )

    assert list(result.columns) == ["date", "sentiment_mean", "share_negative", "n_textos"]
    day1 = result[result["date"] == date(2026, 7, 1)].iloc[0]
    assert day1["sentiment_mean"] == pytest.approx((0.9 - 0.8) / 2)
    assert day1["share_negative"] == pytest.approx(0.5)
    assert day1["n_textos"] == 2

    day2 = result[result["date"] == date(2026, 7, 2)].iloc[0]
    assert day2["sentiment_mean"] == pytest.approx(0.0)
    assert day2["n_textos"] == 1


def test_daily_sentiment_index_empty_input():
    df = pd.DataFrame(columns=["date", "text"])
    result = daily_index.daily_sentiment_index(df, date_col="date", text_col="text")
    assert result.empty


def test_load_combined_daily_sentiment_raises_when_no_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_index, "COPOM_PROCESSED_PATH", tmp_path / "missing_copom.parquet")
    monkeypatch.setattr(daily_index, "RSS_PROCESSED_PATH", tmp_path / "missing_rss.parquet")

    with pytest.raises(FileNotFoundError):
        daily_index.load_combined_daily_sentiment()


def test_load_combined_daily_sentiment_combines_copom_and_rss(tmp_path, monkeypatch):
    copom_path = tmp_path / "copom.parquet"
    rss_path = tmp_path / "rss.parquet"

    copom_df = pd.DataFrame(
        {
            "data_referencia": [pd.Timestamp("2026-07-01", tz="America/Sao_Paulo")],
            "texto": ["bom"],
        }
    )
    copom_df.to_parquet(copom_path)

    rss_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-07-01", tz="America/Sao_Paulo")],
            "title": ["ruim"],
        }
    )
    rss_df.to_parquet(rss_path)

    monkeypatch.setattr(daily_index, "COPOM_PROCESSED_PATH", copom_path)
    monkeypatch.setattr(daily_index, "RSS_PROCESSED_PATH", rss_path)

    result = daily_index.load_combined_daily_sentiment(pipeline_fn=fake_pipeline)

    assert daily_index.PROCESSED_PATH.exists()
    assert len(result) == 1
    assert result["n_textos"].iloc[0] == 2
    assert result["sentiment_mean"].iloc[0] == pytest.approx((0.9 - 0.8) / 2)
