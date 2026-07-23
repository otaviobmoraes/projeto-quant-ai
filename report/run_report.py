"""Script de conveniencia: roda a avaliacao final (walk-forward purgado),
gera todos os graficos/resumo do relatorio de uma vez e imprime um veredito
em linguagem simples sobre o modelo.

Uso:
    .venv\\Scripts\\python.exe -m report.run_report   (Windows)
    .venv/bin/python -m report.run_report             (bash)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest import ablation
from data.gdelt_news import TONE_PROCESSED_PATH
from data.iv_surface import PROCESSED_PATH as IV_PROCESSED_PATH
from data.ptax import PROCESSED_PATH as PTAX_PROCESSED_PATH
from report import plots, summary
from strategy import signal, sizing
from vol import implied, realized

OUT_DIR = Path(__file__).resolve().parent / "output"

CONFIGS_TESTED = [
    "Fase 4 (preliminar): split unico 80/20, noticia bruta, RV em nivel",
    "Fase 4 (refinado): CV expansiva 5 folds, noticia suavizada 21d, log-RV",
    "Fase 6 (final): walk-forward PURGADO 5 folds + embargo 5d, noticia suavizada 21d, log-RV",
]


def _fold_comparison(per_fold_baseline: list[dict], per_fold_news: list[dict], metric: str = "r2_oos") -> tuple[int, int]:
    """Conta em quantos folds a noticia melhorou vs piorou o `metric`."""
    melhorou = sum(1 for b, n in zip(per_fold_baseline, per_fold_news) if n[metric] > b[metric])
    piorou = sum(1 for b, n in zip(per_fold_baseline, per_fold_news) if n[metric] < b[metric])
    return melhorou, piorou


def build_verdict(result: dict) -> str:
    """Veredito em linguagem simples: o modelo previu bem? A noticia ajudou?
    Pura funcao de dados -- nao faz I/O, so interpreta o resultado da ablacao.
    """
    baseline_r2 = result["baseline"]["r2_oos"]
    news_r2 = result["com_noticia"]["r2_oos"]
    delta = news_r2 - baseline_r2

    if baseline_r2 < 0:
        forecast_line = (
            f"NAO -- o HAR-RV baseline tem R2 fora da amostra negativo ({baseline_r2:.3f}). "
            "Isso quer dizer que a previsao erra MAIS do que simplesmente usar a media "
            "historica de RV como \"previsao\" -- o modelo, do jeito que esta hoje, nao "
            "tem poder preditivo real pra RV de 21 dias nesse periodo."
        )
    elif baseline_r2 < 0.10:
        forecast_line = f"FRACO -- R2 fora da amostra positivo mas baixo ({baseline_r2:.3f})."
    else:
        forecast_line = f"RAZOAVEL -- R2 fora da amostra de {baseline_r2:.3f}."

    melhorou, piorou = _fold_comparison(result["per_fold"]["baseline"], result["per_fold"]["com_noticia"])
    n_folds = len(result["per_fold"]["baseline"])
    if delta > 0.02 and melhorou > piorou:
        news_line = (
            f"A noticia ajuda de forma perceptivel na media (R2 {delta:+.3f}) e melhora "
            f"em {melhorou} de {n_folds} folds -- efeito consistente."
        )
    elif delta > 0:
        news_line = (
            f"A noticia ajuda pouco na media (R2 {delta:+.3f}), mas de forma INCONSISTENTE: "
            f"melhora em {melhorou} de {n_folds} folds e piora em {piorou} -- nao da pra "
            "confiar que o efeito e real, pode ser ruido de um fold especifico."
        )
    else:
        news_line = f"A noticia NAO ajuda na media (R2 {delta:+.3f})."

    return (
        f"O modelo consegue prever bem a RV futura? {forecast_line}\n\n"
        f"A noticia (GDELT) melhora a previsao? {news_line}"
    )


def _print_header(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _print_header("1. Avaliacao final do modelo (walk-forward purgado)")
    result = ablation.load_and_run_purged_ablation(horizon=21, n_splits=5, embargo_days=5)
    print(f"Baseline (HAR-RV):  {summary.format_metrics(result['baseline'])}")
    print(f"Com noticia:        {summary.format_metrics(result['com_noticia'])}")

    md = summary.ablation_summary_md(result, CONFIGS_TESTED)
    (OUT_DIR / "ablation_summary.md").write_text(md, encoding="utf-8")

    fig = plots.plot_ablation_folds(result["per_fold"]["baseline"], result["per_fold"]["com_noticia"])
    fig.savefig(OUT_DIR / "ablation_folds.png", dpi=150)

    _print_header("2. Veredito")
    print(build_verdict(result))

    _print_header("3. Graficos historicos")
    rv_series = realized.load_ptax_realized_vol(window=21)["rv_pct"].dropna()
    fig = plots.plot_series(rv_series, title="RV realizada USD/BRL (janela 21d, anualizada)", ylabel="RV (% a.a.)", color=plots.SERIES_GREEN)
    fig.savefig(OUT_DIR / "rv_history.png", dpi=150)
    print(f"rv_history.png ({len(rv_series)} pontos)")

    if TONE_PROCESSED_PATH.exists():
        tone_series = pd.read_parquet(TONE_PROCESSED_PATH).set_index("date")["tone"].sort_index()
        fig = plots.plot_series(tone_series, title="Tom medio diario de noticias (GDELT)", ylabel="Tom", color=plots.SERIES_BLUE)
        fig.savefig(OUT_DIR / "gdelt_tone.png", dpi=150)
        print(f"gdelt_tone.png ({len(tone_series)} pontos)")

    _print_header("4. Sinal de hoje (estrategia)")
    if IV_PROCESSED_PATH.exists():
        iv_df = implied.load_iv_atm_processed(target_days=21)
        iv_today = iv_df.iloc[-1]
        rv_today = rv_series.iloc[-1]
        spot = pd.read_parquet(PTAX_PROCESSED_PATH)
        spot_today = spot[spot["tipo"] == "venda"].sort_values("date").iloc[-1]["value"]

        sig = signal.generate_signal(rv_today, iv_today["iv_atm_pct"], band_pct=1.0)
        label = {signal.LONG_VOL: "COMPRAR VOL", signal.SHORT_VOL: "VENDER VOL", signal.NO_TRADE: "NAO OPERAR"}[sig]
        n = sizing.size_straddle(target_vega=1000, spot=spot_today, ttm_days=21, iv_pct=iv_today["iv_atm_pct"])

        print(f"RV realizada (21d):  {rv_today:.2f}%")
        print(f"IV ATM (21d):        {iv_today['iv_atm_pct']:.2f}%")
        print(f"Sinal:               {label}")
        print(f"Straddles p/ vega alvo R$1000/ponto: {n:,.0f}")
    else:
        print("Superficie de IV ainda nao coletada -- rode data.iv_surface primeiro.")

    print()
    print(f"Arquivos gerados em: {OUT_DIR}")


if __name__ == "__main__":
    main()
