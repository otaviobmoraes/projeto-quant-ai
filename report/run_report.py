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
from credibility import ablation as credibility_ablation
from data.gdelt_news import TONE_PROCESSED_PATH
from data.iv_surface import PROCESSED_PATH as IV_PROCESSED_PATH
from data.ptax import PROCESSED_PATH as PTAX_PROCESSED_PATH
from report import plots, summary
from strategy import signal, sizing
from vol import implied, realized

OUT_DIR = Path(__file__).resolve().parent / "output"

CONFIGS_TESTED = [
    "Fase 4 (preliminar): split unico 80/20, noticia bruta, RV em nivel (proxy retorno^2)",
    "Fase 4 (refinado): CV expansiva 5 folds, noticia suavizada 21d, log-RV (proxy retorno^2)",
    "Fase 6 (final v1): walk-forward PURGADO 5 folds + embargo 5d, noticia suavizada 21d, log-RV (proxy retorno^2)",
    "Diagnostico: persistencia pura (sem modelo) -- tambem R2 negativo, confirma que o problema nao e a regressao",
    "Diagnostico: GARCH(1,1) em retorno^2 -- pior que HAR-RV simples",
    "Diagnostico: HAR-RV com estimador Parkinson (OHLC) -- RMSE menor em TODOS os folds, adotado como baseline oficial",
    "Fase 6 (final v2, oficial): walk-forward PURGADO 5 folds + embargo 5d, noticia suavizada 21d, log-RV, variancia Parkinson",
    "Credibilidade Tier 1: theta_baseline + dispersao (Focus/meta), log-RV, variancia Parkinson",
]


def _fold_comparison(
    per_fold_baseline: list[dict], per_fold_other: list[dict], metric: str = "r2_oos"
) -> tuple[int, int]:
    """Conta em quantos folds a versao aumentada melhorou vs piorou o `metric`."""
    melhorou = sum(1 for b, o in zip(per_fold_baseline, per_fold_other) if o[metric] > b[metric])
    piorou = sum(1 for b, o in zip(per_fold_baseline, per_fold_other) if o[metric] < b[metric])
    return melhorou, piorou


def build_verdict(result: dict, comparison_key: str = "com_noticia", comparison_label: str = "notícia (GDELT)") -> str:
    """Veredito em linguagem simples: o modelo previu bem? A camada extra
    (noticia ou credibilidade) ajudou? Pura funcao de dados -- nao faz I/O,
    so interpreta o resultado da ablacao.
    """
    baseline_r2 = result["baseline"]["r2_oos"]
    other_r2 = result[comparison_key]["r2_oos"]
    delta = other_r2 - baseline_r2

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

    melhorou, piorou = _fold_comparison(result["per_fold"]["baseline"], result["per_fold"][comparison_key])
    n_folds = len(result["per_fold"]["baseline"])
    if delta > 0.02 and melhorou > piorou:
        other_line = (
            f"Ajuda de forma perceptivel na media (R2 {delta:+.3f}) e melhora "
            f"em {melhorou} de {n_folds} folds -- efeito consistente."
        )
    elif delta > 0:
        other_line = (
            f"Ajuda pouco na media (R2 {delta:+.3f}), mas de forma INCONSISTENTE: "
            f"melhora em {melhorou} de {n_folds} folds e piora em {piorou} -- nao da pra "
            "confiar que o efeito e real, pode ser ruido de um fold especifico."
        )
    else:
        other_line = (
            f"NAO ajuda na media (R2 {delta:+.3f}) -- melhora em {melhorou} de {n_folds} "
            f"folds, piora em {piorou}."
        )

    return (
        f"O modelo consegue prever bem a RV futura? {forecast_line}\n\n"
        f"A camada de {comparison_label} melhora a previsao? {other_line}"
    )


def _print_header(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _print_header("1. Avaliacao final do modelo (walk-forward purgado, baseline Parkinson)")
    result = ablation.load_and_run_purged_ablation(
        horizon=21, n_splits=5, embargo_days=5, use_parkinson=True
    )
    print(f"Baseline (HAR-RV, Parkinson):  {summary.format_metrics(result['baseline'])}")
    print(f"Com noticia:                   {summary.format_metrics(result['com_noticia'])}")

    md = summary.ablation_summary_md(result, CONFIGS_TESTED)
    (OUT_DIR / "ablation_summary.md").write_text(md, encoding="utf-8")

    fig = plots.plot_ablation_folds(result["per_fold"]["baseline"], result["per_fold"]["com_noticia"])
    fig.savefig(OUT_DIR / "ablation_folds.png", dpi=150)

    _print_header("2. Veredito -- noticia (GDELT)")
    print(build_verdict(result, comparison_key="com_noticia", comparison_label="notícia (GDELT)"))

    _print_header("2b. Veredito -- credibilidade (Focus/meta, Tier 1)")
    try:
        cred_result = credibility_ablation.load_and_run_credibility_ablation(
            horizon=21, n_splits=5, embargo_days=5, use_parkinson=True
        )
        print(f"Baseline (HAR-RV, Parkinson):  {summary.format_metrics(cred_result['baseline'])}")
        print(f"Com credibilidade:             {summary.format_metrics(cred_result['com_credibilidade'])}")
        print()
        print(
            build_verdict(
                cred_result, comparison_key="com_credibilidade", comparison_label="credibilidade (theta_t + dispersao)"
            )
        )

        fig = plots.plot_ablation_folds(
            cred_result["per_fold"]["baseline"], cred_result["per_fold"]["com_credibilidade"]
        )
        fig.savefig(OUT_DIR / "ablation_folds_credibility.png", dpi=150)
    except FileNotFoundError as e:
        print(f"Credibilidade ainda nao coletada -- {e}")

    _print_header("3. Graficos historicos")
    fx_df = pd.read_parquet(realized.FX_SPOT_PROCESSED_PATH).set_index("date").sort_index()
    rv_series = realized.parkinson_vol(fx_df["high"], fx_df["low"], window=21).dropna()
    fig = plots.plot_series(
        rv_series,
        title="RV realizada USD/BRL (Parkinson, janela 21d, anualizada)",
        ylabel="RV (% a.a.)",
        color=plots.SERIES_GREEN,
    )
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
