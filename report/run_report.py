"""Script de conveniencia: roda a avaliacao final (walk-forward purgado),
gera todos os graficos/resumo do relatorio de uma vez e imprime um veredito
em linguagem simples sobre o modelo.

Uso:
    .venv\\Scripts\\python.exe -m report.run_report   (Windows)
    .venv/bin/python -m report.run_report             (bash)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest import ablation, engine
from backtest.walk_forward import purged_walk_forward_splits_by_step
from credibility import ablation as credibility_ablation
from data.gdelt_news import (
    TONE_PROCESSED_PATH,
    VOLUME_PROCESSED_PATH,
    fiscal_risk_surprise,
    load_fiscal_risk_series,
)
from data.iv_surface import PROCESSED_PATH as IV_PROCESSED_PATH
from data.ptax import PROCESSED_PATH as PTAX_PROCESSED_PATH
from report import plots, summary
from sentiment.daily_index import FISCAL_SENTIMENT_PROCESSED_PATH
from strategy import signal, sizing
from vol import implied, realized
from vol.forecast import BASELINE_FEATURES, NEWS_FEATURES, build_dataset, fit_har

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

    _print_header("2c. Veredito -- risco fiscal (GDELT, atencao da imprensa)")
    if VOLUME_PROCESSED_PATH.exists():
        try:
            fiscal_result = ablation.load_and_run_fiscal_risk_ablation(
                horizon=21, n_splits=5, embargo_days=5, use_parkinson=True
            )
            print(f"Baseline (HAR-RV, Parkinson):  {summary.format_metrics(fiscal_result['baseline'])}")
            print(f"Com risco fiscal:              {summary.format_metrics(fiscal_result['com_noticia'])}")
            print()
            print(
                build_verdict(
                    fiscal_result, comparison_key="com_noticia", comparison_label="risco fiscal (GDELT, % de cobertura)"
                )
            )

            fig = plots.plot_ablation_folds(
                fiscal_result["per_fold"]["baseline"], fiscal_result["per_fold"]["com_noticia"]
            )
            fig.savefig(OUT_DIR / "ablation_folds_fiscal_risk.png", dpi=150)
        except FileNotFoundError as e:
            print(f"Risco fiscal ainda nao coletado -- {e}")
    else:
        print(
            "Risco fiscal (GDELT) ainda nao coletado -- rode "
            "data.gdelt_news.load_gdelt_volume_processed primeiro."
        )

    _print_header("2d. Veredito -- risco fiscal REFINADO (surpresa + sentimento FinBERT-PT-BR)")
    if FISCAL_SENTIMENT_PROCESSED_PATH.exists():
        try:
            fiscal_v2_result = ablation.load_and_run_fiscal_risk_ablation_v2(
                horizon=21, n_splits=5, embargo_days=5, use_parkinson=True
            )
            print(f"Baseline (HAR-RV, Parkinson):     {summary.format_metrics(fiscal_v2_result['baseline'])}")
            print(f"Com risco fiscal refinado:        {summary.format_metrics(fiscal_v2_result['com_risco_fiscal_v2'])}")
            print()
            print(
                build_verdict(
                    fiscal_v2_result,
                    comparison_key="com_risco_fiscal_v2",
                    comparison_label="risco fiscal refinado (surpresa + sentimento FinBERT nas manchetes em portugues)",
                )
            )

            fig = plots.plot_ablation_folds(
                fiscal_v2_result["per_fold"]["baseline"], fiscal_v2_result["per_fold"]["com_risco_fiscal_v2"]
            )
            fig.savefig(OUT_DIR / "ablation_folds_fiscal_risk_v2.png", dpi=150)
        except FileNotFoundError as e:
            print(f"Risco fiscal refinado ainda nao coletado -- {e}")
    else:
        print(
            "Sentimento fiscal (FinBERT) ainda nao coletado -- rode "
            "sentiment.daily_index.load_fiscal_risk_sentiment_index() primeiro "
            "(precisa de data.gdelt_news.load_gdelt_headlines_processed_chunked antes)."
        )

    _print_header("3. Graficos historicos")
    fx_df = pd.read_parquet(realized.FX_SPOT_PROCESSED_PATH).set_index("date").sort_index()
    close = fx_df["close"]
    daily_variance = realized.parkinson_daily_variance(fx_df["high"], fx_df["low"])
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

    _print_header("5. Backtest de P&L (ILUSTRATIVO -- ver aviso abaixo)")
    if IV_PROCESSED_PATH.exists():
        print(
            "AVISO: a B3 so publica o snapshot do dia da superficie de IV (sem historico\n"
            "pra download) -- so ha 1 dia real de IV conhecido. A IV de ENTRADA de cada\n"
            "trade abaixo e uma PROXY (RV Parkinson trailing x premio de risco fixo,\n"
            "calibrado nesse unico dia real). O preco de entrada/saida e o payoff\n"
            "terminal usam dado 100% real. Resultado ILUSTRATIVO -- testa o motor e da\n"
            "uma nocao de ordem de grandeza, NAO e evidencia de lucro real.\n"
        )

        dataset = build_dataset(close, horizon=21, daily_variance=daily_variance)
        # Reestima os coeficientes a cada ~21 dias (passo mensal) em vez de 5
        # folds fixos (~130 dias cada) -- o sinal de alocacao reage mais
        # rapido a mudanca de regime (ex.: salto de risco fiscal).
        step_folds = purged_walk_forward_splits_by_step(
            dataset, min_train_size=252, step_size=21, horizon=21, embargo_days=5
        )
        rv_forecast = engine.generate_oos_rv_forecast(
            dataset, BASELINE_FEATURES, horizon=21, n_splits=5, embargo_days=5, folds=step_folds
        )

        iv_df = implied.load_iv_atm_processed(target_days=21)
        calib_date = iv_df["refdate"].iloc[-1]
        iv_real = iv_df["iv_atm_pct"].iloc[-1]
        rv_calib_idx = rv_series.index.asof(calib_date)
        risk_premium = engine.calibrate_risk_premium(rv_series.loc[rv_calib_idx], iv_real)
        iv_proxy_series = engine.proxy_iv(rv_series, risk_premium)

        print(f"Calibracao: {calib_date.date()}  IV_real={iv_real:.2f}%  "
              f"RV_trailing={rv_series.loc[rv_calib_idx]:.2f}%  premio_de_risco={risk_premium:.3f}x")

        trades = engine.run_backtest(
            close, rv_forecast, iv_proxy_series, horizon=21, band_pct=1.0, target_vega=1000.0, spread_pct=0.05
        )
        stats = engine.summarize_backtest(trades, horizon=21)

        if stats["n_trades"] > 0:
            print()
            print(f"Trades: {stats['n_trades']} ({stats['n_long_vol']} compra vol, {stats['n_short_vol']} venda vol)")
            print(f"Taxa de acerto: {stats['win_rate']:.1%}")
            print(f"PnL total (unidades do modelo): {stats['total_pnl']:,.0f}")
            print(f"PnL medio por trade: {stats['avg_pnl']:,.0f}  (desvio: {stats['pnl_std']:,.0f})")
            print(f"Sharpe (anualizado): {stats['sharpe']:.3f}")

            trades.to_csv(OUT_DIR / "backtest_trades.csv", index=False)
            fig = plots.plot_cumulative_pnl(trades)
            fig.savefig(OUT_DIR / "backtest_pnl.png", dpi=150)
            print(f"\nbacktest_trades.csv ({len(trades)} trades) e backtest_pnl.png salvos.")
        else:
            print("Nenhum trade gerado com esses parametros.")

        _print_header("5b. Risco fiscal (GDELT) na previsao de RV -- compara rendimento dos trades")
        if VOLUME_PROCESSED_PATH.exists():
            fiscal_news = load_fiscal_risk_series()
            dataset_fiscal = build_dataset(
                close, news=fiscal_news, horizon=21, news_smooth_window=21, daily_variance=daily_variance
            )
            # Mesmos folds (mesmo periodo de teste) para as duas versoes do
            # modelo -- comparacao justa. O dataset com noticia comeca mais
            # tarde (so a partir da 1a data com cobertura de risco fiscal no
            # GDELT), entao esse "baseline" tem MENOS trades que o da secao 5.
            fiscal_folds = purged_walk_forward_splits_by_step(
                dataset_fiscal, min_train_size=252, step_size=21, horizon=21, embargo_days=5
            )

            rv_forecast_baseline_aligned = engine.generate_oos_rv_forecast(
                dataset_fiscal, BASELINE_FEATURES, horizon=21, n_splits=5, embargo_days=5, folds=fiscal_folds
            )
            rv_forecast_fiscal = engine.generate_oos_rv_forecast(
                dataset_fiscal, NEWS_FEATURES, horizon=21, n_splits=5, embargo_days=5, folds=fiscal_folds
            )

            # Peso (coeficiente OLS) que o modelo atribui a noticia de risco
            # fiscal em cada fold -- estimado a partir dos dados, nao
            # escolhido a dedo (o time e mais forte em quant/eng que em ML;
            # regressao simples e defensavel > blend arbitrario).
            news_weights = [
                fit_har(train, NEWS_FEATURES, log_target=True).params.get("news", float("nan"))
                for train, _ in fiscal_folds
            ]

            trades_baseline_aligned = engine.run_backtest(
                close, rv_forecast_baseline_aligned, iv_proxy_series, horizon=21, band_pct=1.0,
                target_vega=1000.0, spread_pct=0.05,
            )
            trades_fiscal = engine.run_backtest(
                close, rv_forecast_fiscal, iv_proxy_series, horizon=21, band_pct=1.0,
                target_vega=1000.0, spread_pct=0.05,
            )
            stats_baseline_aligned = engine.summarize_backtest(trades_baseline_aligned, horizon=21)
            stats_fiscal = engine.summarize_backtest(trades_fiscal, horizon=21)

            print(f"Peso medio da noticia de risco fiscal (coef. OLS, log-RV) nos "
                  f"{len(fiscal_folds)} folds: {np.nanmean(news_weights):+.4f} "
                  f"(desvio: {np.nanstd(news_weights):.4f})")
            print()
            print(f"{'':22s}{'SEM risco fiscal':>20s}{'COM risco fiscal':>20s}")
            for label, key in [
                ("Trades", "n_trades"), ("Taxa de acerto", "win_rate"),
                ("PnL total", "total_pnl"), ("PnL medio", "avg_pnl"), ("Sharpe", "sharpe"),
            ]:
                v_base = stats_baseline_aligned.get(key, float("nan"))
                v_fiscal = stats_fiscal.get(key, float("nan"))
                if key == "win_rate":
                    print(f"{label:22s}{v_base:>19.1%} {v_fiscal:>19.1%}")
                elif key in ("total_pnl", "avg_pnl"):
                    print(f"{label:22s}{v_base:>20,.0f}{v_fiscal:>20,.0f}")
                elif key == "sharpe":
                    print(f"{label:22s}{v_base:>20.3f}{v_fiscal:>20.3f}")
                else:
                    print(f"{label:22s}{v_base:>20.0f}{v_fiscal:>20.0f}")

            trades_baseline_aligned.to_csv(OUT_DIR / "backtest_trades_baseline_aligned.csv", index=False)
            trades_fiscal.to_csv(OUT_DIR / "backtest_trades_fiscal_risk.csv", index=False)
            print(f"\nbacktest_trades_baseline_aligned.csv ({len(trades_baseline_aligned)} trades) e "
                  f"backtest_trades_fiscal_risk.csv ({len(trades_fiscal)} trades) salvos.")
        else:
            print(
                "Risco fiscal (GDELT) ainda nao coletado -- rode "
                "data.gdelt_news.load_gdelt_volume_processed primeiro."
            )

        _print_header("5c. Risco fiscal REFINADO (surpresa + sentimento) -- compara rendimento dos trades")
        if FISCAL_SENTIMENT_PROCESSED_PATH.exists():
            fiscal_surprise_series = fiscal_risk_surprise(load_fiscal_risk_series(), window=63)
            fiscal_sentiment_df = pd.read_parquet(FISCAL_SENTIMENT_PROCESSED_PATH)
            fiscal_sentiment_series = fiscal_sentiment_df.set_index("date")["sentiment_mean"].sort_index()

            dataset_fiscal_v2 = ablation.build_dataset_with_fiscal_risk(
                close, fiscal_surprise_series, fiscal_sentiment_series,
                horizon=21, daily_variance=daily_variance, sentiment_smooth_window=5,
            )
            fiscal_v2_folds = purged_walk_forward_splits_by_step(
                dataset_fiscal_v2, min_train_size=252, step_size=21, horizon=21, embargo_days=5
            )

            rv_forecast_baseline_v2 = engine.generate_oos_rv_forecast(
                dataset_fiscal_v2, BASELINE_FEATURES, horizon=21, n_splits=5, embargo_days=5, folds=fiscal_v2_folds
            )
            rv_forecast_fiscal_v2 = engine.generate_oos_rv_forecast(
                dataset_fiscal_v2, ablation.FISCAL_RISK_FEATURES, horizon=21, n_splits=5, embargo_days=5,
                folds=fiscal_v2_folds,
            )

            v2_weights = {"fiscal_surprise": [], "fiscal_sentiment": []}
            for train, _ in fiscal_v2_folds:
                model = fit_har(train, ablation.FISCAL_RISK_FEATURES, log_target=True)
                v2_weights["fiscal_surprise"].append(model.params.get("fiscal_surprise", float("nan")))
                v2_weights["fiscal_sentiment"].append(model.params.get("fiscal_sentiment", float("nan")))

            trades_baseline_v2 = engine.run_backtest(
                close, rv_forecast_baseline_v2, iv_proxy_series, horizon=21, band_pct=1.0,
                target_vega=1000.0, spread_pct=0.05,
            )
            trades_fiscal_v2 = engine.run_backtest(
                close, rv_forecast_fiscal_v2, iv_proxy_series, horizon=21, band_pct=1.0,
                target_vega=1000.0, spread_pct=0.05,
            )
            stats_baseline_v2 = engine.summarize_backtest(trades_baseline_v2, horizon=21)
            stats_fiscal_v2 = engine.summarize_backtest(trades_fiscal_v2, horizon=21)

            for feat_name, weights in v2_weights.items():
                print(f"Peso medio de '{feat_name}' (coef. OLS, log-RV) nos {len(fiscal_v2_folds)} folds: "
                      f"{np.nanmean(weights):+.4f} (desvio: {np.nanstd(weights):.4f})")
            print()
            print(f"{'':22s}{'SEM risco fiscal':>20s}{'COM risco fiscal v2':>22s}")
            for label, key in [
                ("Trades", "n_trades"), ("Taxa de acerto", "win_rate"),
                ("PnL total", "total_pnl"), ("PnL medio", "avg_pnl"), ("Sharpe", "sharpe"),
            ]:
                v_base = stats_baseline_v2.get(key, float("nan"))
                v_fiscal = stats_fiscal_v2.get(key, float("nan"))
                if key == "win_rate":
                    print(f"{label:22s}{v_base:>19.1%} {v_fiscal:>21.1%}")
                elif key in ("total_pnl", "avg_pnl"):
                    print(f"{label:22s}{v_base:>20,.0f}{v_fiscal:>22,.0f}")
                elif key == "sharpe":
                    print(f"{label:22s}{v_base:>20.3f}{v_fiscal:>22.3f}")
                else:
                    print(f"{label:22s}{v_base:>20.0f}{v_fiscal:>22.0f}")

            trades_baseline_v2.to_csv(OUT_DIR / "backtest_trades_baseline_v2_aligned.csv", index=False)
            trades_fiscal_v2.to_csv(OUT_DIR / "backtest_trades_fiscal_risk_v2.csv", index=False)
            print(f"\nbacktest_trades_baseline_v2_aligned.csv ({len(trades_baseline_v2)} trades) e "
                  f"backtest_trades_fiscal_risk_v2.csv ({len(trades_fiscal_v2)} trades) salvos.")
        else:
            print(
                "Sentimento fiscal (FinBERT) ainda nao coletado -- rode "
                "sentiment.daily_index.load_fiscal_risk_sentiment_index() primeiro."
            )
    else:
        print("Superficie de IV ainda nao coletada -- rode data.iv_surface primeiro.")

    print()
    print(f"Arquivos gerados em: {OUT_DIR}")


if __name__ == "__main__":
    main()
