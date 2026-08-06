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
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew

from backtest import ablation, engine
from backtest.metrics import deflated_sharpe_ratio, pooled_oos_metrics
from backtest.walk_forward import purged_walk_forward_splits, purged_walk_forward_splits_by_step
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
from vol.forecast import BASELINE_FEATURES, NEWS_FEATURES, build_dataset, fit_har, persistence_forecast

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
    "Risco fiscal (GDELT, % de cobertura, nivel bruto) -- nao ajuda",
    "Risco fiscal (surpresa/z-score, janela 63d) -- nao ajuda",
    "Risco fiscal (surpresa/z-score, janela 21d) -- efeito marginal, dentro do ruido",
    "Diagnostico R2: media por fold vs POOLED -- media por fold e instavel com "
    "folds pequenos (explode negativo); adotado R2 pooled como metrica oficial",
    "Sweep de esquema de walk-forward (5/8/10 folds fixos + 6 variantes de "
    "step_size) com R2 pooled -- so 1 config bateu persistencia (378d/42d), "
    "vizinhos proximos NAO bateram -- rejeitado como selecao de config por sorte, "
    "mantido 5 folds fixos como esquema oficial",
    "HAR-RV + termo de leverage (rv_d x indicador de retorno negativo) -- sem "
    "efeito (multicolinearidade com rv_d/rv_w/rv_m)",
    "Risco fiscal REFINADO v2 (surpresa + sentimento FinBERT-PT-BR, 15/15 "
    "janelas coletadas, 22 dias com manchete real) -- com dado parcial (12/15) "
    "parecia ajudar (R2 +0.073, 3 de 5 folds), mas com a coleta completa "
    "INVERTEU (R2 -0.568) -- artefato de amostra pequena, descartado",
    "Pesquisa na literatura (Kambouroudis et al. 2021; Barndorff-Nielsen, "
    "Kinnebrock & Shephard 2010) + 3 extensoes testadas com R2 pooled: "
    "overnight return (gap fechamento->abertura), leverage via semivariancia "
    "(rv_d_pos/rv_d_neg, decomposicao aditiva em vez da interacao "
    "multiplicativa) e ensemble (media das 3 variantes) -- nenhuma moveu o "
    "R2 de forma perceptivel (delta entre +0.004 e -0.001, dentro do ruido); "
    "todas continuam perdendo pra persistencia pura",
    "Risco global exogeno (VIX + DXY via yfinance, primeira feature que NAO "
    "deriva do proprio preco/imprensa do USD/BRL) -- R2 pooled: baseline "
    "-0.011, so VIX -0.020 (pior), so DXY -0.012 (sem efeito), VIX+DXY "
    "-0.020 -- nenhuma ajuda; persistencia continua em +0.134 no mesmo "
    "periodo. 12a tentativa consecutiva sem melhorar o R2 nesse dataset.",
    "DECISAO: persistencia pura (rv_m) adotada como previsao OFICIAL da "
    "estrategia (secao 5) no lugar do HAR-RV -- bateu o HAR-RV em TODAS as "
    "13 comparacoes de R2 pooled feitas neste projeto (R2 +0.126 vs -0.510 "
    "no esquema oficial). O backtest ilustrativo (Sharpe) PIOROU com essa "
    "troca (1.087 -> 0.014) apesar do R2 melhorar -- diagnostico: "
    "RV_previsto (persistencia) e IV_proxy tem correlacao ~0.994 (IV_proxy "
    "= RV_trailing x premio fixo, persistencia = RV_trailing quase igual), "
    "entao o spread que decide compra/venda e dominado por uma constante "
    "multiplicativa, nao por sinal -- R2 mede acerto medio em TODOS os "
    "dias, taxa de acerto mede so o subconjunto pequeno e nao-aleatorio de "
    "dias em que esse spread quase-constante cruzou a banda por ruido de "
    "curto prazo. R2 (pooled, medido contra o valor real) continua a "
    "metrica confiavel; Sharpe do backtest ilustrativo nao deveria ser "
    "usado pra escolher entre modelos ate haver IV real historica.",
    "Persistencia + CORRECAO DE RESIDUO (regressao no residuo target-"
    "persistencia, ver backtest.engine.generate_residual_corrected_forecast) "
    "-- testadas as 6 camadas ja avaliadas contra o HAR-RV (so rv_d/rv_w/rv_m, "
    "noticia, credibilidade, risco fiscal v2, overnight, leverage, VIX+DXY) "
    "como corretoras da persistencia em vez de substitutas dela. TODAS "
    "pioraram o R2 pooled vs persistencia pura, sem excecao (delta entre "
    "-0.133 e -1.851) -- persistencia nao tem parametro nenhum (zero risco "
    "de overfitting); qualquer correcao via regressao introduz variancia de "
    "estimacao sem sinal real pra compensar, piorando a previsao. 19a "
    "tentativa consecutiva sem melhorar o modelo nesse dataset -- "
    "simplicidade (persistencia pura, sem nenhuma camada) e o resultado "
    "mais robusto encontrado.",
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

    Le o R2 POOLED (backtest.metrics.pooled_oos_metrics, concatena as
    previsoes de todos os folds antes de calcular) como numero principal --
    o R2 medio por fold (`result["baseline"]`) fica instavel com folds
    pequenos (a media de cada fold, usada como referencia do R2, tem alta
    variancia amostral) e pode exagerar o quao ruim o modelo parece. A
    contagem de "melhora em N de M folds" continua usando o R2 por fold, que
    ainda serve bem pra esse proposito especifico (robustez/consistencia).
    """
    baseline_r2 = result["baseline_pooled"]["r2_oos"]
    other_r2 = result[f"{comparison_key}_pooled"]["r2_oos"]
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
    print(f"Baseline (HAR-RV, Parkinson):  {summary.format_metrics(result['baseline_pooled'])}")
    print(f"Com noticia:                   {summary.format_metrics(result['com_noticia_pooled'])}")

    # Persistencia pura (rv_m, sem nenhum parametro ajustado) como segundo
    # baseline de referencia -- em TODAS as comparacoes feitas neste projeto
    # (13 tentativas: HAR-RV puro, com noticia/credibilidade/risco fiscal em
    # varias formas, extensoes de literatura, risco global exogeno), a
    # persistencia bateu o HAR-RV. Por isso a secao 5 (backtest) usa
    # persistencia como previsao oficial em vez do HAR-RV -- essa linha so
    # documenta a comparacao no dataset Parkinson padrao (nao exatamente o
    # mesmo recorte trimado por disponibilidade de noticia do `result`
    # acima, mas o mesmo esquema oficial de 5 folds).
    _persist_close, _persist_daily_variance = realized.load_parkinson_prices_and_variance()
    _persist_dataset = build_dataset(_persist_close, horizon=21, daily_variance=_persist_daily_variance)
    _persist_folds = purged_walk_forward_splits(_persist_dataset, n_splits=5, horizon=21, embargo_days=5)
    _persist_forecast = persistence_forecast(_persist_dataset).reindex(
        pd.concat([test["target"] for _, test in _persist_folds]).index
    )
    _persist_metrics = pooled_oos_metrics(_persist_dataset["target"], _persist_forecast)
    print(f"Persistencia pura (rv_m):      {summary.format_metrics(_persist_metrics)}")

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
        print(f"Baseline (HAR-RV, Parkinson):  {summary.format_metrics(cred_result['baseline_pooled'])}")
        print(f"Com credibilidade:             {summary.format_metrics(cred_result['com_credibilidade_pooled'])}")
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
            print(f"Baseline (HAR-RV, Parkinson):  {summary.format_metrics(fiscal_result['baseline_pooled'])}")
            print(f"Com risco fiscal:              {summary.format_metrics(fiscal_result['com_noticia_pooled'])}")
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
            print(f"Baseline (HAR-RV, Parkinson):     {summary.format_metrics(fiscal_v2_result['baseline_pooled'])}")
            print(f"Com risco fiscal refinado:        {summary.format_metrics(fiscal_v2_result['com_risco_fiscal_v2_pooled'])}")
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
        # Persistencia pura (rv_m) em vez de HAR-RV: em TODAS as 13
        # comparacoes feitas neste projeto (HAR-RV puro, com camadas de
        # noticia/credibilidade/risco fiscal, extensoes de literatura,
        # risco global exogeno), a persistencia bateu o HAR-RV no R2 pooled
        # -- ver secao 1 (+0.126 vs -0.510 no esquema oficial). Sem
        # coeficiente nenhum pra ajustar, entao nao precisa de walk-forward
        # (nao ha risco de overfitting num modelo sem parametros).
        rv_forecast = persistence_forecast(dataset)

        iv_df = implied.load_iv_atm_processed(target_days=21)
        calib_date = iv_df["refdate"].iloc[-1]
        iv_real = iv_df["iv_atm_pct"].iloc[-1]
        rv_calib_idx = rv_series.index.asof(calib_date)
        risk_premium = engine.calibrate_risk_premium(rv_series.loc[rv_calib_idx], iv_real)
        iv_proxy_series = engine.proxy_iv(rv_series, risk_premium)

        print(f"Calibracao: {calib_date.date()}  IV_real={iv_real:.2f}%  "
              f"RV_trailing={rv_series.loc[rv_calib_idx]:.2f}%  premio_de_risco={risk_premium:.3f}x")

        # Diagnostico: RV_previsto (persistencia) e IV_proxy sao QUASE A
        # MESMA SERIE -- IV_proxy = RV_trailing_21d x 0.847 (premio de risco
        # fixo), e persistencia = RV_trailing_22d (rv_m). Correlacao ~0.994
        # nesse periodo. Isso significa que o spread que decide compra/venda
        # e dominado por uma constante multiplicativa (~1.18x), nao por
        # sinal genuino -- o momento exato em que ele cruza a banda de 1.0
        # e ditado por ruido de curtissimo prazo na vol recente, nao pela
        # qualidade da previsao. E por isso que trocar HAR-RV por
        # persistencia MELHOROU o R2 (+0.126 vs -0.510, secao 1) mas
        # PIOROU o backtest (Sharpe 1.087->0.014): R2 mede acerto medio em
        # TODOS os dias da amostra; taxa de acerto mede so os poucos dias
        # em que o spread cruzou a banda -- um subconjunto pequeno e
        # nao-aleatorio, escolhido por um limiar sensivel a ruido quando a
        # previsao e quase colinear com a propria proxy de IV. Reforca por
        # que a secao 5 e ILUSTRATIVA: o problema maior nao e qual modelo
        # usamos, e a IV-proxy ser um multiplo constante da RV trailing.
        _corr = rv_forecast.corr(iv_proxy_series.reindex(rv_forecast.index))
        print(f"Diagnostico: correlacao RV_previsto x IV_proxy = {_corr:.3f} "
              "(quase colineares -- ver nota no codigo sobre por que isso "
              "desconecta R2 de taxa de acerto)")

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

            # Deflated Sharpe Ratio: varia a banda morta (band_pct) -- o
            # hiperparametro mais natural da estrategia -- como familia de
            # tentativas, e ajusta o Sharpe da config escolhida (1.0) pelo
            # numero de variacoes testadas (guardrail do CLAUDE.md: registrar
            # configuracoes testadas + DSR, nao so reportar a melhor sem
            # disclosure).
            band_pcts_tested = [0.5, 1.0, 1.5, 2.0]
            trial_sharpes = [stats["sharpe"]]
            for bp in band_pcts_tested:
                if bp == 1.0:
                    continue
                t = engine.run_backtest(
                    close, rv_forecast, iv_proxy_series, horizon=21, band_pct=bp,
                    target_vega=1000.0, spread_pct=0.05,
                )
                s = engine.summarize_backtest(t, horizon=21)
                if s["n_trades"] > 1:
                    trial_sharpes.append(s["sharpe"])

            if len(trial_sharpes) >= 2 and stats["n_trades"] > 1:
                capital_at_risk = (trades["n_contracts"] * trades["premium"]).abs()
                trade_returns = (trades["pnl_net"] / capital_at_risk).to_numpy()
                dsr = deflated_sharpe_ratio(
                    sr_hat=stats["sharpe"],
                    sr_trials_std=float(np.std(trial_sharpes, ddof=1)),
                    n_trials=len(trial_sharpes),
                    n_obs=stats["n_trades"],
                    skew=float(_skew(trade_returns)),
                    kurtosis=float(_kurtosis(trade_returns, fisher=False)),
                )
                print(f"Deflated Sharpe Ratio: {dsr:.3f}  "
                      f"(band_pct=1.0 escolhido entre {len(trial_sharpes)} variacoes testadas "
                      f"{band_pcts_tested}, Sharpes={[round(s, 2) for s in trial_sharpes]})")

            trades.to_csv(OUT_DIR / "backtest_trades.csv", index=False)
            fig = plots.plot_cumulative_pnl(trades)
            fig.savefig(OUT_DIR / "backtest_pnl.png", dpi=150)
            print(f"\nbacktest_trades.csv ({len(trades)} trades) e backtest_pnl.png salvos.")
        else:
            print("Nenhum trade gerado com esses parametros.")

        _print_header("5b. Risco fiscal (GDELT) na previsao de RV -- compara rendimento dos trades")
        print(
            "Nota: esta secao (e a 5c) testam se a camada ajuda um HAR-RV AJUSTADO -- "
            "pergunta de pesquisa diferente da secao 5 acima, que ja usa persistencia "
            "(sem parametros) como previsao oficial por bater o HAR-RV em toda comparacao feita.\n"
        )
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
