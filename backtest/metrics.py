"""Metricas robustas de avaliacao: Sharpe, Probabilistic Sharpe Ratio (PSR),
Deflated Sharpe Ratio (DSR) e acuracia direcional.

Bailey, D. H. & Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality".

Guardrail do CLAUDE.md: "Registrar o numero de configuracoes testadas e usar
Deflated Sharpe Ratio. Nao repetir o grid search cego da referencia de
2023." O DSR ajusta o Sharpe observado pelo numero de configuracoes
testadas (`n_trials`) -- sem isso, escolher a melhor entre N variantes e so
reportar essa superestima sistematicamente a performance esperada fora da
amostra (overfitting de selecao).

`directional_accuracy`: a estrategia so precisa do SINAL de (RV previsto -
referencia), nao da magnitude exata -- R2/RMSE punem erro de magnitude
pesado mesmo quando o sinal de compra/venda de vol estaria certo. E uma
metrica mais proxima do que a estrategia de fato precisa acertar.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest, norm

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns, annualization_factor: int = 252) -> float:
    """Sharpe anualizado (assume taxa livre de risco ja descontada dos
    retornos, i.e. `returns` sao retornos em excesso)."""
    returns = np.asarray(returns, dtype=float)
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float(returns.mean() / std * math.sqrt(annualization_factor))


def probabilistic_sharpe_ratio(
    sr_hat: float, benchmark_sr: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """P(SR verdadeiro > benchmark_sr | Sharpe observado = sr_hat), ajustando
    pela nao-normalidade dos retornos. `kurtosis` na convencao normal=3 (nao
    curtose em excesso).
    """
    denom = math.sqrt(1 - skew * sr_hat + ((kurtosis - 1) / 4) * sr_hat**2)
    z = (sr_hat - benchmark_sr) * math.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe_ratio(sr_trials_std: float, n_trials: int) -> float:
    """Sharpe maximo esperado por puro acaso ao testar `n_trials`
    configuracoes independentes cujos Sharpes tem dispersao `sr_trials_std`
    (aproximacao de teoria de valores extremos; requer n_trials >= 2).
    """
    if n_trials < 2:
        raise ValueError("n_trials precisa ser >= 2 para a aproximacao de valor extremo")
    return float(
        sr_trials_std
        * (
            (1 - EULER_MASCHERONI) * norm.ppf(1 - 1 / n_trials)
            + EULER_MASCHERONI * norm.ppf(1 - 1 / (n_trials * math.e))
        )
    )


def deflated_sharpe_ratio(
    sr_hat: float,
    sr_trials_std: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR: PSR avaliado contra o Sharpe maximo esperado por acaso dado
    `n_trials` configuracoes testadas -- quanto mais configuracoes testadas
    (maior `n_trials`) ou mais dispersos os resultados entre elas (maior
    `sr_trials_std`), maior a barra que o Sharpe observado precisa vencer
    para nao ser so sorte de selecao.
    """
    benchmark = expected_max_sharpe_ratio(sr_trials_std, n_trials)
    return probabilistic_sharpe_ratio(sr_hat, benchmark, n_obs, skew, kurtosis)


def pooled_oos_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """RMSE/MAE/R2 fora da amostra POOLED: concatena as previsoes de TODOS os
    folds do walk-forward antes de calcular, em vez de calcular R2 fold a
    fold e tirar a media (o que `vol.forecast.evaluate` faz).

    Por que isso importa: o R2 por fold usa a MEDIA DAQUELE FOLD como
    referencia (`ss_tot`). Com folds pequenos (poucas dezenas de
    observacoes, ex.: reestimacao mensal), essa media fica instavel e o R2
    de um fold isolado pode explodir para valores extremamente negativos
    mesmo com erro absoluto pequeno -- e a media desses R2 instaveis fica
    dominada pelos piores folds, exagerando o quao "quebrado" o modelo
    parece. R2 pooled usa a media de TODO o periodo fora da amostra como
    referencia unica -- e o padrao em avaliacao de ML financeiro (ex.: Gu,
    Kelly & Xiu 2020, "Empirical Asset Pricing via Machine Learning") e bem
    mais estavel a esse efeito de tamanho de fold.
    """
    combined = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1, join="inner").dropna()
    aligned_actual, aligned_pred = combined["actual"], combined["predicted"]
    err = aligned_actual - aligned_pred
    ss_res = float((err**2).sum())
    ss_tot = float(((aligned_actual - aligned_actual.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(err.abs().mean()),
        "r2_oos": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_obs": int(len(aligned_actual)),
    }


def diebold_mariano(
    actual: pd.Series,
    pred_a: pd.Series,
    pred_b: pd.Series,
    horizon: int = 1,
    independent_only: bool = False,
) -> dict:
    """Teste de Diebold-Mariano (1995) -- H0: os dois modelos tem a MESMA
    acuracia preditiva. Perda quadratica; `pred_a` e a candidata e `pred_b` a
    referencia, entao t POSITIVO significa que A erra MENOS que B.

    E o TERCEIRO PORTAO do criterio do projeto (delta R2 > 0, >=4/5 folds,
    DM p < 0.05) e ate agora era calculado ad-hoc em script, o que abre espaco
    para inconsistencia entre analises. Aqui fica uma implementacao unica.

    `independent_only=True` subamostra 1 observacao a cada `horizon` para que
    as janelas do alvo NAO se sobreponham. E a diferenca que este projeto ja
    mediu valer um veredito: com janelas sobrepostas os MESMOS dados deram
    p=0.2166 onde a versao independente deu p=0.9531, porque observacoes
    vizinhas compartilham quase todos os retornos e inflam o n efetivo. Em
    h=1 as duas versoes coincidem (o alvo nao se sobrepoe).

    Sem correcao HAC: a subamostragem independente ja remove a autocorrelacao
    que a correcao trataria, e e o caminho mais transparente de auditar.
    """
    df = pd.concat(
        [actual.rename("y"), pred_a.rename("a"), pred_b.rename("b")], axis=1, join="inner"
    ).dropna()
    if independent_only and horizon > 1:
        df = df.iloc[::horizon]

    d = ((df["y"] - df["a"]) ** 2 - (df["y"] - df["b"]) ** 2).to_numpy()
    n = int(d.size)
    if n < 3:
        return {
            "n": n, "t_stat": float("nan"), "p_value": float("nan"), "media_perda": float("nan")
        }

    desvio = float(np.std(d, ddof=1))
    if desvio == 0:
        return {"n": n, "t_stat": 0.0, "p_value": 1.0, "media_perda": 0.0}

    # d < 0 significa que A tem perda menor. Invertemos o sinal para que
    # t POSITIVO signifique "A e melhor", que e como o projeto le o resultado.
    t_stat = -float(np.mean(d)) / (desvio / np.sqrt(n))
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    return {"n": n, "t_stat": t_stat, "p_value": p_value, "media_perda": float(np.mean(d))}


def directional_accuracy(forecast: pd.Series, actual: pd.Series, reference: pd.Series) -> dict:
    """O modelo "acerta" se prever corretamente de que lado de `reference` o
    valor `actual` vai cair -- ex.: reference = RV atual (persistencia) ou
    IV: o que importa pra estrategia e se RV_previsto e RV_realizado ficam
    do MESMO LADO da referencia, nao se a magnitude prevista bate exata.

    Inclui um teste binomial (H0: acerto = 50%, i.e. equivalente a cara ou
    coroa) -- accuracy sozinha nao diz se o resultado e estatisticamente
    diferente de sorte.
    """
    aligned = pd.concat(
        [forecast.rename("forecast"), actual.rename("actual"), reference.rename("reference")],
        axis=1,
        join="inner",
    ).dropna()

    predicted_sign = np.sign(aligned["forecast"] - aligned["reference"])
    actual_sign = np.sign(aligned["actual"] - aligned["reference"])
    correct = predicted_sign == actual_sign

    n = int(len(correct))
    hits = int(correct.sum())
    accuracy = hits / n if n > 0 else float("nan")
    pvalue = float(binomtest(hits, n, 0.5).pvalue) if n > 0 else float("nan")

    return {"n": n, "hits": hits, "accuracy": accuracy, "pvalue_vs_50pct": pvalue}


def max_drawdown(returns) -> dict:
    """Maior queda acumulada da curva de patrimonio construida por
    capitalizacao composta dos retornos por trade.

    Devolve a profundidade (fracao do pico), o indice do pico e o do vale --
    os indices permitem localizar QUANDO o rebaixamento aconteceu, que e a
    pergunta que um avaliador faz depois de ver a magnitude.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return {"max_drawdown": float("nan"), "pico": -1, "vale": -1}
    equity = np.cumprod(1.0 + r)
    picos = np.maximum.accumulate(equity)
    rebaixamento = equity / picos - 1.0
    vale = int(np.argmin(rebaixamento))
    return {
        "max_drawdown": float(rebaixamento[vale]),
        "pico": int(np.argmax(equity[: vale + 1])) if vale >= 0 else -1,
        "vale": vale,
    }


def annualized_return(returns, trades_per_year: float) -> float:
    """Retorno anualizado GEOMETRICO dos retornos por trade.

    Geometrico, e nao media aritmetica vezes o numero de trades, porque a
    estrategia reinveste: uma sequencia +50%/-50% tem media aritmetica zero e
    retorno real de -25%. Reportar a media aritmetica aqui superestimaria
    sistematicamente o desempenho.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0 or np.any(r <= -1.0):
        return float("nan")
    total = float(np.prod(1.0 + r))
    anos = r.size / trades_per_year
    return float(total ** (1.0 / anos) - 1.0) if anos > 0 else float("nan")


def performance_summary(
    returns,
    n_trials: int,
    trades_per_year: float = 12.0,
    sr_trials_std: float | None = None,
) -> dict:
    """Conjunto completo de metricas de desempenho de uma serie de retornos
    POR TRADE (nao sobrepostos).

    `n_trials`: numero de configuracoes testadas no projeto ate aqui. Entra no
    Deflated Sharpe, que desconta do Sharpe observado aquilo que se esperaria
    obter por acaso ao testar muitas configuracoes (Bailey & Lopez de Prado,
    2014). Passar o numero REAL -- inclusive experimentos de previsao, nao so
    variantes de backtest -- e a escolha conservadora e a correta.

    `sr_trials_std`: dispersao dos Sharpes entre as configuracoes testadas. Se
    nao informada, usa 1.0, que e deliberadamente PUNITIVO: quanto maior a
    dispersao, mais alto o Sharpe maximo esperado por acaso e menor o DSR.

    ADVERTENCIA sobre o denominador: os retornos usados aqui sao relativos ao
    premio comprometido no trade. Para venda de volatilidade a perda potencial
    e ILIMITADA, entao esse denominador SUBESTIMA o risco real da perna
    vendida. Qualquer leitura de "retorno sobre capital" aqui precisa carregar
    essa ressalva.
    """
    r = np.asarray(returns, dtype=float)
    n = int(r.size)
    if n == 0:
        return {"n_trades": 0}

    sr = sharpe_ratio(r, annualization_factor=trades_per_year)
    dd = max_drawdown(r)
    skew = float(stats.skew(r)) if n > 2 else 0.0
    kurt = float(stats.kurtosis(r, fisher=False)) if n > 3 else 3.0

    return {
        "n_trades": n,
        "sharpe": sr,
        "retorno_anualizado": annualized_return(r, trades_per_year),
        "retorno_medio_por_trade": float(r.mean()),
        "volatilidade_por_trade": float(r.std(ddof=1)) if n > 1 else float("nan"),
        "max_drawdown": dd["max_drawdown"],
        "trade_do_vale": dd["vale"],
        "win_rate": float((r > 0).mean()),
        "skew": skew,
        "kurtose": kurt,
        "psr_vs_zero": probabilistic_sharpe_ratio(sr, 0.0, n, skew, kurt),
        # DSR nao e definido com menos de 2 configuracoes: nao existe "maximo
        # esperado por acaso" sobre uma amostra de uma. Devolvemos NaN em vez
        # de estourar, para o resumo continuar utilizavel em diagnostico.
        "deflated_sharpe": (
            deflated_sharpe_ratio(
                sr, sr_trials_std if sr_trials_std is not None else 1.0, n_trials, n, skew, kurt
            )
            if n_trials >= 2
            else float("nan")
        ),
        "n_trials_usado": int(n_trials),
    }


def max_drawdown_absolute(pnl) -> dict:
    """Maior rebaixamento da curva de P&L ACUMULADO, em unidades absolutas.

    Diferente de `max_drawdown`, que capitaliza retornos percentuais: aqui a
    curva e a soma acumulada do P&L. Existe porque para uma carteira VENDIDA em
    opcao a perda de um unico trade pode superar varias vezes o premio
    comprometido, o que torna qualquer retorno percentual sobre premio
    inferior a -100% -- e capitalizacao composta abaixo de -100% nao tem
    sentido (o patrimonio ficaria negativo e depois "recuperaria").
    """
    x = np.asarray(pnl, dtype=float)
    if x.size == 0:
        return {"max_drawdown": float("nan"), "pico": -1, "vale": -1}
    curva = np.cumsum(x)
    picos = np.maximum.accumulate(curva)
    rebaixamento = curva - picos
    vale = int(np.argmin(rebaixamento))
    return {
        "max_drawdown": float(rebaixamento[vale]),
        "pico": int(np.argmax(curva[: vale + 1])),
        "vale": vale,
    }


def pnl_performance_summary(
    pnl,
    n_trials: int,
    trades_per_year: float = 12.0,
    sr_trials_std: float | None = None,
) -> dict:
    """Resumo de desempenho a partir do P&L POR TRADE, em unidades absolutas.

    POR QUE NAO HA "RETORNO SOBRE CAPITAL" AQUI: a estrategia vende
    volatilidade na maioria das operacoes, e a perda de uma venda de straddle e
    ilimitada -- nao ha base de capital derivavel do premio que a absorva. O
    capital efetivamente exigido e a MARGEM, que depende de regras da camara e
    nao esta nos dados. Declarar uma base arbitraria produziria justamente o
    numero mais visivel da secao a partir de uma suposicao inventada.

    O que sobrevive a essa limitacao, e por isso e o que reportamos:
    - SHARPE, que e invariante a escala (dividir todo o P&L por qualquer
      constante nao o altera), portanto valido sem base de capital;
    - PSR e DEFLATED SHARPE, que derivam do Sharpe;
    - drawdown em unidades ABSOLUTAS, bem definido sobre P&L acumulado.
    """
    x = np.asarray(pnl, dtype=float)
    n = int(x.size)
    if n == 0:
        return {"n_trades": 0}

    sr = sharpe_ratio(x, annualization_factor=trades_per_year)
    dd = max_drawdown_absolute(x)
    skew = float(stats.skew(x)) if n > 2 else 0.0
    kurt = float(stats.kurtosis(x, fisher=False)) if n > 3 else 3.0

    return {
        "n_trades": n,
        "sharpe": sr,
        "pnl_total": float(x.sum()),
        "pnl_medio_por_trade": float(x.mean()),
        "pnl_por_ano": float(x.sum() / (n / trades_per_year)),
        "volatilidade_por_trade": float(x.std(ddof=1)) if n > 1 else float("nan"),
        "max_drawdown_abs": dd["max_drawdown"],
        "trade_do_vale": dd["vale"],
        "maior_perda": float(x.min()),
        "maior_ganho": float(x.max()),
        "win_rate": float((x > 0).mean()),
        "skew": skew,
        "kurtose": kurt,
        "psr_vs_zero": probabilistic_sharpe_ratio(sr, 0.0, n, skew, kurt),
        "deflated_sharpe": (
            deflated_sharpe_ratio(
                sr, sr_trials_std if sr_trials_std is not None else 1.0, n_trials, n, skew, kurt
            )
            if n_trials >= 2
            else float("nan")
        ),
        "n_trials_usado": int(n_trials),
    }
