"""Inferencia sobre o Sharpe com operacoes SOBREPOSTAS: erro padrao HAC e
bootstrap de bloco estudentizado.

O PROBLEMA QUE ISSO RESOLVE. O backtest do projeto so aceita trades NAO
sobrepostos, o que da 40 operacoes a partir de 224 pregoes com IV real. Com
n=40 o erro padrao do Sharpe e ~0,16, entao qualquer valor entre -0,3 e +0,3 e
indistinguivel de zero -- nenhum modelo consegue produzir resultado
reportavel nesse regime. Pior: QUAIS 40 datas se escolhe e arbitrario, e este
projeto ja mediu que isso importa (selecionar janelas independentes por
POSICAO deu R2 +0,09 e por DATA deu -1,09 nos MESMOS dados).

O QUE SOBREPOR RESOLVE E O QUE NAO RESOLVE. Entrar em toda data com sinal usa
as 224 datas e elimina a arbitrariedade da escolha -- o Sharpe estimado passa
a ser a media sobre TODAS as datas de inicio possiveis, nao sobre uma amostra
sortuda. Isso reduz genuinamente a variancia do ESTIMADOR. Mas NAO multiplica
a informacao por 5,6: trades que compartilham 20 dos 21 dias carregam quase o
mesmo P&L, e tratar isso como 224 observacoes independentes e exatamente a
armadilha nº 2 deste projeto (janelas sobrepostas inflam p-valor). A serie de
P&L por data de entrada tem estrutura aproximada de MA(h-1) por construcao.

A SOLUCAO, em duas camadas, ambas reportadas:
1. Erro padrao HAC (Newey-West) com defasagens ate o horizonte, que corrige
   analiticamente a autocorrelacao induzida pela sobreposicao.
2. Bootstrap de bloco ESTUDENTIZADO (Ledoit & Wolf, 2008, "Robust performance
   hypothesis testing with the Sharpe ratio", Journal of Empirical Finance
   15(5):850-859), que nao supoe normalidade -- necessario aqui, onde o P&L
   tem assimetria -1,6 e kurtose de 4 a 11. O bloco preserva a dependencia;
   o comprimento e fixado no HORIZONTE, que e a estrutura conhecida da
   sobreposicao, NAO um parametro varrido.

ANUALIZACAO -- ponto que muda o numero e passa despercebido: o fator continua
sendo sqrt(252/horizonte), e nao sqrt(numero de trades por ano). Amostrar mais
datas de inicio estima melhor o MESMO Sharpe por ciclo de operacao; nao
encurta o ciclo. Usar sqrt(252) porque se entrou em 224 dias inflaria o
resultado por um fator de 3,5.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _sharpe(x: np.ndarray, annualization: float) -> float:
    desvio = x.std(ddof=1)
    if desvio == 0:
        return 0.0
    return float(x.mean() / desvio * np.sqrt(annualization))


def _hac_covariance(v: np.ndarray, lags: int) -> np.ndarray:
    """Covariancia HAC (Newey-West, com pesos de Bartlett) das colunas de `v`,
    que devem estar centradas."""
    n = v.shape[0]
    psi = v.T @ v / n
    for lag in range(1, lags + 1):
        peso = 1.0 - lag / (lags + 1.0)
        gamma = v[lag:].T @ v[:-lag] / n
        psi += peso * (gamma + gamma.T)
    return psi


def sharpe_se_hac(pnl, annualization: float = 12.0, lags: int | None = None) -> dict:
    """Erro padrao do Sharpe robusto a autocorrelacao e nao-normalidade, pelo
    metodo delta com covariancia HAC (Ledoit & Wolf, 2008, secao 3.1).

    O Sharpe e funcao de dois momentos, SR = mu / sqrt(gama2 - mu^2), com
    gama2 = E[r^2]. Aplica-se o metodo delta sobre o vetor (mu, gama2), cuja
    covariancia e estimada com Newey-West -- por isso o resultado ja incorpora
    tanto a cauda pesada quanto a autocorrelacao da sobreposicao.

    `lags=None` usa o horizonte implicito em `annualization` (252/annualization),
    que e a extensao exata da sobreposicao.
    """
    x = np.asarray(pnl, dtype=float)
    n = x.size
    if n < 3:
        return {"sharpe": float("nan"), "se": float("nan"), "t_stat": float("nan"),
                "p_value": float("nan"), "n": int(n)}

    if lags is None:
        lags = max(1, int(round(252.0 / annualization)) - 1)
    lags = min(lags, n - 2)

    mu = float(x.mean())
    gama2 = float((x**2).mean())
    var = gama2 - mu**2
    if var <= 0:
        return {"sharpe": 0.0, "se": float("nan"), "t_stat": float("nan"),
                "p_value": float("nan"), "n": int(n)}

    v = np.column_stack([x - mu, x**2 - gama2])
    psi = _hac_covariance(v, lags)

    # gradiente de SR em relacao a (mu, gama2)
    grad = np.array([gama2 / var**1.5, -mu / (2 * var**1.5)])
    se_nao_anual = float(np.sqrt(grad @ psi @ grad / n))

    fator = np.sqrt(annualization)
    sr = mu / np.sqrt(var) * fator
    se = se_nao_anual * fator
    t = sr / se if se > 0 else float("nan")
    return {
        "sharpe": float(sr),
        "se": se,
        "t_stat": float(t),
        "p_value": float(2 * stats.norm.sf(abs(t))) if np.isfinite(t) else float("nan"),
        "n": int(n),
        "lags_hac": int(lags),
    }


def _blocos_circulares(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Indices de uma reamostra por blocos CIRCULARES (Politis & Romano, 1992):
    o fim da serie emenda no comeco, o que evita que as observacoes das pontas
    sejam subamostradas."""
    n_blocos = int(np.ceil(n / block_size))
    inicios = rng.integers(0, n, size=n_blocos)
    idx = np.concatenate([np.arange(i, i + block_size) for i in inicios]) % n
    return idx[:n]


def block_bootstrap_sharpe(
    pnl,
    annualization: float = 12.0,
    block_size: int | None = None,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Intervalo de confianca e p-valor do Sharpe por bootstrap de bloco
    ESTUDENTIZADO (Ledoit & Wolf, 2008).

    Estudentizado, e nao percentil simples, porque o percentil supoe que a
    distribuicao do estimador nao depende do seu proprio desvio -- suposicao
    ruim com cauda pesada. Aqui cada reamostra produz seu proprio t, e o
    intervalo sai dos quantis desses t.

    H0 testada: Sharpe = 0.

    `block_size=None` usa o horizonte (252/annualization), que e a extensao da
    sobreposicao entre operacoes -- estrutura CONHECIDA, nao parametro
    ajustado. Varrer o comprimento do bloco ate o p-valor agradar seria
    exatamente a busca que o Deflated Sharpe existe para punir.
    """
    x = np.asarray(pnl, dtype=float)
    n = x.size
    if n < 10:
        return {"n": int(n), "sharpe": float("nan"), "ic_baixo": float("nan"),
                "ic_alto": float("nan"), "p_value": float("nan")}

    if block_size is None:
        block_size = max(1, int(round(252.0 / annualization)))
    block_size = min(block_size, n // 2)

    base = sharpe_se_hac(x, annualization=annualization)
    sr_hat, se_hat = base["sharpe"], base["se"]
    if not np.isfinite(se_hat) or se_hat <= 0:
        return {**base, "ic_baixo": float("nan"), "ic_alto": float("nan"),
                "block_size": int(block_size), "n_boot": 0}

    rng = np.random.default_rng(seed)
    ts = np.empty(n_boot)
    ts[:] = np.nan
    for b in range(n_boot):
        amostra = x[_blocos_circulares(n, block_size, rng)]
        r = sharpe_se_hac(amostra, annualization=annualization)
        if np.isfinite(r["se"]) and r["se"] > 0:
            ts[b] = (r["sharpe"] - sr_hat) / r["se"]

    ts = ts[np.isfinite(ts)]
    if ts.size < 100:
        return {**base, "ic_baixo": float("nan"), "ic_alto": float("nan"),
                "block_size": int(block_size), "n_boot": int(ts.size)}

    q_baixo, q_alto = np.quantile(ts, [1 - alpha / 2, alpha / 2])
    # IC estudentizado: os quantis entram INVERTIDOS (subtraidos), por
    # construcao do pivo t = (SR* - SR_hat)/se*.
    ic_baixo = sr_hat - q_baixo * se_hat
    ic_alto = sr_hat - q_alto * se_hat

    t_obs = sr_hat / se_hat
    p = float((np.abs(ts) >= abs(t_obs)).mean())

    return {
        **base,
        "ic_baixo": float(ic_baixo),
        "ic_alto": float(ic_alto),
        "p_value_bootstrap": p,
        "block_size": int(block_size),
        "n_boot": int(ts.size),
        "significante_5pct": bool(p < alpha),
    }


def effective_sample_size(n_trades: int, span_days: int, horizon: int) -> dict:
    """Quantas observacoes INDEPENDENTES existem de fato por tras de `n_trades`
    operacoes sobrepostas.

    Sobrepor aumenta o numero de trades mas nao a informacao: o teto e o numero
    de janelas nao sobrepostas que cabem no periodo, `span_days / horizon`.
    Reportar isso ao lado do n bruto e o que impede a leitura errada de "agora
    temos 224 operacoes".
    """
    teto = max(1, int(span_days // horizon))
    return {
        "n_trades": int(n_trades),
        "n_efetivo": int(min(n_trades, teto)),
        "fator_de_sobreposicao": float(n_trades / min(n_trades, teto)),
    }