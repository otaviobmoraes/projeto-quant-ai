"""Apreçamento Black-76 (opcao sobre futuro/forward) e gregas.

Opcao de dolar (DOL/WDO) e opcao SOBRE FUTURO, nao sobre spot -- por isso
Black-76 (nao Black-Scholes-spot). Convencoes desta implementacao:

- `sigma` e `r` em DECIMAL (0.0916, nao 9.16) -- diferente do resto do
  projeto, que guarda IV/RV em pontos percentuais (iv_pct, rv_pct). Quem
  chama a partir de vol/implied.py ou vol/realized.py deve dividir por 100
  na fronteira.
- `T` em anos (dias corridos / 365).
- Sem curva de juros propria coletada ainda: `r=0.0` por padrao (desconto
  neutro) -- simplificacao explicita, documentada; afeta o NIVEL do premio,
  nao a forma do vega, que e o que a Fase 5 usa para dimensionar posicao.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def _d1_d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        raise ValueError("T e sigma precisam ser > 0")
    vol_sqrt_t = sigma * T**0.5
    d1 = (math.log(F / K) + (sigma**2 / 2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def call_price(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    d1, d2 = _d1_d2(F, K, T, sigma)
    df = _discount_factor(r, T)
    return df * (F * norm.cdf(d1) - K * norm.cdf(d2))


def put_price(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    d1, d2 = _d1_d2(F, K, T, sigma)
    df = _discount_factor(r, T)
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def vega(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Vega (igual para call e put no Black-76): variacao do premio para uma
    variacao de 1.00 (100 pontos percentuais) em sigma. Para vega "por ponto
    percentual de vol" (convencao usual de mesa), multiplique por 0.01.
    """
    d1, _ = _d1_d2(F, K, T, sigma)
    df = _discount_factor(r, T)
    return df * F * norm.pdf(d1) * T**0.5


def straddle_vega(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Vega do straddle (call + put no mesmo strike) = 2x o vega de uma perna,
    ja que call e put tem o mesmo vega no Black-76."""
    return 2 * vega(F, K, T, sigma, r)


def _discount_factor(r: float, T: float) -> float:
    return math.exp(-r * T)
