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


# Limites da busca da vol implicita. 0.1% a 500% ao ano cobre com folga
# qualquer regime de USD/BRL (o pico da COVID nao passou de ~40%); o limite
# superior largo existe so para o brentq ter onde procurar em premios
# absurdos (erro de digitacao/negocio fora de mercado), que sao rejeitados
# depois pelo filtro de sanidade de quem chama.
_SIGMA_MIN = 1e-3
_SIGMA_MAX = 5.0

# Valor no tempo minimo (como fracao de F) para a inversao ser BEM-POSTA.
# Motivo: numa opcao fundo dentro do dinheiro o premio e intrinseco puro ate
# a precisao do float (o premio e a diferenca de dois numeros grandes quase
# iguais -- cancelamento catastrofico), o vega vai a zero e NENHUMA sigma
# explica o preco melhor que outra. Devolver um numero ali seria inventar
# precisao que o dado nao tem. Limite deliberadamente permissivo: em F~4000
# equivale a ~0,004 de valor no tempo, o que so descarta o caso degenerado.
_MIN_TIME_VALUE_FRAC = 1e-6


def implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    option_type: str,
    r: float = 0.0,
) -> float | None:
    """Vol implicita (DECIMAL) invertida numericamente do preco de uma opcao
    sobre futuro, via Brent -- o "inverter via Black-76 (numerico)" previsto
    no CLAUDE.md como diferencial de rigor.

    Brent (e nao Newton com vega) porque nao precisa de derivada, nao diverge
    perto do vencimento -- onde o vega tende a zero e Newton fica instavel --
    e da convergencia garantida uma vez que a raiz esta entre os limites.

    Devolve None (em vez de levantar) quando NAO ha vol que explique o preco:

    - preco abaixo do valor intrinseco (arbitragem estatica) -- tipico de
      negocio fora de mercado ou de dado sujo;
    - preco acima do teto teorico (F para call, K para put, descontados);
    - valor no tempo pequeno demais para a inversao ser bem-posta (opcao
      fundo dentro do dinheiro -- ver _MIN_TIME_VALUE_FRAC);
    - preco fora do intervalo alcancavel em [_SIGMA_MIN, _SIGMA_MAX].

    Retornar None e deliberado: numa serie histórica montada de NEGOCIOS
    esparsos, uma fracao dos registros e inevitavelmente inutilizavel, e a
    chamada precisa poder descartar linha a linha sem abortar a coleta.

    `option_type`: "C" (call) ou "P" (put).
    """
    if T <= 0 or price <= 0 or F <= 0 or K <= 0:
        return None

    kind = option_type.upper()
    if kind not in ("C", "P"):
        raise ValueError(f'option_type precisa ser "C" ou "P", recebido {option_type!r}')

    pricer = call_price if kind == "C" else put_price
    df = _discount_factor(r, T)

    # Limites de nao-arbitragem: fora deles nenhuma sigma > 0 resolve.
    intrinsic = df * max(F - K, 0.0) if kind == "C" else df * max(K - F, 0.0)
    ceiling = df * F if kind == "C" else df * K
    if price <= intrinsic or price >= ceiling:
        return None
    if price - intrinsic <= _MIN_TIME_VALUE_FRAC * F:
        return None

    lo, hi = _SIGMA_MIN, _SIGMA_MAX
    if pricer(F, K, T, lo, r) > price or pricer(F, K, T, hi, r) < price:
        return None

    from scipy.optimize import brentq

    try:
        return float(brentq(lambda s: pricer(F, K, T, s, r) - price, lo, hi, xtol=1e-8))
    except (ValueError, RuntimeError):
        return None
