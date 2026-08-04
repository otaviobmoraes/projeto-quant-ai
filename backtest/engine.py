"""Motor de backtest de P&L para a estrategia de straddle ATM (RV prevista
vs IV -- strategy/signal.py + strategy/sizing.py).

ILUSTRATIVO, nao evidencia de lucro real: a B3 so publica o snapshot do dia
da superficie de IV (sem historico pra download -- ver data/iv_surface.py),
entao so temos 1 dia real de IV. Pra simular uma serie historica de trades,
a IV de ENTRADA de cada trade e uma PROXY: RV realizada trailing (Parkinson)
escalada por um premio de risco FIXO, calibrado no unico dia real que temos
(`calibrate_risk_premium`). Essa e a simplificacao central e deve ser
comunicada sempre que o resultado for reportado.

Tudo o resto do trade usa dado 100% real:
- O preco de entrada/saida (fx_spot ou PTAX).
- O payoff terminal do straddle, |S_{t+horizon} - K| -- exato, sem
  aproximacao de delta-hedge (padrao pra estrategia de NIVEL de vol, nao de
  gamma scalping intradiario).
- A previsao de RV que gera o sinal e sempre FORA DA AMOSTRA (walk-forward
  purgado, backtest/walk_forward.py) -- nunca a previsao in-sample do
  proprio periodo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.metrics import directional_accuracy, sharpe_ratio
from backtest.walk_forward import purged_walk_forward_splits
from backtest.costs import transaction_cost
from strategy.signal import LONG_VOL, NO_TRADE, generate_signal
from strategy.sizing import size_straddle
from vol.black76 import call_price, put_price
from vol.forecast import TRADING_DAYS_PER_YEAR, fit_har, persistence_forecast, predict


def generate_oos_rv_forecast(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    n_splits: int,
    embargo_days: int,
    log_target: bool = True,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> pd.Series:
    """Serie de previsao de RV FORA DA AMOSTRA (walk-forward purgado): em
    cada fold, ajusta o modelo so no treino e preve no teste, concatenando
    as previsoes de todos os folds -- o "historico de previsoes" que o
    modelo teria produzido rodando ao vivo. Datas fora de qualquer fold de
    teste (treino inicial, janelas de purga) ficam sem previsao.

    `folds`: se informado, usa esses folds prontos (ex.:
    `purged_walk_forward_splits_by_step`, reestimando com mais frequencia)
    em vez de gerar via `purged_walk_forward_splits(n_splits=...)`.
    """
    if folds is None:
        folds = purged_walk_forward_splits(
            dataset, n_splits=n_splits, horizon=horizon, embargo_days=embargo_days
        )
    preds = []
    for train, test in folds:
        model = fit_har(train, feature_cols, log_target=log_target)
        preds.append(predict(model, test, feature_cols, log_target=log_target))
    if not preds:
        return pd.Series(dtype=float)
    return pd.concat(preds).sort_index()


def evaluate_directional_accuracy_per_fold(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    n_splits: int,
    embargo_days: int,
    log_target: bool = True,
    reference: pd.Series | None = None,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> list[dict]:
    """Acuracia direcional (o modelo previu do lado certo de `reference`?)
    fold a fold, nos mesmos folds purgados de generate_oos_rv_forecast.

    `reference`: serie externa pra comparar (ex.: IV proxy). Se None, usa a
    persistencia (RV atual, `vol.forecast.persistence_forecast`) como
    referencia -- testa se o modelo acerta a DIRECAO da mudanca de RV, sem
    depender de nenhuma proxy de IV.

    `folds`: ver generate_oos_rv_forecast.
    """
    if folds is None:
        folds = purged_walk_forward_splits(
            dataset, n_splits=n_splits, horizon=horizon, embargo_days=embargo_days
        )
    results = []
    for train, test in folds:
        model = fit_har(train, feature_cols, log_target=log_target)
        pred = predict(model, test, feature_cols, log_target=log_target)
        ref = reference.reindex(test.index) if reference is not None else persistence_forecast(test)
        results.append(directional_accuracy(pred, test["target"], ref))
    return results


def calibrate_risk_premium(rv_at_calibration_date: float, iv_at_calibration_date: float) -> float:
    """Multiplicador IV/RV calibrado no UNICO dia com IV real conhecida.
    Assumido CONSTANTE ao longo de toda a historia simulada -- a
    simplificacao central desta proxy (na pratica o premio de risco varia no
    tempo, sobretudo em estresse; nao ha dado ainda pra estimar essa
    variacao).
    """
    return iv_at_calibration_date / rv_at_calibration_date


def proxy_iv(rv_trailing: pd.Series, risk_premium: float) -> pd.Series:
    """IV de entrada PROXY = RV trailing (Parkinson, mesma janela do
    horizonte de trade) x premio de risco fixo calibrado."""
    return (rv_trailing * risk_premium).rename("iv_proxy")


def run_backtest(
    prices: pd.Series,
    rv_forecast: pd.Series,
    iv_proxy_series: pd.Series,
    horizon: int = 21,
    band_pct: float = 1.0,
    target_vega: float = 1000.0,
    spread_pct: float = 0.05,
    r: float = 0.0,
) -> pd.DataFrame:
    """Simula uma serie de trades de straddle ATM NAO SOBREPOSTOS: sempre
    que ha sinal (fora da banda morta), abre a posicao no fechamento do dia
    (K = preco do dia, IV = iv_proxy_series do dia) e resolve o payoff
    terminal exato `horizon` dias uteis depois (|S_{t+horizon} - K|), sem
    reabrir posicao antes desse trade expirar.

    Retorna um DataFrame com uma linha por trade executado.
    """
    common_index = prices.index.intersection(rv_forecast.index).intersection(iv_proxy_series.index)
    common_index = common_index.sort_values()

    records = []
    i = 0
    n = len(common_index)
    while i < n:
        t = common_index[i]
        rv_hat = rv_forecast.loc[t]
        iv_t = iv_proxy_series.loc[t]
        if pd.isna(rv_hat) or pd.isna(iv_t) or iv_t <= 0:
            i += 1
            continue

        sig = generate_signal(rv_hat, iv_t, band_pct=band_pct)
        if sig == NO_TRADE:
            i += 1
            continue

        pos_in_prices = prices.index.get_loc(t)
        exit_pos = pos_in_prices + horizon
        if exit_pos >= len(prices):
            break  # sem dado suficiente pra resolver o trade ate o fim
        exit_date = prices.index[exit_pos]

        spot_entry = float(prices.loc[t])
        spot_exit = float(prices.loc[exit_date])
        K = spot_entry
        T = horizon / 365
        sigma = iv_t / 100

        premium = call_price(spot_entry, K, T, sigma, r) + put_price(spot_entry, K, T, sigma, r)
        n_contracts = size_straddle(
            target_vega=target_vega, spot=spot_entry, ttm_days=horizon, iv_pct=iv_t, r=r
        )
        payoff = abs(spot_exit - K)
        cost = transaction_cost(premium, n_contracts, spread_pct)

        # LONG_VOL: compra o straddle (paga premio, recebe payoff).
        # SHORT_VOL: vende o straddle (recebe premio, paga payoff).
        direction = 1 if sig == LONG_VOL else -1
        pnl_gross = direction * n_contracts * (payoff - premium)
        pnl_net = pnl_gross - cost

        records.append(
            {
                "entry_date": t,
                "exit_date": exit_date,
                "signal": sig,
                "rv_forecast": rv_hat,
                "iv_proxy": iv_t,
                "n_contracts": n_contracts,
                "premium": premium,
                "payoff": payoff,
                "cost": cost,
                "pnl_gross": pnl_gross,
                "pnl_net": pnl_net,
            }
        )

        # nao sobrepoe trades: proximo trade so depois deste expirar.
        next_idx = common_index.searchsorted(exit_date)
        i = max(next_idx, i + 1)

    return pd.DataFrame(records)


def summarize_backtest(trades: pd.DataFrame, horizon: int = 21) -> dict:
    """Resumo do backtest: numero de trades, taxa de acerto, PnL total/medio
    e Sharpe (anualizado assumindo ~252/horizon trades por ano, ja que os
    trades nao se sobrepoem). PnL em unidades do modelo (normalizado pelo
    vega alvo de referencia), nao uma unidade monetaria de contrato B3 real.
    """
    if trades.empty:
        return {"n_trades": 0}

    capital_at_risk = (trades["n_contracts"] * trades["premium"]).abs()
    trade_returns = (trades["pnl_net"] / capital_at_risk).to_numpy()

    trades_per_year = TRADING_DAYS_PER_YEAR / horizon
    sharpe = sharpe_ratio(trade_returns, annualization_factor=trades_per_year)

    return {
        "n_trades": int(len(trades)),
        "win_rate": float((trades["pnl_net"] > 0).mean()),
        "total_pnl": float(trades["pnl_net"].sum()),
        "avg_pnl": float(trades["pnl_net"].mean()),
        "pnl_std": float(trades["pnl_net"].std()),
        "sharpe": sharpe,
        "n_long_vol": int((trades["signal"] == LONG_VOL).sum()),
        "n_short_vol": int((trades["signal"] != LONG_VOL).sum()),
    }
