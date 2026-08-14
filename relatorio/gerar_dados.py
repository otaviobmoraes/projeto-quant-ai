"""Recalcula TODOS os numeros citados no relatorio final e gera as figuras.

Roda antes de `gerar_relatorio.py`. Salva:
  relatorio/resultados.json  -- numeros usados no texto (nada hard-coded la)
  relatorio/figuras/*.png    -- graficos

Motivo de existir: o relatorio nao pode ter numero digitado a mao. Se o
dado ou o modelo mudar, roda-se este script de novo e o relatorio reflete
o estado real do projeto.

Uso:
    python -m relatorio.gerar_dados
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from backtest import engine
from backtest.metrics import pooled_oos_metrics
from backtest.walk_forward import purged_walk_forward_splits
from credibility.credibility import PROCESSED_PATH as CREDIBILITY_PATH
from data.b3_futures import PROCESSED_PATH as B3_PATH
from data.fx_spot import PROCESSED_PATH as FX_PATH
from data.ptax import PROCESSED_PATH as PTAX_PATH
from vol.forecast import BASELINE_FEATURES, build_dataset, persistence_forecast
from vol.realized import (
    forward_realized_skewness,
    load_b3_futures_prices_and_variance,
    load_parkinson_prices_and_variance,
    parkinson_vol,
)

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figuras"

# Paleta consistente com report/plots.py
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SEC = "#3a3a38"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
GREEN = "#008300"
RED = "#d03b3b"
AMBER = "#b8860b"

HORIZONS = [1, 3, 5, 10, 15, 21]

# ---------------------------------------------------------------------------
# JANELA PRIMARIA DO RELATORIO -- nao remover sem ler isto.
#
# A amostra do projeto foi estendida de 830 para 2.135 pregoes (retroagindo a
# 2018-01). Se este script simplesmente usasse tudo, o R2 em h=21 saltaria de
# -0,255 para +0,381 e o relatorio passaria a AFIRMAR QUE O MODELO FUNCIONA.
#
# Esse salto e ARTEFATO, nao melhora, e esta diagnosticado em tres frentes
# (report/run_report.py:CONFIGS_TESTED): (a) a persistencia, que nao tem
# parametro nenhum, salta junto (-0,367 -> +0,365); (b) com periodo de TESTE
# fixo, treinar com 8,5 anos em vez de 1 muda o RMSE em 1,4% e o R2 continua
# negativo; (c) a autocorrelacao de cada subperiodo e MENOR que a da amostra
# completa. A amostra mistura regimes de vol muito diferentes, e isso infla o
# denominador (ss_tot) de qualquer metrica calculada no pool.
#
# Por isso os numeros PRINCIPAIS do relatorio ficam presos a janela original, e
# a amostra estendida entra como resultado SEPARADO e rotulado -- inclusive
# porque a comparacao entre as duas e, ela mesma, um dos achados.
# ---------------------------------------------------------------------------
JANELA_PRIMARIA_INICIO = "2023-04-10"


def _janela_primaria(*series: pd.Series) -> tuple[pd.Series, ...]:
    """Corta as series para a janela primaria do relatorio (ver acima)."""
    corte = pd.Timestamp(JANELA_PRIMARIA_INICIO)
    out = []
    for s in series:
        idx = s.index
        limite = corte.tz_localize(idx.tz) if getattr(idx, "tz", None) is not None else corte
        out.append(s[idx >= limite])
    return tuple(out)


def _axes(figsize=(9, 4.6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    return fig, ax


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"  figura: {name}")


def evaluate_source(close, variance, horizon):
    """R2 pooled do HAR-RV e da persistencia, nos MESMOS folds purgados."""
    ds = build_dataset(close, horizon=horizon, daily_variance=variance)
    folds = purged_walk_forward_splits(ds, n_splits=5, horizon=horizon, embargo_days=5)
    idx = pd.concat([t["target"] for _, t in folds]).index
    har = engine.generate_oos_rv_forecast(
        ds, BASELINE_FEATURES, horizon=horizon, n_splits=5, embargo_days=5, folds=folds
    )
    per = persistence_forecast(ds).reindex(idx)
    return {
        "har": pooled_oos_metrics(ds["target"], har),
        "persistencia": pooled_oos_metrics(ds["target"], per),
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = {}

    print("1/6 carregando fontes...")
    close_yf, var_yf = load_parkinson_prices_and_variance()
    close_b3_full, var_b3_full = load_b3_futures_prices_and_variance()
    # numeros principais presos a janela original -- ver JANELA_PRIMARIA_INICIO
    close_b3, var_b3 = _janela_primaria(close_b3_full, var_b3_full)
    fut_full = pd.read_parquet(B3_PATH).set_index("date").sort_index()
    _corte = pd.Timestamp(JANELA_PRIMARIA_INICIO).tz_localize(fut_full.index.tz)
    fut = fut_full[fut_full.index >= _corte]
    spot = pd.read_parquet(FX_PATH).set_index("date").sort_index()
    ptax = pd.read_parquet(PTAX_PATH)
    ptax = ptax[ptax["tipo"] == "venda"].set_index("date")["value"].sort_index()

    res["amostra"] = {
        "b3_pregoes": int(len(fut)),
        "b3_inicio": str(fut.index.min().date()),
        "b3_fim": str(fut.index.max().date()),
        "b3_contratos": int(fut["ticker"].nunique()),
        "b3_rolagens": int(fut["contract_changed"].sum()),
        "yf_dias": int(len(spot)),
    }
    res["amostra_estendida"] = {
        "b3_pregoes": int(len(fut_full)),
        "b3_inicio": str(fut_full.index.min().date()),
        "b3_fim": str(fut_full.index.max().date()),
        "janelas_independentes_original": int(len(fut) // 21),
        "janelas_independentes_estendida": int(len(fut_full) // 21),
    }

    print("2/6 varredura de horizonte (as duas fontes)...")
    sweep = {"yfinance": {}, "b3": {}}
    for h in HORIZONS:
        sweep["yfinance"][h] = evaluate_source(close_yf, var_yf, h)
        sweep["b3"][h] = evaluate_source(close_b3, var_b3, h)
        print(f"   h={h:2d} ok")
    res["horizonte"] = {
        fonte: {
            str(h): {
                "har_r2": v["har"]["r2_oos"],
                "persist_r2": v["persistencia"]["r2_oos"],
                "har_rmse": v["har"]["rmse"],
                "persist_rmse": v["persistencia"]["rmse"],
                "n_obs": v["har"]["n_obs"],
            }
            for h, v in fonte_dados.items()
        }
        for fonte, fonte_dados in sweep.items()
    }

    # ---- Figura: R2 x horizonte (o achado central) ----
    fig, ax = _axes((9.2, 4.8))
    ax.axhline(0, color=SEC, linewidth=1.2, zorder=2)
    ax.plot(HORIZONS, [sweep["b3"][h]["har"]["r2_oos"] for h in HORIZONS],
            "o-", color=BLUE, linewidth=2, label="HAR-RV (futuro B3)", zorder=3)
    ax.plot(HORIZONS, [sweep["b3"][h]["persistencia"]["r2_oos"] for h in HORIZONS],
            "s-", color=GREEN, linewidth=2, label="Persistência (futuro B3)", zorder=3)
    ax.plot(HORIZONS, [sweep["yfinance"][h]["persistencia"]["r2_oos"] for h in HORIZONS],
            "^--", color=MUTED, linewidth=1.6, label="Persistência (spot yfinance)", zorder=3)
    ax.axvline(21, color=RED, linewidth=1.2, linestyle=":", zorder=2)
    ax.text(20.4, ax.get_ylim()[1] * 0.92, "horizonte da\nestratégia", color=RED,
            fontsize=8.5, ha="right", va="top")
    ax.set_xticks(HORIZONS)  # horizonte e discreto: sem 2.5 ou 7.5 dias uteis
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_xlabel("Horizonte de previsão (dias úteis)", color=SEC, fontsize=10)
    ax.set_ylabel("R² fora da amostra (pooled)", color=SEC, fontsize=10)
    ax.set_title("Onde existe previsibilidade — e onde a estratégia precisa dela",
                 color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=SEC)
    _save(fig, "r2_por_horizonte.png")

    print("3/6 evidencia do bug de alinhamento...")
    intra = (spot["close"] - spot["open"]).abs()
    gap = (spot["open"] - spot["close"].shift(1)).abs()
    fut_intra = (fut["last"] - fut["open"]).abs() / 1000.0
    res["bug_alinhamento"] = {
        "yf_intra_medio": float(intra.mean()),
        "yf_gap_medio": float(gap.mean()),
        "yf_razao": float(gap.mean() / intra.mean()),
        "yf_corr_open_close": float(spot["close"].corr(spot["open"])),
        "b3_intra_medio": float(fut_intra.mean()),
    }

    fig, ax = _axes((8.4, 4.2))
    rotulos = ["Movimento DENTRO\nda barra\n|close − open|", "Salto ENTRE barras\n|open − close anterior|"]
    ax.bar(rotulos, [intra.mean(), gap.mean()], color=[RED, BLUE], width=0.5, zorder=3)
    ax.text(0, intra.mean(), f"  {intra.mean():.5f}", va="bottom", ha="center", color=SEC, fontsize=10)
    ax.text(1, gap.mean(), f"  {gap.mean():.5f}", va="bottom", ha="center", color=SEC, fontsize=10)
    ax.set_ylabel("BRL (média)", color=SEC, fontsize=10)
    ax.set_title(f"yfinance BRL=X: quase todo movimento acontece FORA da barra "
                 f"({gap.mean()/intra.mean():.0f}x)", color=INK, fontsize=11.5, loc="left")
    _save(fig, "bug_alinhamento.png")

    print("4/6 autocorrelacao das duas fontes...")
    lags = [1, 2, 3, 5, 10, 15, 21]
    acf_b3 = [float(var_b3.dropna().autocorr(l)) for l in lags]
    acf_yf = [float(var_yf.dropna().autocorr(l)) for l in lags]
    res["autocorrelacao"] = {"lags": lags, "b3": acf_b3, "yfinance": acf_yf}

    fig, ax = _axes((8.6, 4.2))
    w = 0.38
    x = np.arange(len(lags))
    ax.bar(x - w / 2, acf_b3, w, color=BLUE, label="Futuro B3 (sessão ~9h)", zorder=3)
    ax.bar(x + w / 2, acf_yf, w, color=MUTED, label="Spot yfinance (24h)", zorder=3)
    ax.axhline(0, color=AXIS, linewidth=1, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([f"lag {l}" for l in lags], fontsize=9)
    ax.set_ylabel("Autocorrelação da variância diária", color=SEC, fontsize=10)
    ax.set_title("A memória longa que a previsão de 21 dias exige existe no spot, não no futuro",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=SEC)
    _save(fig, "autocorrelacao.png")

    print("5/6 validacao de alinhamento contra PTAX...")
    fut_ret = np.log(fut["settlement"]).diff()
    fut_ret[fut["contract_changed"].values] = np.nan
    spot_ret = np.log(spot["close"]).diff()
    ptax_ret = np.log(ptax).diff()
    corr = {"futuro_b3": {}, "yfinance": {}}
    for k in (-1, 0, 1):
        for nome, serie in [("futuro_b3", fut_ret), ("yfinance", spot_ret)]:
            j = pd.concat([serie.rename("x"), ptax_ret.shift(k).rename("p")], axis=1, join="inner").dropna()
            corr[nome][str(k)] = float(j["x"].corr(j["p"]))
    res["validacao_ptax"] = corr

    fig, ax = _axes((8.4, 4.2))
    ks = [-1, 0, 1]
    x = np.arange(len(ks))
    ax.bar(x - 0.19, [corr["futuro_b3"][str(k)] for k in ks], 0.38, color=BLUE,
           label="Futuro B3 (ajuste oficial)", zorder=3)
    ax.bar(x + 0.19, [corr["yfinance"][str(k)] for k in ks], 0.38, color=MUTED,
           label="yfinance (close)", zorder=3)
    ax.axhline(0, color=AXIS, linewidth=1, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(["defasagem −1", "MESMO DIA", "defasagem +1"], fontsize=9.5)
    ax.set_ylabel("Correlação com retorno do PTAX", color=SEC, fontsize=10)
    ax.set_title("PTAX (BCB) como árbitro: só o futuro da B3 alinha no mesmo dia",
                 color=INK, fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=SEC)
    _save(fig, "validacao_ptax.png")

    print("6/6 historico de RV...")
    rv_b3 = parkinson_vol(fut["high"], fut["low"], window=21).dropna()
    rv_yf = parkinson_vol(spot["high"], spot["low"], window=21).dropna()
    res["rv"] = {
        "b3_media": float(rv_b3.mean()), "b3_max": float(rv_b3.max()), "b3_min": float(rv_b3.min()),
        "yf_media": float(rv_yf.mean()),
        "corr_rv21": float(pd.concat([rv_b3.rename("a"), rv_yf.rename("b")], axis=1,
                                      join="inner").dropna().corr().iloc[0, 1]),
    }

    fig, ax = _axes((9.4, 4.4))
    ax.plot(rv_b3.index, rv_b3.to_numpy(), color=BLUE, linewidth=1.6,
            label="Futuro de dólar B3 (oficial)", zorder=3)
    ax.plot(rv_yf.index, rv_yf.to_numpy(), color=MUTED, linewidth=1.2, alpha=0.85,
            label="Spot yfinance (proxy)", zorder=3)
    ax.set_ylabel("RV 21d anualizada (% a.a.)", color=SEC, fontsize=10)
    ax.set_title("Volatilidade realizada do USD/BRL — as duas fontes", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=SEC)
    _save(fig, "rv_historico.png")

    print("7/7 credibilidade x assimetria (teste na dimensao que a teoria preve)...")
    # A teoria de Barro-Gordon preve efeito no SKEW (cauda direita), nao no
    # NIVEL de vol. Como nao ha historico de skew IMPLICITO (a B3 sobrescreve
    # a superficie), usamos a contraparte REALIZADA.
    ret_b3 = np.log(fut["settlement"]).diff()
    ret_b3[fut["contract_changed"].values] = np.nan  # emenda de contrato nao e movimento real
    cred = pd.read_parquet(CREDIBILITY_PATH).set_index("date").sort_index()

    skew_res = {}
    for h in (10, 21, 42):
        sk = forward_realized_skewness(ret_b3, horizon=h)
        df = pd.DataFrame({"skew": sk})
        df["theta"] = cred["theta_baseline"].reindex(df.index, method="ffill")
        df = df.dropna()
        # Janelas de skew futura se SOBREPOEM: observacoes consecutivas
        # compartilham quase todos os retornos, o que infla artificialmente a
        # significancia. Reportamos as duas versoes; a independente e a valida.
        ind = df.iloc[::h]
        r_o, p_o = stats.pearsonr(df["theta"], df["skew"])
        r_i, p_i = stats.pearsonr(ind["theta"], ind["skew"])
        skew_res[str(h)] = {
            "skew_media": float(df["skew"].mean()),
            "n_sobrepostas": int(len(df)), "r_sobrepostas": float(r_o), "p_sobrepostas": float(p_o),
            "n_independentes": int(len(ind)), "r_independentes": float(r_i), "p_independentes": float(p_i),
        }
    res["credibilidade_skew"] = skew_res

    print("7/9 investigacoes novas (amostra estendida, ML, estimadores)...")
    res.update(_investigacoes_novas(close_b3_full, var_b3_full))

    print("8/9 IV propria a partir de negocios de opcao...")
    res["iv_propria"] = _iv_propria()

    print("9/9 h=1, combinacao, rolagem, GARCH, camadas e backtest com IV real...")
    res["h1_positivo"] = _h1_positivo(var_b3_full)
    res["combinacao"] = _combinacao(var_b3_full)
    res["rolagem"] = _rolagem()
    res["garch"] = _garch(close_b3_full)
    res["camadas_horizonte_curto"] = _camadas_horizonte_curto()
    res["backtest_iv_real"] = _backtest_iv_real(close_b3_full, var_b3_full)
    res["backtest_completo"] = _backtest_completo(close_b3_full, var_b3_full)

    (OUT_DIR / "resultados.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nresultados.json salvo ({len(res)} blocos)")


def _oos_pooled(var_features, var_target, horizon, modelo="har"):
    """R2/RMSE pooled de um modelo nos folds purgados, com features de uma
    serie de variancia e ALVO de outra (permite variar so um dos dois)."""
    from vol.forecast import (
        fit_har,
        forward_target_from_variance,
        har_features_from_variance,
        predict,
    )
    from vol.realized import TRADING_DAYS_PER_YEAR

    ds = har_features_from_variance(var_features)
    ds["target"] = forward_target_from_variance(var_target, horizon)
    ds = ds.dropna()
    folds = purged_walk_forward_splits(ds, 5, horizon, 5)
    preds, alvos, por_fold = [], [], []
    for tr, te in folds:
        if modelo == "persistencia":
            p = np.sqrt(te["rv_m"] * TRADING_DAYS_PER_YEAR) * 100
        else:
            p = predict(
                fit_har(tr, BASELINE_FEATURES, log_target=True),
                te, BASELINE_FEATURES, log_target=True,
            )
        preds.append(p)
        alvos.append(te["target"])
        por_fold.append(pooled_oos_metrics(te["target"], p)["r2_oos"])
    a, p = pd.concat(alvos), pd.concat(preds)
    m = pooled_oos_metrics(a, p)
    return {"r2_oos": m["r2_oos"], "rmse": m["rmse"], "n_obs": int(len(a)), "por_fold": por_fold}


def _investigacoes_novas(close_full: pd.Series, var_full: pd.Series) -> dict:
    """Resultados das investigacoes feitas depois da primeira versao do
    relatorio: amostra estendida, ML, estimadores de variancia, combinacao e a
    analise por subperiodo (que e a conclusao metodologica central)."""
    from vol.realized import load_b3_variance

    out: dict = {}
    var_orig = _janela_primaria(var_full)[0]

    # --- por que a amostra estendida NAO deve virar o numero principal ------
    ext = {}
    for h in (1, 21):
        ext[str(h)] = {
            "original_har": _oos_pooled(var_orig, var_orig, h)["r2_oos"],
            "original_persist": _oos_pooled(var_orig, var_orig, h, "persistencia")["r2_oos"],
            "estendida_har": _oos_pooled(var_full, var_full, h)["r2_oos"],
            "estendida_persist": _oos_pooled(var_full, var_full, h, "persistencia")["r2_oos"],
        }
    out["artefato_amostra"] = ext

    # --- autocorrelacao por subperiodo (o diagnostico que sustenta tudo) ----
    janelas = {
        "2018-2019": ("2018-01-01", "2020-01-01"),
        "2021-2022": ("2021-01-01", "2023-01-01"),
        "2023-2026": ("2023-01-01", "2027-01-01"),
    }
    acf = {"lags": [1, 5, 10, 21], "completa": [], "por_subperiodo": {}}
    acf["completa"] = [float(var_full.autocorr(lag)) for lag in acf["lags"]]
    for rot, (a, b) in janelas.items():
        s = var_full[(var_full.index >= a) & (var_full.index < b)].dropna()
        acf["por_subperiodo"][rot] = [float(s.autocorr(lag)) for lag in acf["lags"]]
    out["autocorrelacao_subperiodo"] = acf

    # --- estimadores de variancia, com alvo ARBITRO (close-to-close) -------
    _, arbitro = load_b3_variance("close_to_close")
    estim = {}
    for e in ("parkinson", "garman_klass", "rogers_satchell", "full_day"):
        _, v = load_b3_variance(e)
        estim[e] = {str(h): _oos_pooled(v, arbitro, h)["r2_oos"] for h in (1, 5, 21)}
    out["estimadores"] = {
        "nota": "alvo arbitro close-to-close: nao pertence a nenhum candidato",
        "r2": estim,
    }

    # --- ML: XGBoost vs HAR, mesmos folds e features ------------------------
    from backtest.ml_ablation import run_ml_ablation

    ml = {}
    for h in (1, 5, 21):
        r = run_ml_ablation(var_full, horizon=h)
        modelos = ("har", "xgb_arvore", "xgb_linear", "persistencia")
        ml[str(h)] = {k: r[k]["r2_oos"] for k in modelos}
    out["ml"] = ml

    # --- combinacao de previsoes (parecia o melhor achado; nao sobrevive) ---
    comb = {}
    for rot, v in [("COMPLETA", var_full)] + [
        (r, var_full[(var_full.index >= a) & (var_full.index < b)]) for r, (a, b) in janelas.items()
    ]:
        linha = {}
        for h in (1, 21):
            if len(v) < 250:
                continue
            har = _oos_pooled(v, v, h)
            per = _oos_pooled(v, v, h, "persistencia")
            linha[str(h)] = {"har": har["r2_oos"], "persist": per["r2_oos"]}
        comb[rot] = linha
    out["combinacao_subperiodo"] = comb
    return out


SUBPERIODOS = {
    "2018-2019": ("2018-01-01", "2020-01-01"),
    "2021-2022": ("2021-01-01", "2023-01-01"),
    "2023-2026": ("2023-01-01", "2027-01-01"),
}


def _fatias(var: pd.Series) -> dict[str, pd.Series]:
    out = {r: var[(var.index >= a) & (var.index < b)] for r, (a, b) in SUBPERIODOS.items()}
    out["COMPLETA"] = var
    return out


def _dm(alvo, f1, f2, passo):
    """Diebold-Mariano com janelas INDEPENDENTES (uma a cada `passo`).
    d < 0 => f1 melhor. Devolve (n, t, p)."""
    d = (((alvo - f1) ** 2) - ((alvo - f2) ** 2)).iloc[:: max(passo, 1)]
    n = len(d)
    if n < 3:
        return n, float("nan"), float("nan")
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    return n, float(t), float(2 * (1 - stats.t.cdf(abs(t), n - 1)))


def _har_e_persistencia(var: pd.Series, horizon: int):
    """Previsoes OOS de HAR e persistencia nos mesmos folds purgados."""
    from vol.forecast import (
        fit_har,
        forward_target_from_variance,
        har_features_from_variance,
        predict,
    )
    from vol.realized import TRADING_DAYS_PER_YEAR

    ds = har_features_from_variance(var)
    ds["target"] = forward_target_from_variance(var, horizon)
    ds = ds.dropna()
    folds = purged_walk_forward_splits(ds, 5, horizon, 5)
    H, P, A, fh = [], [], [], 0
    for tr, te in folds:
        modelo = fit_har(tr, BASELINE_FEATURES, log_target=True)
        h = predict(modelo, te, BASELINE_FEATURES, log_target=True)
        p = np.sqrt(te["rv_m"] * TRADING_DAYS_PER_YEAR) * 100
        H.append(h)
        P.append(p)
        A.append(te["target"])
        r_h = pooled_oos_metrics(te["target"], h)["r2_oos"]
        r_p = pooled_oos_metrics(te["target"], p)["r2_oos"]
        if r_h > r_p:
            fh += 1
    return pd.concat(A), pd.concat(H), pd.concat(P), fh, len(folds)


def _h1_positivo(var: pd.Series) -> dict:
    """O unico resultado positivo do projeto: em h=1 o alvo NAO se sobrepoe,
    entao cada observacao e independente e o Diebold-Mariano e legitimo."""
    out = {}
    for rot, v in _fatias(var).items():
        if len(v) < 250:
            continue
        a, h, p, fh, nf = _har_e_persistencia(v, 1)
        n, t, pv = _dm(a, h, p, 1)
        out[rot] = {
            "n": int(len(a)),
            "har_r2": pooled_oos_metrics(a, h)["r2_oos"],
            "persist_r2": pooled_oos_metrics(a, p)["r2_oos"],
            "folds_har_melhor": fh, "n_folds": nf,
            "dm_t": t, "dm_p": pv,
        }
    return out


def _combinacao(var: pd.Series) -> dict:
    """Combinacao de pesos iguais (Bates & Granger 1969). Parecia o melhor
    achado do projeto na amostra completa e NAO sobrevive ao subperiodo --
    e o quinto falso positivo, o mais didatico deles."""
    out = {}
    for rot, v in _fatias(var).items():
        if len(v) < 250:
            continue
        linha = {}
        for h in (1, 21):
            a, hh, pp, _, _ = _har_e_persistencia(v, h)
            cc = (hh + pp) / 2
            r_h = pooled_oos_metrics(a, hh)["r2_oos"]
            r_p = pooled_oos_metrics(a, pp)["r2_oos"]
            base = hh if r_h >= r_p else pp
            n, t, pv = _dm(a, cc, base, h)
            linha[str(h)] = {
                "har": r_h, "persist": r_p,
                "combinado": pooled_oos_metrics(a, cc)["r2_oos"],
                "delta": pooled_oos_metrics(a, cc)["r2_oos"] - max(r_h, r_p),
                "dm_p": pv,
            }
        out[rot] = linha
    return out


def _rolagem() -> dict:
    """Ciclo de rolagem: a monotonia aparente que se revelou artefato de
    agregacao, e a correcao que nao ajuda."""
    from backtest.roll_ablation import run_roll_ablation
    from vol.roll import deseasonalize, load_b3_futures_with_dte, seasonal_factor

    _, var, dte = load_b3_futures_with_dte()
    fator = seasonal_factor(var, dte)
    ajust = deseasonalize(var, dte, fator)
    out = {
        "fator_amostra_completa": {str(k): float(v) for k, v in fator.items()},
        "acf_original": [float(var.autocorr(lag)) for lag in (1, 5, 21)],
        "acf_dessazonalizada": [float(ajust.autocorr(lag)) for lag in (1, 5, 21)],
        "ablacao": {},
    }
    for h in (1, 21):
        r = run_roll_ablation(var, dte, horizon=h)
        out["ablacao"][str(h)] = {
            "baseline": r["baseline_pooled"]["r2_oos"],
            "dessazonalizada": r["dessazonalizada_pooled"]["r2_oos"],
            "com_dte": r["com_dte_pooled"]["r2_oos"],
        }
    return out


def _garch(precos: pd.Series) -> dict:
    """GARCH(1,1) contra HAR sobre o MESMO input (retorno^2) -- o veredito
    antigo tinha sido obtido sobre fonte defeituosa e so em h=21."""
    from vol.garch import garch_forward_target_forecast
    from vol.realized import log_returns

    var_ret = log_returns(precos) ** 2
    out = {}
    for h in (1, 5, 21):
        a, hh, _, _, _ = _har_e_persistencia(var_ret, h)
        from vol.forecast import forward_target_from_variance, har_features_from_variance

        ds = har_features_from_variance(var_ret)
        ds["target"] = forward_target_from_variance(var_ret, h)
        ds = ds.dropna()
        folds = purged_walk_forward_splits(ds, 5, h, 5)
        G = []
        for tr, te in folds:
            try:
                G.append(garch_forward_target_forecast(precos, tr.index[-1], te.index, h))
            except Exception:
                G.append(pd.Series(np.nan, index=te.index))
        g = pd.concat(G)
        ok = g.notna()
        out[str(h)] = {
            "garch": pooled_oos_metrics(a[ok], g[ok])["r2_oos"],
            "har_mesmo_input": pooled_oos_metrics(a[ok], hh[ok])["r2_oos"],
            "n": int(ok.sum()),
        }
    return out


def _camadas_horizonte_curto() -> dict:
    """As camadas de informacao reavaliadas em h=1 e h=5. Elas so tinham sido
    testadas em h=21, onde nem o baseline funciona."""
    from backtest import ablation
    from credibility import ablation as cred

    testes = {
        "noticia_bruta": lambda h: ablation.load_and_run_purged_ablation(
            horizon=h, n_splits=5, embargo_days=5, source="b3", news_smooth_window=None),
        "noticia_21d": lambda h: ablation.load_and_run_purged_ablation(
            horizon=h, n_splits=5, embargo_days=5, source="b3", news_smooth_window=21),
        "risco_fiscal": lambda h: ablation.load_and_run_fiscal_risk_ablation(
            horizon=h, n_splits=5, embargo_days=5, source="b3", use_surprise=True),
        "credibilidade": lambda h: cred.load_and_run_credibility_ablation(
            horizon=h, n_splits=5, embargo_days=5, source="b3"),
    }
    out = {}
    for nome, fn in testes.items():
        out[nome] = {}
        for h in (1, 5):
            try:
                r = fn(h)
                outra = [k for k in r if k.endswith("_pooled") and k != "baseline_pooled"][0]
                pf = r["per_fold"]
                chave = [k for k in pf if k != "baseline"][0]
                melhora = sum(
                    1 for x, y in zip(pf[chave], pf["baseline"])
                    if x["r2_oos"] > y["r2_oos"]
                )
                out[nome][str(h)] = {
                    "baseline": r["baseline_pooled"]["r2_oos"],
                    "com_camada": r[outra]["r2_oos"],
                    "delta": r[outra]["r2_oos"] - r["baseline_pooled"]["r2_oos"],
                    "folds": melhora, "n_folds": len(pf["baseline"]),
                }
            except Exception as e:  # dado auxiliar ausente nao derruba o relatorio
                out[nome][str(h)] = {"erro": f"{type(e).__name__}: {str(e)[:80]}"}
    return out


def _backtest_completo(precos: pd.Series, var: pd.Series, n_trials: int = 35) -> dict:
    """Backtest com IV REAL, com o conjunto de metricas que uma secao de
    desempenho precisa ter: trades, Sharpe, retorno anualizado, drawdown
    maximo e Deflated Sharpe.

    `n_trials` recebe o numero REAL de configuracoes testadas no projeto --
    nao apenas as variantes de backtest. E a escolha conservadora: o DSR
    desconta do Sharpe observado aquilo que se obteria por acaso testando
    muitas configuracoes, e subdeclarar esse numero inflaria o resultado.

    Roda tambem faixas de banda morta diferentes. O objetivo NAO e escolher a
    melhor -- isso seria selecao -- e sim mostrar se o veredito e estavel ou
    se depende de um ajuste especifico.
    """
    from backtest.engine import generate_oos_rv_forecast, run_backtest
    from backtest.metrics import pnl_performance_summary
    from backtest.walk_forward import purged_walk_forward_splits_by_step
    from vol.forecast import forward_target_from_variance, har_features_from_variance

    try:
        from data.b3_options import load_option_trades
        from vol.iv_trades import daily_atm_iv

        iv_real = daily_atm_iv(load_option_trades())
    except (ImportError, FileNotFoundError):
        return {"disponivel": False}
    if iv_real.empty:
        return {"disponivel": False}

    idx = pd.DatetimeIndex([d.date() for d in precos.index])
    p = pd.Series(precos.to_numpy(), index=idx)
    v = pd.Series(var.to_numpy(), index=idx)
    ds = har_features_from_variance(v)
    ds["target"] = forward_target_from_variance(v, 21)
    ds = ds.dropna()
    folds = purged_walk_forward_splits_by_step(
        ds, min_train_size=252, step_size=21, horizon=21, embargo_days=5
    )
    fc = generate_oos_rv_forecast(ds, BASELINE_FEATURES, 21, 5, 5, log_target=True, folds=folds)

    trades_por_ano = 252 / 21

    # SEM BASE DE CAPITAL, e a razao e substantiva, nao preferencia.
    # Duas tentativas anteriores falharam: dividir pelo premio de cada trade e
    # dividir pelo maior premio comprometido. Ambas produziram retornos abaixo
    # de -100%, drawdown de -186% e retorno anualizado NaN. A causa e que a
    # estrategia opera majoritariamente VENDIDA, e a perda de uma venda de
    # straddle nao tem teto -- um unico trade perde varias vezes o premio.
    # O capital realmente exigido e a MARGEM, que depende de regras da camara e
    # nao esta nos dados. Declarar uma base arbitraria fabricaria justamente o
    # numero mais visivel da secao.
    #
    # Reportamos entao o que sobrevive: Sharpe (invariante a escala, portanto
    # valido sem base de capital), PSR, Deflated Sharpe e drawdown em unidades
    # ABSOLUTAS de P&L. Ver backtest.metrics.pnl_performance_summary.
    todos = [
        run_backtest(p, fc, iv_real, horizon=21, band_pct=b, spread_pct=0.05)
        for b in (0.5, 1.0, 2.0)
    ]
    bandas: dict = {}
    retornos: dict[str, np.ndarray] = {}
    for banda, tr in zip((0.5, 1.0, 2.0), todos):
        if tr.empty:
            bandas[str(banda)] = {"n_trades": 0}
            continue
        ret = tr["pnl_net"].to_numpy()
        retornos[str(banda)] = ret
        resumo = pnl_performance_summary(ret, n_trials=n_trials, trades_per_year=trades_por_ano)
        resumo.update({
            "pnl_bruto": float(tr["pnl_gross"].sum()),
            "long": int((tr["signal"] == 1).sum()),
            "short": int((tr["signal"] == -1).sum()),
            "inicio": str(tr["entry_date"].min().date()),
            "fim": str(tr["exit_date"].max().date()),
        })
        bandas[str(banda)] = resumo

    # O DSR padrao usa dispersao 1,0 entre trials, que e deliberadamente
    # punitiva. Tendo rodado varias bandas, da para usar a dispersao OBSERVADA
    # dos Sharpes -- mais fiel, e sem afrouxar nada arbitrariamente.
    sharpes = [b["sharpe"] for b in bandas.values() if isinstance(b, dict) and "sharpe" in b]
    if len(sharpes) > 1 and "1.0" in retornos:
        std_obs = float(np.std(sharpes, ddof=1))
        if std_obs > 0:
            refinado = pnl_performance_summary(
                retornos["1.0"], n_trials=n_trials,
                trades_per_year=trades_por_ano, sr_trials_std=std_obs,
            )
            bandas["1.0"]["deflated_sharpe_std_observado"] = refinado["deflated_sharpe"]
            bandas["sr_trials_std_observado"] = std_obs

    if "1.0" in retornos:
        bandas["curva_banda_1"] = [float(x) for x in np.cumsum(retornos["1.0"])]
    bandas["disponivel"] = True
    bandas["n_trials"] = n_trials
    return bandas


def _backtest_iv_real(precos: pd.Series, var: pd.Series) -> dict:
    """Backtest com IV REAL (primeira vez possivel) contra a proxy, e a
    decomposicao que mostra a perda ANTES dos custos."""
    try:
        from data.b3_options import load_option_trades
        from vol.iv_trades import daily_atm_iv
    except ImportError:
        return {"disponivel": False}
    from backtest.engine import generate_oos_rv_forecast, proxy_iv, run_backtest
    from backtest.metrics import sharpe_ratio
    from backtest.walk_forward import purged_walk_forward_splits_by_step
    from vol.forecast import forward_target_from_variance, har_features_from_variance
    from vol.realized import TRADING_DAYS_PER_YEAR

    try:
        iv_real = daily_atm_iv(load_option_trades())
    except FileNotFoundError:
        return {"disponivel": False}
    if iv_real.empty:
        return {"disponivel": False}

    idx = pd.DatetimeIndex([d.date() for d in precos.index])
    p = pd.Series(precos.to_numpy(), index=idx)
    v = pd.Series(var.to_numpy(), index=idx)
    ds = har_features_from_variance(v)
    ds["target"] = forward_target_from_variance(v, 21)
    ds = ds.dropna()
    folds = purged_walk_forward_splits_by_step(
        ds, min_train_size=252, step_size=21, horizon=21, embargo_days=5
    )
    fc = generate_oos_rv_forecast(ds, BASELINE_FEATURES, 21, 5, 5, log_target=True, folds=folds)
    trail = np.sqrt(v.rolling(21).mean() * TRADING_DAYS_PER_YEAR) * 100

    out = {"disponivel": True, "sobreposicao_dias": int(len(fc.index.intersection(iv_real.index)))}
    for rot, ivs in (("iv_real", iv_real),
                     ("iv_proxy", proxy_iv(trail, 1.29).reindex(iv_real.index).dropna())):
        tr = run_backtest(p, fc, ivs, horizon=21, band_pct=1.0, spread_pct=0.05)
        if tr.empty:
            out[rot] = {"trades": 0}
            continue
        bruto = run_backtest(p, fc, ivs, horizon=21, band_pct=1.0, spread_pct=0.0)
        out[rot] = {
            "trades": int(len(tr)),
            "long": int((tr["signal"] == 1).sum()),
            "short": int((tr["signal"] == -1).sum()),
            "pnl_liquido": float(tr["pnl_net"].sum()),
            "pnl_bruto_sem_custo": float(bruto["pnl_gross"].sum()),
            "acerto": float((tr["pnl_net"] > 0).mean()),
            "sharpe": float(
                sharpe_ratio(tr["pnl_net"] / 1000, annualization_factor=np.sqrt(252 / 21))
            ),
        }
    return out


def _iv_propria() -> dict:
    """IV historica reconstruida de negocios reais de opcao (Black-76 invertido)
    e o premio de risco de variancia medido a partir dela."""
    try:
        from vol.iv_trades import load_and_evaluate

        r = load_and_evaluate()
    except FileNotFoundError as e:
        return {"disponivel": False, "motivo": str(e)}
    return {
        "disponivel": True,
        "n_negocios": r["n_negocios"],
        "n_pregoes_com_iv": r["n_pregoes_com_iv"],
        "periodo": r["periodo"],
        "avaliacao": r["avaliacao"],
        "premio": r["premio"],
    }


if __name__ == "__main__":
    main()
