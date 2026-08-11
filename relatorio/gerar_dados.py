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
from data.b3_futures import PROCESSED_PATH as B3_PATH
from data.fx_spot import PROCESSED_PATH as FX_PATH
from credibility.credibility import PROCESSED_PATH as CREDIBILITY_PATH
from data.ptax import PROCESSED_PATH as PTAX_PATH
from vol.forecast import BASELINE_FEATURES, build_dataset, persistence_forecast
from vol.realized import (
    load_b3_futures_prices_and_variance,
    load_parkinson_prices_and_variance,
    forward_realized_skewness,
    parkinson_daily_variance,
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
    close_b3, var_b3 = load_b3_futures_prices_and_variance()
    fut = pd.read_parquet(B3_PATH).set_index("date").sort_index()
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

    (OUT_DIR / "resultados.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nresultados.json salvo ({len(res)} blocos)")


if __name__ == "__main__":
    main()
