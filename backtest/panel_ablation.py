"""Ablacao: modelo LOCAL (so USD/BRL) vs modelo GLOBAL em painel de moedas
emergentes -- ver vol/panel.py para o desenho e as referencias.

*** CRITERIO DE SUCESSO, DECLARADO ANTES DE RODAR (os tres portoes do
CLAUDE.md). O modelo global so e considerado melhora se as TRES condicoes
valerem contra o baseline local correspondente:
    1. delta R2 pooled > 0;
    2. melhora em >= 4 dos 5 folds purgados (esquema oficial do projeto);
    3. Diebold-Mariano p < 0.05 em janelas INDEPENDENTES.
Abaixo disso o veredito e "nao estabelecido", qualquer que seja a magnitude.
O criterio NAO sera afrouxado depois de ver o resultado. ***

RESULTADO PRINCIPAL E POR SUBPERIODO, nao pooled: esta linha de trabalho ja
registrou CINCO casos em que uma estatistica agregada produziu achado que a
analise por subperiodo matou. O pooled entra como contexto.

BASELINE CORRETO. A comparacao que importa e global vs local com a MESMA
especificacao (alvo em razao), porque so assim a unica variavel e a ORIGEM
DOS DADOS de estimacao. O HAR em nivel entra como terceira coluna por ser o
baseline oficial do projeto, mas comparar global (razao) contra local (nivel)
confundiria dois efeitos.
"""

from __future__ import annotations

import pandas as pd

from backtest.metrics import diebold_mariano
from backtest.walk_forward import purged_walk_forward_splits
from vol.forecast import (
    BASELINE_FEATURES,
    build_scale_free_dataset,
    fit_har,
    fit_har_scale_free,
    forecast_metrics,
    forward_target_from_variance,
    har_features_from_variance,
    predict,
    predict_scale_free,
)
from vol.panel import build_panel, fit_global_model, predict_with_global_model, purge_panel_by_date

_VEREDITO = {True: "PASSOU nos tres portoes", False: "NAO ESTABELECIDO"}

SUBPERIODOS: dict[str, tuple[str | None, str | None]] = {
    "2019-2020": ("2019-01-01", "2020-12-31"),
    "2021-2022": ("2021-01-01", "2022-12-31"),
    "2023-2026": ("2023-01-01", "2026-12-31"),
}


def build_local_dataset(daily_variance: pd.Series, horizon: int) -> pd.DataFrame:
    """Dataset local do BRL (medido no futuro da B3) em formato livre de
    escala, com o alvo em NIVEL preservado para a avaliacao."""
    df = har_features_from_variance(daily_variance)
    df["target"] = forward_target_from_variance(daily_variance, horizon)
    return build_scale_free_dataset(df.dropna())


def run_panel_ablation(
    daily_variance: pd.Series,
    em_fx: pd.DataFrame,
    horizon: int = 21,
    n_splits: int = 5,
    embargo_days: int = 5,
    include_brl_in_panel: bool = True,
) -> dict:
    """Roda a ablacao completa nos mesmos folds purgados.

    `include_brl_in_panel=True` (primario): o painel de treino recebe TAMBEM a
    historia do BRL medida no futuro da B3 -- e o desenho de Bollerslev et al.,
    em que o ativo de interesse participa do pool. `False` isola a
    TRANSFERENCIA PURA: coeficientes estimados sem ver o BRL nenhuma vez.

    Todas as previsoes saem na escala de NIVEL e sao avaliadas contra o MESMO
    alvo em nivel, no MESMO conjunto de linhas de teste (armadilha 4).
    """
    local = build_local_dataset(daily_variance, horizon)
    painel_ext = build_panel(em_fx, horizon=horizon)

    folds = purged_walk_forward_splits(
        local, n_splits=n_splits, horizon=horizon, embargo_days=embargo_days
    )

    # bloco do BRL no mesmo formato do painel, para poder entrar no pool
    brl_bloco = local.reset_index(names="date")[
        ["date", "ratio_d", "ratio_w", "rv_trailing", "target_ratio", "target"]
    ].copy()
    brl_bloco["currency"] = "brl"

    preds: dict[str, list[pd.Series]] = {"local_nivel": [], "local_razao": [], "global": []}
    alvos: list[pd.Series] = []
    por_fold: list[dict] = []

    for i, (train, test) in enumerate(folds, 1):
        alvos.append(test["target"])

        # --- baselines locais ---
        m_nivel = fit_har(train, BASELINE_FEATURES, log_target=True)
        p_nivel = predict(m_nivel, test, BASELINE_FEATURES, log_target=True)

        m_razao = fit_har_scale_free(train)
        p_razao = predict_scale_free(m_razao, test)

        # --- modelo global ---
        # purga por DATA, cortando TODAS as moedas (ver vol/panel.py)
        train_end = train.index.max()
        pool = painel_ext
        if include_brl_in_panel:
            pool = pd.concat([painel_ext, brl_bloco], ignore_index=True)
        pool_train = purge_panel_by_date(pool, train_end, horizon, embargo_days)

        m_global = fit_global_model(pool_train)
        p_global = predict_with_global_model(m_global, test)

        preds["local_nivel"].append(p_nivel)
        preds["local_razao"].append(p_razao)
        preds["global"].append(p_global)

        por_fold.append(
            {
                "fold": i,
                "n_teste": len(test),
                "n_treino_local": len(train),
                "n_treino_painel": len(pool_train),
                "inicio": str(test.index.min().date()),
                "fim": str(test.index.max().date()),
                **{
                    f"r2_{k}": forecast_metrics(v[-1], test["target"])["r2_oos"]
                    for k, v in preds.items()
                },
            }
        )

    alvo = pd.concat(alvos).sort_index()
    series = {k: pd.concat(v).sort_index() for k, v in preds.items()}

    # --- portao 1: R2 pooled --------------------------------------------
    pooled = {k: forecast_metrics(v, alvo) for k, v in series.items()}

    # --- portao 2: contagem de folds ------------------------------------
    folds_melhores = sum(1 for f in por_fold if f["r2_global"] > f["r2_local_razao"])

    # --- portao 3: Diebold-Mariano em janelas independentes -------------
    dm = {
        "vs_local_razao": diebold_mariano(
            alvo, series["global"], series["local_razao"], horizon=horizon, independent_only=True
        ),
        "vs_local_nivel": diebold_mariano(
            alvo, series["global"], series["local_nivel"], horizon=horizon, independent_only=True
        ),
    }

    # --- resultado por subperiodo (o principal) -------------------------
    por_subperiodo = []
    for nome, (i0, i1) in SUBPERIODOS.items():
        sl = alvo.index[(alvo.index >= i0) & (alvo.index <= i1)]
        if len(sl) < 30:
            continue
        linha = {"periodo": nome, "n": len(sl)}
        for k, v in series.items():
            m = forecast_metrics(v.loc[sl], alvo.loc[sl])
            linha[f"r2_{k}"] = m["r2_oos"]
            linha[f"vies_{k}"] = m["vies_pct"]
        por_subperiodo.append(linha)

    aprovou = (
        pooled["global"]["r2_oos"] > pooled["local_razao"]["r2_oos"]
        and folds_melhores >= 4
        and dm["vs_local_razao"]["p_value"] < 0.05
    )

    return {
        "horizonte": horizon,
        "painel_inclui_brl": include_brl_in_panel,
        "moedas": sorted(em_fx["currency"].unique().tolist()),
        "pooled": pooled,
        "por_fold": por_fold,
        "por_subperiodo": por_subperiodo,
        "folds_melhores": folds_melhores,
        "n_folds": len(folds),
        "diebold_mariano": dm,
        "coeficientes_globais": dict(m_global.params),
        "coeficientes_locais_ultimo_fold": dict(m_razao.params),
        "passou_tres_portoes": bool(aprovou),
    }


def load_and_run_panel_ablation(
    horizon: int = 21,
    n_splits: int = 5,
    embargo_days: int = 5,
    estimator: str = "parkinson",
    include_brl_in_panel: bool = True,
) -> dict:
    """Le o futuro da B3 e o painel de moedas ja coletados e roda a ablacao."""
    from data.em_fx import load_em_fx
    from vol.realized import load_b3_variance

    _, var = load_b3_variance(estimator=estimator)
    var = pd.Series(var.to_numpy(), index=pd.DatetimeIndex([d.date() for d in var.index]))

    em = load_em_fx()
    em["date"] = pd.to_datetime(
        pd.DatetimeIndex([d.date() for d in pd.to_datetime(em["date"])])
    )

    return run_panel_ablation(
        var,
        em,
        horizon=horizon,
        n_splits=n_splits,
        embargo_days=embargo_days,
        include_brl_in_panel=include_brl_in_panel,
    )


def format_verdict(res: dict) -> str:
    """Relatorio textual da ablacao, com os tres portoes explicitos."""
    p = res["pooled"]
    dm = res["diebold_mariano"]["vs_local_razao"]
    linhas = [
        f"Painel: {', '.join(res['moedas'])}"
        f"{' + brl' if res['painel_inclui_brl'] else '  (SEM brl -- transferencia pura)'}",
        f"Horizonte h={res['horizonte']}   folds={res['n_folds']}",
        "",
        f"{'modelo':<22}{'R2 pooled':>12}{'RMSE':>10}{'vies %':>10}",
    ]
    rotulos = {
        "local_nivel": "local HAR (nivel)",
        "local_razao": "local HAR (razao)",
        "global": "GLOBAL (painel)",
    }
    for k, rot in rotulos.items():
        linhas.append(
            f"{rot:<22}{p[k]['r2_oos']:>12.4f}{p[k]['rmse']:>10.4f}{p[k]['vies_pct']:>10.2f}"
        )

    delta = p["global"]["r2_oos"] - p["local_razao"]["r2_oos"]
    linhas += [
        "",
        "TRES PORTOES (contra o local com a MESMA especificacao):",
        f"  1. delta R2 pooled > 0        : {delta:+.4f}   "
        f"{'OK' if delta > 0 else 'REPROVA'}",
        f"  2. >= 4 de {res['n_folds']} folds melhorando : {res['folds_melhores']} de "
        f"{res['n_folds']}   {'OK' if res['folds_melhores'] >= 4 else 'REPROVA'}",
        f"  3. DM p < 0.05 (independente) : p={dm['p_value']:.4f} (t={dm['t_stat']:+.3f}, "
        f"n={dm['n']})   {'OK' if dm['p_value'] < 0.05 else 'REPROVA'}",
        "",
        f"VEREDITO: {_VEREDITO[res['passou_tres_portoes']]}",
    ]
    return "\n".join(linhas)