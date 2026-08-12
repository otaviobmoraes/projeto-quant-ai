"""Roda TODAS as ablacoes nas duas fontes de preco e salva a comparacao.

Separado de gerar_dados.py porque e a parte cara (ajusta modelo em cada fold,
para cada camada, em duas fontes). Salva relatorio/ablacoes_por_fonte.json,
consumido por gerar_relatorio.py.

Por que rodar nas DUAS fontes: os vereditos originais do projeto foram
obtidos sobre o yfinance, cujo `close` esta defasado 1 dia (ver
data/b3_futures.py). Rodar tambem na fonte correta mostra quais vereditos
eram propriedade do dado, e nao do fenomeno.

Uso:
    python -m relatorio.gerar_ablacoes
"""

from __future__ import annotations

import json
from pathlib import Path

from backtest import ablation
from credibility import ablation as cred_ablation

OUT_PATH = Path(__file__).resolve().parent / "ablacoes_por_fonte.json"

FONTES = ("yfinance", "b3")

# (nome, funcao, chave do resultado, kwargs extras)
CAMADAS = [
    ("noticia_tom_gdelt", ablation.load_and_run_purged_ablation, "com_noticia", {}),
    ("credibilidade", cred_ablation.load_and_run_credibility_ablation, "com_credibilidade", {}),
    ("risco_fiscal_nivel", ablation.load_and_run_fiscal_risk_ablation, "com_noticia", {}),
    ("risco_fiscal_surpresa", ablation.load_and_run_fiscal_risk_ablation, "com_noticia",
     {"use_surprise": True, "surprise_window": 21}),
    ("risco_fiscal_v2_finbert", ablation.load_and_run_fiscal_risk_ablation_v2,
     "com_risco_fiscal_v2", {}),
]


def rodar(fn, chave, fonte, **kwargs) -> dict:
    r = fn(horizon=21, n_splits=5, embargo_days=5, source=fonte, **kwargs)
    baseline = r["baseline_pooled"]["r2_oos"]
    camada = r[f"{chave}_pooled"]["r2_oos"]
    por_fold = r["per_fold"]
    melhora = sum(
        1 for b, c in zip(por_fold["baseline"], por_fold[chave]) if c["r2_oos"] > b["r2_oos"]
    )
    return {
        "baseline": baseline,
        "camada": camada,
        "delta": camada - baseline,
        "melhora_em": melhora,
        "n_folds": len(por_fold["baseline"]),
        "n_obs": r["baseline_pooled"]["n_obs"],
    }


def main():
    res = {}
    for nome, fn, chave, kwargs in CAMADAS:
        res[nome] = {}
        for fonte in FONTES:
            try:
                res[nome][fonte] = rodar(fn, chave, fonte, **kwargs)
            except Exception as exc:  # dado faltando nao deve derrubar a bateria inteira
                res[nome][fonte] = {"erro": f"{type(exc).__name__}: {exc}"}
        print(f"  {nome} ok", flush=True)

    OUT_PATH.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'camada':26s} {'fonte':9s} {'baseline':>9s} {'c/ camada':>10s} {'delta':>8s} {'folds+':>7s}")
    for nome, linha in res.items():
        for fonte, d in linha.items():
            if "erro" in d:
                print(f"{nome:26s} {fonte:9s}  ERRO: {d['erro'][:50]}")
            else:
                print(f"{nome:26s} {fonte:9s} {d['baseline']:+9.3f} {d['camada']:+10.3f} "
                      f"{d['delta']:+8.3f} {d['melhora_em']}/{d['n_folds']}")
    print(f"\nsalvo: {OUT_PATH}")


if __name__ == "__main__":
    main()
