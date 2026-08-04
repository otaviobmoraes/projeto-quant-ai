"""Resumo textual (markdown) dos resultados de ablacao para o pre-relatorio.

CLAUDE.md: "Registrar o numero de configuracoes testadas [...] e usar
Deflated Sharpe Ratio." O resumo deixa explicitas quais e quantas
configuracoes foram avaliadas, pra qualquer leitor auditar se houve
p-hacking de configuracao -- e reporta o DSR quando ele e fornecido.
"""

from __future__ import annotations


def format_metrics(metrics: dict) -> str:
    return f"RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  R2_oos={metrics['r2_oos']:.3f}"


def ablation_summary_md(
    result: dict, configs_tested: list[str], dsr: float | None = None
) -> str:
    """Monta um resumo em markdown do resultado da ablacao (baseline vs
    com_noticia): metricas, numero de folds, lista de configuracoes testadas
    e (se fornecido) o Deflated Sharpe Ratio da configuracao escolhida.
    """
    n_folds = result.get("n_splits", len(result.get("per_fold", {}).get("baseline", [])))
    lines = [
        "## Ablacao: RV baseline vs RV com noticia",
        "",
        f"- **Baseline (HAR-RV, R2 pooled)**: {format_metrics(result['baseline_pooled'])}",
        f"- **Com noticia (R2 pooled)**: {format_metrics(result['com_noticia_pooled'])}",
        f"- Folds (walk-forward purgado): {n_folds}",
    ]
    if dsr is not None:
        lines.append(f"- **Deflated Sharpe Ratio** (config. final, ajustado por n. de testes): {dsr:.3f}")
    lines += [
        "",
        f"### Configuracoes testadas ({len(configs_tested)})",
        "",
    ]
    lines += [f"- {c}" for c in configs_tested]
    return "\n".join(lines)
