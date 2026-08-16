"""Testes do diagrama de blocos do pipeline de ML (report/ml_diagram.py).

Diagrama nao tem resultado numerico a conferir, entao o que se testa e o que
de fato quebra na pratica: que ele constroi sem erro (uma propriedade invalida
de matplotlib ja derrubou a primeira versao), que grava o arquivo, e que os
blocos rejeitados continuam marcados como rejeitados -- se alguem mudar isso
sem querer, o diagrama passa a contar uma historia falsa sobre o projeto.
"""

import matplotlib
import pytest

matplotlib.use("Agg")

from report import ml_diagram  # noqa: E402


@pytest.fixture(scope="module")
def fig():
    f = ml_diagram.build_ml_diagram()
    yield f
    matplotlib.pyplot.close(f)


def test_constroi_sem_erro(fig):
    assert fig is not None
    assert len(fig.axes) == 1


def _textos(fig) -> list[str]:
    ax = fig.axes[0]
    return [t.get_text() for t in ax.texts] + [t.get_text() for t in fig.texts]


def test_cobre_as_cinco_etapas(fig):
    for titulo, _ in ml_diagram.COLUNAS:
        espacado = " ".join(titulo)
        assert espacado in _textos(fig), f"coluna ausente: {titulo}"


def test_nomeia_as_entradas_de_dado(fig):
    txt = " | ".join(_textos(fig))
    for esperado in ("Futuro de dólar B3", "Negócios de opção B3", "Spread bid-ask"):
        assert esperado in txt


def test_nomeia_os_modelos_avaliados(fig):
    txt = " | ".join(_textos(fig))
    for modelo in ("Persistência", "HAR-RV", "HAR livre de escala", "GARCH", "XGBoost", "painel"):
        assert modelo in txt


def test_marca_os_guardrails_de_validacao(fig):
    txt = " | ".join(_textos(fig))
    for guardrail in ("PURGADO", "Três portões", "subperíodo", "Deflated Sharpe"):
        assert guardrail in txt


def test_camadas_rejeitadas_desenhadas_como_rejeitadas(fig):
    """O que impede o diagrama de mentir: as camadas descartadas têm de
    continuar em traço pontilhado, não sólido."""
    ax = fig.axes[0]
    pontilhados = [
        p for p in ax.patches
        if p.get_linestyle() not in ("solid", "-") and p.get_edgecolor() is not None
    ]
    # GDELT, Focus/COPOM, VIX+DXY, painel EM, GARCH, XGBoost, modelo global,
    # mais o quadradinho da legenda
    assert len(pontilhados) >= 8


def test_caminho_oficial_desenhado_solido(fig):
    ax = fig.axes[0]
    solidos = [p for p in ax.patches if p.get_linestyle() in ("solid", "-")]
    assert len(solidos) >= 15


def test_grava_arquivo(tmp_path):
    destino = tmp_path / "diagrama_ml.png"
    caminho = ml_diagram.save_ml_diagram(destino, dpi=72)
    assert caminho == destino
    assert destino.exists()
    assert destino.stat().st_size > 20_000


def test_cria_diretorio_se_faltar(tmp_path):
    destino = tmp_path / "sub" / "dir" / "d.png"
    ml_diagram.save_ml_diagram(destino, dpi=72)
    assert destino.exists()
