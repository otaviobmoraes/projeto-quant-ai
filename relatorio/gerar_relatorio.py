# -*- coding: utf-8 -*-
"""Gera o relatorio final do VolBoy (.docx).

Le os numeros de relatorio/resultados.json (produzido por gerar_dados.py) --
NADA de numero digitado a mao aqui.

Uso:
    python -m relatorio.gerar_dados      # primeiro: recalcula tudo
    python -m relatorio.gerar_relatorio  # depois: monta o documento
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

BASE = Path(__file__).resolve().parent
FIG = BASE / "figuras"
OUT_PATH = BASE / "VolBoy_Relatorio_Final.docx"

R = json.loads((BASE / "resultados.json").read_text(encoding="utf-8"))

INK = RGBColor(0x0B, 0x0B, 0x0B)
SEC = RGBColor(0x3A, 0x3A, 0x38)
MUTED = RGBColor(0x6B, 0x6A, 0x66)
BLUE = RGBColor(0x1F, 0x5C, 0xA8)
GREEN = RGBColor(0x00, 0x63, 0x00)
RED = RGBColor(0xA8, 0x2A, 0x2A)

doc = Document()

# ---------------------------------------------------------------- estilos
_normal = doc.styles["Normal"]
_normal.font.name = "Calibri"
_normal.font.size = Pt(10.5)
_normal.font.color.rgb = INK
_normal.paragraph_format.space_after = Pt(6)
_normal.paragraph_format.line_spacing = 1.15

for lvl, size, color, before in [(1, 18, INK, 20), (2, 14, INK, 14), (3, 11.5, SEC, 10)]:
    st = doc.styles[f"Heading {lvl}"]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = color
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(6)

_s = doc.sections[0]
_s.left_margin = _s.right_margin = Cm(2.3)
_s.top_margin = _s.bottom_margin = Cm(2.0)


def h1(t): doc.add_heading(t, 1)
def h2(t): doc.add_heading(t, 2)
def h3(t): doc.add_heading(t, 3)


def p(text="", *, bold=False, italic=False, size=10.5, color=None, align=None, after=6, before=0):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(after)
    para.paragraph_format.space_before = Pt(before)
    if align is not None:
        para.alignment = align
    run = para.add_run(text)
    run.bold, run.italic = bold, italic
    run.font.size = Pt(size)
    run.font.color.rgb = color or INK
    return para


def rich(*parts, after=6, size=10.5):
    """Paragrafo com trechos formatados: (texto, {'bold':True}) ou so texto."""
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(after)
    for part in parts:
        text, fmt = part if isinstance(part, tuple) else (part, {})
        run = para.add_run(text)
        run.bold = fmt.get("bold", False)
        run.italic = fmt.get("italic", False)
        run.font.size = Pt(fmt.get("size", size))
        run.font.color.rgb = fmt.get("color", INK)
        if fmt.get("mono"):
            run.font.name = "Consolas"
    return para


def bullets(items, style="List Bullet", size=10.5):
    for item in items:
        para = doc.add_paragraph(style=style)
        para.paragraph_format.space_after = Pt(3)
        if isinstance(item, tuple):
            label, rest = item
            r1 = para.add_run(label); r1.bold = True; r1.font.size = Pt(size)
            r2 = para.add_run(rest); r2.font.size = Pt(size)
        else:
            para.add_run(item).font.size = Pt(size)


def formula(text, *, size=12, note=None):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(8)
    para.paragraph_format.space_after = Pt(4 if note else 10)
    run = para.add_run(text)
    run.italic = True
    run.font.name = "Cambria Math"
    run.font.size = Pt(size)
    if note:
        p(note, size=9, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), fill)
    tcPr.append(el)


def table(headers, rows, widths=None, header_fill="2A2A28", size=9.2, highlight=None):
    """`highlight`: set de indices de linha para destacar em negrito."""
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, htxt in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(htxt)
        r.bold = True; r.font.size = Pt(size); r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(c, header_fill)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(val))
            r.font.size = Pt(size)
            if highlight and ri in highlight:
                r.bold = True
                shade(cells[i], "FFF4E0")
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def image(name, width=6.3, caption=None):
    doc.add_picture(str(FIG / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        p(caption, italic=True, size=8.8, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, after=12)


def callout(titulo, texto, cor=RED):
    """Caixa de destaque para achados/limitacoes importantes."""
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    c = t.rows[0].cells[0]
    shade(c, "F7F3E8")
    c.text = ""
    para = c.paragraphs[0]
    r = para.add_run(titulo + "  ")
    r.bold = True; r.font.size = Pt(10); r.font.color.rgb = cor
    r2 = para.add_run(texto)
    r2.font.size = Pt(9.8)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def pagebreak():
    doc.add_page_break()


def fmt(x, casas=3, sinal=True):
    """Formata numero no padrao brasileiro (virgula decimal, sinal menos
    tipografico) -- o documento inteiro e em portugues."""
    txt = f"{x:+.{casas}f}" if sinal else f"{x:.{casas}f}"
    return txt.replace("-", "−").replace(".", ",")


def num(x, casas=3):
    """Idem, sem sinal explicito de positivo."""
    return fmt(x, casas, sinal=False)


# =====================================================================
# CAPA
# =====================================================================
p("", size=8)
cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap.paragraph_format.space_before = Pt(70)
r = cap.add_run("VolBoy")
r.bold = True; r.font.size = Pt(52); r.font.color.rgb = INK

sub = doc.add_paragraph(); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("Robô quantitativo de volatilidade em opções de dólar (USD/BRL)")
r.font.size = Pt(14); r.font.color.rgb = SEC

sub2 = doc.add_paragraph(); sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub2.paragraph_format.space_before = Pt(6)
r = sub2.add_run("Relatório Final  ·  Desafio Itaú Asset Quant AI 2026")
r.font.size = Pt(12); r.italic = True; r.font.color.rgb = MUTED

p("", size=14)
image("r2_por_horizonte.png", width=6.0)

tese = doc.add_paragraph(); tese.alignment = WD_ALIGN_PARAGRAPH.CENTER
tese.paragraph_format.space_before = Pt(24)
r = tese.add_run(
    "Negociamos o spread entre a volatilidade que o mercado precifica (IV)\n"
    "e a volatilidade que projetamos que vai se realizar (RV)."
)
r.font.size = Pt(11); r.italic = True; r.font.color.rgb = SEC

amostra = R["amostra"]
rodape = doc.add_paragraph(); rodape.alignment = WD_ALIGN_PARAGRAPH.CENTER
rodape.paragraph_format.space_before = Pt(30)
r = rodape.add_run(
    f"Amostra: {amostra['b3_pregoes']} pregões do futuro de dólar da B3 "
    f"({amostra['b3_inicio']} a {amostra['b3_fim']})  ·  240 testes automatizados  ·  "
    "22 configurações registradas"
)
r.font.size = Pt(9); r.font.color.rgb = MUTED

pagebreak()

# =====================================================================
# SUMARIO EXECUTIVO
# =====================================================================
h1("Sumário executivo")

p("O VolBoy é um robô de volatilidade em opções de dólar. A tese é simples de enunciar: o mercado "
  "de opções embute, no prêmio, uma expectativa de volatilidade futura (a volatilidade implícita, "
  "IV). Se conseguirmos projetar a volatilidade que de fato vai se realizar (RV) melhor que o "
  "mercado, compramos volatilidade quando ela está barata e vendemos quando está cara — capturando "
  "o spread, e não a direção do dólar.")

p("A aposta diferencial do projeto era que fluxo de notícias e um índice de credibilidade do Banco "
  "Central melhorariam essa projeção. Testamos essa hipótese com rigor, e o resultado principal "
  "deste relatório é um resultado negativo bem estabelecido — que consideramos mais valioso, e "
  "certamente mais honesto, do que um número de backtest favorável obtido sem escrutínio.")

h2("Os quatro achados que estruturam este relatório")

b3_21 = R["horizonte"]["b3"]["21"]
yf_21 = R["horizonte"]["yfinance"]["21"]
b3_1 = R["horizonte"]["b3"]["1"]

bullets([
    ("Nenhuma camada de informação melhorou a previsão. ",
     "Notícia (tom do GDELT), credibilidade do BC (Focus vs. meta), risco fiscal em três "
     "formulações — incluindo sentimento via FinBERT-PT-BR — e risco global exógeno (VIX, DXY). "
     "Foram 22 configurações registradas; nenhuma produziu ganho consistente."),
    ("Encontramos e corrigimos dois erros metodológicos nossos. ",
     "Um na métrica (R² calculado por fold e depois promediado, instável com folds pequenos) e "
     "outro na fonte de preço (as barras diárias do yfinance para BRL=X têm abertura ≈ fechamento, "
     "o que defasava em um dia toda feature derivada do fechamento). Ambos foram diagnosticados "
     "com evidência quantitativa e estão documentados neste relatório."),
    ("O único R² positivo não sobreviveu à fonte correta. ",
     f"Sobre o proxy de preço, a persistência atingia R² = {fmt(yf_21['persist_r2'])}. Ao migrar "
     f"para o futuro de dólar da B3 — o instrumento sobre o qual a opção é escrita —, o mesmo "
     f"método cai para {fmt(b3_21['persist_r2'])} e o HAR-RV para {fmt(b3_21['har_r2'])}."),
    ("O diagnóstico final não é 'faltou feature': é descasamento de horizonte. ",
     f"No instrumento correto existe previsibilidade real, ainda que modesta, a 1 dia "
     f"(HAR-RV R² = {fmt(b3_1['har_r2'])}), que decai monotonicamente até ficar francamente "
     f"negativa em 21 dias — o prazo de que a estratégia precisa. Nenhuma feature adicional "
     "resolve isso; é uma propriedade do processo de volatilidade neste período."),
])

callout(
    "Conclusão honesta.",
    "O VolBoy não demonstrou capacidade preditiva superior ao mercado no horizonte que a "
    "estratégia exige. A infraestrutura, o rigor metodológico e o diagnóstico do porquê estão "
    "sólidos e são reproduzíveis; a tese de que notícia melhora previsão de volatilidade de "
    "USD/BRL foi testada e não se sustentou nesta amostra.",
    cor=RED,
)

p("As seções seguintes percorrem a trajetória completa: o que motivou cada decisão, o que "
  "funcionou, o que falhou, e — com igual atenção — os erros que cometemos e como foram "
  "descobertos. A Seção 8 discute o que faríamos diferente e os próximos passos.", before=6)

pagebreak()

# =====================================================================
# 1. CONCEITO
# =====================================================================
h1("1. O conceito do VolBoy")

h2("1.1 Por que negociar volatilidade, e não direção")

p("Prever a direção do dólar é notoriamente difícil: a literatura de câmbio documenta há décadas "
  "que taxas de câmbio de economias abertas se comportam próximo a um passeio aleatório em "
  "horizontes curtos. Volatilidade, ao contrário, tem duas propriedades que a tornam mais "
  "tratável: ela é persistente (períodos turbulentos se agrupam) e é diretamente negociável via "
  "opções.")

p("O VolBoy explora isso operando o spread entre duas quantidades:")

table(
    ["Quantidade", "O que é", "De onde vem"],
    [
        ("IV — volatilidade implícita",
         "A volatilidade que o mercado precifica hoje, embutida no prêmio da opção",
         "Superfície de volatilidade publicada pela B3"),
        ("RV — volatilidade realizada",
         "A volatilidade que de fato se materializou no preço, medida ex-post",
         "Preços do futuro de dólar (B3)"),
    ],
    widths=[1.7, 2.9, 1.9],
)

rich("A regra de decisão é o spread entre a RV que projetamos e a IV cotada: ",
     ("se projetamos RV acima da IV, o mercado está subestimando a turbulência futura e compramos "
      "volatilidade; se projetamos abaixo, vendemos.", {"bold": True}))

h2("1.2 A estrutura escolhida: straddle ATM")

p("Compramos (ou vendemos) volatilidade via straddle no dinheiro — call e put de mesmo strike e "
  "vencimento. A escolha tem uma razão precisa: no dinheiro, o straddle tem delta próximo de zero, "
  "de modo que o resultado depende principalmente de quanto o preço se move, não da direção em que "
  "se move. É a expressão mais limpa de uma aposta em volatilidade pura.")

p("Optamos deliberadamente pela estrutura mais simples disponível. Estruturas mais elaboradas "
  "(strangle, condor, risk reversal) foram consideradas e adiadas: elas introduzem parâmetros "
  "adicionais de calibração que ampliariam o espaço de busca — e, como discutimos na Seção 4.4, "
  "controlar esse espaço foi uma preocupação central do projeto.")

h2("1.3 A hipótese diferencial: informação além do preço")

p("Um HAR-RV alimentado apenas pelo histórico de preços é um ponto de partida bem estabelecido, "
  "não um diferencial. A aposta do VolBoy era que informação exógena antecipasse mudanças de "
  "regime de volatilidade antes que elas aparecessem no próprio preço. Três camadas foram "
  "construídas e testadas:")

bullets([
    ("Fluxo de notícias (GDELT). ",
     "Tom médio diário da cobertura global sobre o Brasil e, depois, atenção e sentimento "
     "especificamente sobre risco fiscal."),
    ("Credibilidade do Banco Central. ",
     "Um índice θₜ derivado do Boletim Focus, ancorado em teoria dos jogos (Barro-Gordon) — "
     "detalhado na Seção 2.7."),
    ("Risco global exógeno. ",
     "VIX e índice dólar (DXY): a única camada testada que não deriva do próprio preço ou da "
     "imprensa sobre o Brasil."),
])

p("A Seção 5 percorre o que aconteceu com cada uma. Adiantando: nenhuma sobreviveu ao teste de "
  "ablação formal.", italic=True, color=SEC)

pagebreak()

# =====================================================================
# 2. ARQUITETURA E MODELAGEM
# =====================================================================
h1("2. Arquitetura e modelagem")

h2("2.1 Visão geral do pipeline")

p("O VolBoy é organizado em seis camadas, cada uma num pacote Python isolado. Os dados fluem em "
  "uma única direção — coleta → processamento → modelagem → estratégia → backtest → relatório — e "
  "nenhuma camada superior importa de uma inferior. Essa disciplina existe por um motivo prático: "
  "permite testar a matemática sem tocar em disco ou rede, o que sustenta os 240 testes "
  "automatizados do projeto.")

image("diagrama_pipeline.png", width=6.4,
      caption="Figura 1 — Arquitetura do VolBoy. Cores indicam a camada; setas indicam fluxo de dados.")

h2("2.2 Fontes de dados")

p("Todas as fontes são públicas e gratuitas. Uma decisão de projeto foi não depender de Bloomberg: "
  "além da restrição de acesso, scraping violaria os termos de uso.")

table(
    ["Dado", "Fonte", "Papel no modelo"],
    [
        ("Futuro de dólar (DOL) — OHLC e preço de ajuste", "B3, arquivo BVBG-086", "Fonte oficial de preço; base da RV (adotada ao final)"),
        ("Câmbio spot USD/BRL", "yfinance (BRL=X)", "Proxy inicial; substituída após diagnóstico (Seção 5.4)"),
        ("PTAX", "API SGS do Banco Central", "Taxa oficial; usada como árbitro de validação"),
        ("Superfície de volatilidade", "B3 (preços referenciais)", "Única fonte de IV disponível"),
        ("Notícias globais", "GDELT DOC 2.0 API", "Tom, atenção e manchetes sobre risco fiscal"),
        ("Expectativas de inflação", "Boletim Focus / BCB (API Olinda)", "Índice de credibilidade θₜ"),
        ("Atividade econômica", "IBC-Br (SGS)", "Hiato do produto (filtro HP expansivo)"),
        ("Risco global", "yfinance (^VIX, DX-Y.NYB)", "Camada exógena de aversão a risco"),
    ],
    widths=[2.2, 1.9, 2.4],
)

p("Cada coletor é idempotente e cacheado: a resposta bruta é gravada em disco e a chamada de rede "
  "só se repete se o arquivo não existir. Isso torna o pipeline reproduzível e evita bater "
  "repetidamente em APIs públicas — uma preocupação que se mostrou concreta quando o GDELT passou "
  "a nos bloquear por limite de taxa (Seção 5.3).")

h2("2.3 Volatilidade realizada: como medimos o alvo")

p("RV é a medida ex-post de quanto o preço efetivamente oscilou. Sem dados intradiários de alta "
  "frequência, ela precisa ser estimada a partir de preços diários. O VolBoy usa dois estimadores.")

h3("Estimador ingênuo: retorno de fechamento ao quadrado")

formula("rₜ = ln(Pₜ / Pₜ₋₁)     e     Var_d(t) ≈ rₜ²")

p("É a simplificação padrão do HAR-RV quando só há preço de fechamento, mas é extremamente "
  "ruidosa: um único retorno ao quadrado é um estimador de variância com apenas um grau de "
  "liberdade, e sua própria variância amostral é enorme.")

h3("Estimador adotado: Parkinson (1980)")

p("Usa o intervalo intradiário (máxima e mínima do pregão), aproveitando o caminho percorrido pelo "
  "preço e não apenas o ponto final:")

formula("Var_Parkinson(t) = ln(Highₜ / Lowₜ)² / (4 · ln 2)",
        note="O fator 4·ln2 vem da esperança do range de um movimento browniano sem drift.")

p("A eficiência teórica é cerca de cinco vezes maior que a do estimador de fechamento. No "
  "diagnóstico empírico do projeto, o Parkinson produziu RMSE menor em todos os folds testados — "
  "por isso foi promovido a estimador oficial. Uma propriedade adicional se revelou decisiva mais "
  "tarde: por depender apenas de máxima e mínima do mesmo pregão, ele é imune tanto à rolagem de "
  "contratos futuros quanto ao erro de alinhamento descrito na Seção 5.4.")

h3("Anualização")

formula("RV_anualizada = √( média(Var_d) × 252 ) × 100",
        note="252 dias úteis por ano; resultado em pontos percentuais ao ano.")

h2("2.4 Volatilidade implícita e apreçamento: Black-76")

p("Opção de dólar na B3 é opção sobre o futuro, não sobre o dólar à vista. Isso obriga o uso de "
  "Black-76 (Black, 1976) em lugar do Black-Scholes clássico: o subjacente é o preço futuro F, que "
  "já incorpora o carrego de juros, de modo que a fórmula não carrega um termo separado de "
  "dividendo ou custo de carregamento.")

formula("d₁ = [ ln(F/K) + (σ²/2)·T ] / (σ·√T)          d₂ = d₁ − σ·√T")
formula("Call = e^(−rT) · [ F·N(d₁) − K·N(d₂) ]")
formula("Put = e^(−rT) · [ K·N(−d₂) − F·N(−d₁) ]",
        note="N(·) é a distribuição normal padrão acumulada; F = preço futuro, K = strike, T = prazo em anos, σ = volatilidade, r = taxa livre de risco.")

p("No VolBoy o Black-76 cumpre dois papéis: calcular o prêmio pago ou recebido em cada operação "
  "simulada (sem o qual não há P&L) e fornecer o vega usado no dimensionamento de posição.")

h2("2.5 Vega e dimensionamento de posição")

formula("Vega = e^(−rT) · F · φ(d₁) · √T",
        note="φ(·) é a densidade normal padrão. No Black-76, call e put têm vega idêntico — logo o vega do straddle é o dobro do de uma perna.")

p("Em vez de operar um número fixo de contratos, o VolBoy dimensiona a posição para entregar uma "
  "exposição-alvo constante a vega:")

formula("n_contratos = vega_alvo / vega_por_contrato")

p("A razão é de gestão de risco: o vega por contrato varia com o nível de volatilidade do mercado. "
  "Operar quantidade fixa faria o risco assumido oscilar involuntariamente conforme o regime — "
  "justamente a variável que a estratégia pretende controlar.")

h2("2.6 A regra de decisão")

formula("spread = RV_prevista − IV          |  spread > banda  → compra volatilidade\n"
        "                                    |  spread < −banda → vende volatilidade\n"
        "                                    |  caso contrário  → não opera", size=10.5)

p("A banda morta não é um detalhe de implementação: opções de dólar têm spread de compra e venda "
  "largo, e operar diferenças pequenas destrói o resultado em custos. A banda exige que a "
  "discrepância entre nossa projeção e o mercado seja grande o suficiente para pagar o custo de "
  "entrada e saída. Seu tamanho é o principal hiperparâmetro da estratégia — e, por isso, é a "
  "família de configurações usada no cálculo do Deflated Sharpe Ratio (Seção 4.4).")

pagebreak()

h2("2.7 Os modelos de previsão de RV")

h3("HAR-RV (Corsi, 2009) — o modelo central")

p("A Hipótese de Mercado Heterogêneo sustenta que participantes operam em horizontes distintos: "
  "quem gira posição no dia reage à volatilidade de ontem; gestores olham a semana; alocadores "
  "olham o mês. O HAR-RV formaliza isso como uma regressão sobre três escalas de tempo:")

formula("RVₜ₊ₕ = β₀ + β₁·RV_d + β₂·RV_w + β₃·RV_m + εₜ")

table(
    ["Termo", "Definição", "Papel"],
    [
        ("RV_d", "Variância realizada de hoje", "Reação imediata"),
        ("RV_w", "Média móvel de 5 dias úteis", "Componente semanal"),
        ("RV_m", "Média móvel de 22 dias úteis", "Componente mensal / memória longa"),
        ("Alvo", "RV anualizada realizada nos h dias seguintes", "Nunca usado como entrada"),
    ],
    widths=[1.0, 3.0, 2.5],
)

p("O ajuste é feito por mínimos quadrados sobre log(RV). A razão é estatística: RV é sempre "
  "positiva e tem distribuição aproximadamente log-normal, com cauda direita pesada. Ajustar em "
  "log evita previsões negativas, estabiliza a variância dos resíduos e, na prática, generaliza "
  "melhor fora da amostra do que ajustar o nível.")

h3("Persistência — o benchmark que precisa ser batido")

formula("RV_prevista(t+h) = RV_realizada recente (RV_m anualizada)")

p("A previsão ingênua: “a volatilidade dos próximos h dias será igual à dos últimos 22 dias”. Não "
  "tem parâmetro algum a estimar. Sua função no projeto é ser a barra mínima — se uma regressão "
  "ajustada não bate uma projeção sem parâmetros, ela não se justifica. Como a Seção 6 mostra, "
  "esse benchmark se revelou surpreendentemente difícil de superar.")

h3("GARCH(1,1) — alternativa testada")

formula("σₜ² = ω + α·εₜ₋₁² + β·σₜ₋₁²")

p("Modela a variância condicional em função do choque mais recente e da própria variância "
  "anterior. Os parâmetros são estimados por máxima verossimilhança apenas no período de treino de "
  "cada fold, congelados dentro do fold — exatamente como os coeficientes do HAR-RV, para que a "
  "comparação seja justa. No diagnóstico, teve desempenho inferior ao HAR-RV e foi descartado como "
  "candidato principal.")

h2("2.8 As camadas de informação")

h3("Notícia: tom e atenção (GDELT)")

p("O GDELT monitora a imprensa mundial e disponibiliza séries agregadas por consulta. Usamos dois "
  "modos distintos, que capturam coisas diferentes:")

bullets([
    ("Tom (timelinetone). ", "Polaridade média da cobertura — positiva ou negativa."),
    ("Atenção (timelinevolraw). ", "Que fração da cobertura total do dia fala do assunto. Mede "
     "quanto o tema domina a pauta, independentemente de o tom ser bom ou ruim."),
])

p("A distinção importa: um pacote fiscal bem recebido e uma crise fiscal geram, ambos, pico de "
  "atenção. Foi essa limitação que motivou a camada seguinte.")

h3("Sentimento fiscal via FinBERT-PT-BR")

p("Para separar “fala-se muito de fiscal” de “fala-se mal do fiscal”, coletamos as manchetes "
  "individuais sobre risco fiscal e as classificamos com o FinBERT-PT-BR. O detalhamento do uso de "
  "IA está na Seção 3.")

h3("Credibilidade do Banco Central: teoria dos jogos")

p("Esta é a camada conceitualmente mais ambiciosa. A intuição vem de Barro e Gordon (1983): sob "
  "discrição — isto é, sem compromisso crível —, o Banco Central tem incentivo a tolerar inflação "
  "surpresa para estimular atividade. O mercado antecipa esse incentivo, e o resultado de "
  "equilíbrio é viés inflacionário sem ganho real de produto.")

formula("L = ½ · [ (π − π*)² + λ·(y − yⁿ)² ]",
        note="Função de perda do Banco Central: desvio da meta de inflação e desvio do produto potencial.")

formula("y = yⁿ + α·(π − πᵉ)",
        note="Curva de Phillips: só há estímulo real se a inflação SURPREENDER a expectativa.")

table(
    ["Equilíbrio", "Mecanismo", "Consequência"],
    [
        ("Nash (discrição)", "O BC reotimiza a cada período tomando as expectativas como dadas; o mercado antecipa", "Viés inflacionário sistemático"),
        ("Stackelberg (compromisso)", "O BC se compromete de forma crível e o mercado acredita", "Expectativas ancoradas, menos flexibilidade"),
    ],
    widths=[1.6, 3.0, 1.9],
)

p("A pergunta empírica é qual equilíbrio o mercado está precificando. Implementamos a versão "
  "baseline (Tier 1) do índice, usando a distância entre a expectativa de inflação do Focus e a "
  "meta:")

formula("gapₜ = πᵉₜ − metaₜ          θₜ = exp( −|gapₜ| / escala )",
        note="θₜ próximo de 1 indica expectativas ancoradas (alta credibilidade); próximo de 0, expectativas distantes da meta.")

p("A escala é um parâmetro fixo (2,0 p.p.), deliberadamente não normalizado pelo desvio-padrão "
  "histórico do gap. A razão merece registro porque foi uma decisão consciente: o gap brasileiro é "
  "cronicamente elevado e pouco volátil em torno desse patamar. Uma normalização auto-calibrada "
  "mediria “quão incomum é este gap frente ao seu próprio histórico”, e não “quão perto da meta "
  "estão as expectativas” — colapsando θ para perto de zero mesmo com desvios moderados. Seria uma "
  "medida de surpresa, não de credibilidade.")

p("A dispersão entre os respondentes do Focus entra como segunda variável, separada: ela mede "
  "desacordo entre analistas, um conceito relacionado mas distinto de distância da meta.")

pagebreak()

# =====================================================================
# 3. IA GENERATIVA
# =====================================================================
h1("3. Uso de Inteligência Artificial")

p("A IA cumpre dois papéis distintos no VolBoy, e vale separá-los com clareza porque respondem a "
  "perguntas diferentes.")

h2("3.1 IA dentro do modelo: FinBERT-PT-BR")

p("O FinBERT-PT-BR é um modelo de linguagem especializado em texto financeiro em português. "
  "Usamos transfer learning puro — o modelo pré-treinado, sem ajuste fino próprio. A decisão foi "
  "deliberada: aprendizado de máquina é a competência menos madura da equipe, e treinar um "
  "classificador do zero com poucos rótulos produziria um resultado pior e menos defensável do que "
  "usar um modelo já validado pela comunidade.")

p("Cada manchete recebe um rótulo (positivo, negativo ou neutro) com um grau de confiança, "
  "convertidos em um escore assinado e agregados em um índice diário.")

h3("Rigor aplicado ao uso do modelo")

p("Dois cuidados metodológicos merecem destaque, porque ambos surgiram de verificação empírica e "
  "não de suposição:")

bullets([
    ("Filtro de idioma obrigatório. ",
     "A consulta de risco fiscal usa termos em inglês, mas o GDELT monitora a imprensa mundial. "
     "Ao inspecionar a amostra coletada, apenas cerca de um terço das manchetes estava em "
     "português. Aplicar um modelo treinado em português ao lote inteiro produziria escores sem "
     "significado para a maioria dos textos. O pipeline filtra por idioma antes da classificação."),
    ("Alinhamento temporal explícito. ",
     "O índice diário do FinBERT é indexado por data-calendário simples, enquanto o restante do "
     "pipeline usa timestamps com fuso horário de São Paulo. Sem normalização, a junção casaria "
     "zero registros e a feature entraria silenciosamente como valor ausente — um erro que não "
     "quebra o código e não aparece em nenhum teste superficial. Há normalização explícita e teste "
     "de regressão para isso."),
])

h2("3.2 IA no desenvolvimento: Claude Code")

p("Todo o desenvolvimento do VolBoy foi conduzido em par com o Claude Code, da Anthropic: desenho "
  "de arquitetura, implementação, revisão crítica de metodologia e redação técnica. O edital "
  "explicita que o modelo quantitativo não precisa incorporar IA diretamente — o uso de IA "
  "generativa no processo de construção atende ao critério.")

p("Vale um registro específico sobre o valor prático desse uso: os dois erros metodológicos "
  "descritos na Seção 5.4 — a instabilidade do R² por fold e o desalinhamento da fonte de preço — "
  "foram identificados durante revisões críticas conduzidas nesse processo, não em testes "
  "planejados. Ambos passariam despercebidos numa leitura convencional de código, porque nenhum "
  "deles causa erro de execução: produzem números plausíveis e silenciosamente errados.")

callout(
    "Nota sobre honestidade metodológica.",
    "Optamos por relatar os erros que cometemos e como foram descobertos, em vez de apresentar "
    "apenas o pipeline final corrigido. Um relatório que só mostra o caminho certo esconde "
    "justamente a parte do trabalho que dá confiança no resultado.",
    cor=BLUE,
)

pagebreak()

# =====================================================================
# 4. METODOLOGIA DE AVALIACAO
# =====================================================================
h1("4. Metodologia de avaliação e mitigação de vieses")

p("Esta seção descreve como o VolBoy é avaliado. Ela recebe atenção desproporcional neste "
  "relatório por uma razão de princípio: em estratégias quantitativas, quase todo resultado "
  "espetacular é um artefato de avaliação mal construída. Preferimos gastar esforço garantindo que "
  "um resultado ruim seja verdadeiro a produzir um bom que não seja.")

h2("4.1 Walk-forward purgado com embargo")

p("A regra mais básica é nunca usar divisão aleatória entre treino e teste em série temporal — "
  "isso permitiria treinar com dados posteriores aos de teste. Mas há um vazamento mais sutil, que "
  "exige tratamento adicional: a sobreposição de rótulos.")

p("O alvo de cada observação é a volatilidade realizada nos h dias seguintes. Isso significa que o "
  "rótulo da observação em t “enxerga” preços até t+h. Se uma observação de treino estiver "
  "próxima do início do período de teste, seu rótulo se sobrepõe parcialmente a esse período — e o "
  "modelo aprende, indiretamente, com informação que não estaria disponível.")

bullets([
    ("Purga. ", "Remove do treino toda observação cujo rótulo se sobreponha ao período de teste."),
    ("Embargo. ", "Exclui uma janela adicional após o teste, protegendo contra autocorrelação "
     "residual — a informação de um dia de teste ainda se reflete nos dias imediatamente seguintes."),
])

image("diagrama_walk_forward.png", width=6.3,
      caption="Figura 2 — Esquema de 5 folds com treino expansivo, purga e embargo. O treino usa apenas dados anteriores ao teste, com uma faixa descartada entre eles.")

h2("4.2 As métricas, e por que cada uma foi escolhida")

h3("Métricas de erro de previsão")

table(
    ["Métrica", "Fórmula", "Por que usamos"],
    [
        ("RMSE", "√( média( (real − previsto)² ) )", "Penaliza erros grandes desproporcionalmente; mesma unidade da RV"),
        ("MAE", "média( |real − previsto| )", "Robusta a valores extremos; comparada ao RMSE, revela se há poucos erros muito grandes"),
        ("R² fora da amostra", "1 − SS_res / SS_tot", "Métrica principal: compara o modelo contra prever simplesmente a média"),
    ],
    widths=[1.1, 2.3, 3.1],
)

p("A interpretação do R² merece precisão, porque ela sustenta toda a análise deste relatório: "
  "R² = 1 é previsão perfeita; R² = 0 significa que o modelo erra tanto quanto usar a média "
  "histórica como previsão; e ", after=2)
rich(("R² < 0 significa que o modelo erra MAIS do que a média — ou seja, ativamente atrapalha.",
      {"bold": True}))

h3("A correção que fizemos: R² agregado em vez de médio por fold")

p("A primeira implementação calculava o R² separadamente em cada fold e tirava a média. Isso "
  "contém um defeito que só aparece com folds pequenos: o R² de cada fold usa a média daquele fold "
  "como referência. Em janelas curtas, essa média é instável, o denominador fica minúsculo e um "
  "erro absoluto pequeno pode gerar um R² individual extremamente negativo. A média desses valores "
  "instáveis passa a ser dominada pelos piores folds.")

p("A correção é concatenar as previsões de todos os folds antes de calcular — usando a média de "
  "todo o período fora da amostra como referência única. É o padrão em avaliação de aprendizado de "
  "máquina aplicado a finanças (Gu, Kelly e Xiu, 2020).")

table(
    ["Baseline oficial (mesmo modelo, mesmos dados)", "R² médio por fold", "R² agregado (correto)"],
    [("HAR-RV com Parkinson, 5 folds", "−1,307", "−0,510")],
    widths=[3.2, 1.6, 1.7],
)

callout(
    "Leitura correta deste caso.",
    "O modelo não melhorou — a régua estava errada. A métrica anterior exagerava a gravidade do "
    "problema. Registramos isso porque a diferença entre “o modelo é ruim” e “nossa medição do "
    "modelo era ruim” é exatamente o tipo de distinção que separa uma avaliação confiável de uma "
    "enganosa.",
    cor=BLUE,
)

h3("Acurácia direcional")

p("R² e RMSE punem erro de magnitude mesmo quando a decisão de compra ou venda estaria correta. A "
  "acurácia direcional mede algo mais próximo do que a estratégia precisa acertar: de que lado de "
  "uma referência o valor real caiu. Acompanha um teste binomial contra a hipótese de 50% — "
  "acurácia isolada não distingue habilidade de sorte.")

h3("Métricas do backtest de P&L")

table(
    ["Métrica", "O que mede", "Cuidado na leitura"],
    [
        ("Taxa de acerto", "Fração de operações com resultado positivo", "Só considera os dias em que houve operação — subconjunto pequeno e não aleatório"),
        ("Sharpe anualizado", "Retorno por unidade de risco", "Amostra pequena o torna ruidoso; não corrige seleção de configuração"),
        ("PSR", "Probabilidade de o Sharpe verdadeiro superar um benchmark", "Ajusta para não-normalidade dos retornos (opções têm assimetria e curtose)"),
        ("DSR", "Sharpe ajustado pelo nº de configurações testadas", "A métrica honesta quando várias variantes foram avaliadas"),
    ],
    widths=[1.2, 2.3, 3.0],
)

h2("4.3 Custos de transação")

formula("custo = |n_contratos| × prêmio × spread_pct",
        note="Cobrado na abertura E no encerramento da posição. Parâmetro padrão: 5% do prêmio.")

p("Sem acesso ao livro de ofertas, o custo é aproximado como fração do prêmio. Escolhemos um valor "
  "deliberadamente conservador. Um achado da Seção 7 sugere, porém, que mesmo 5% pode ser "
  "otimista para este mercado.")

h2("4.4 Controle de overfitting")

p("Testar muitas configurações e reportar a melhor superestima sistematicamente o desempenho "
  "esperado — o número reportado passa a ser a estatística de máximo de um conjunto de tentativas, "
  "não uma medida de habilidade. O Deflated Sharpe Ratio corrige isso comparando o Sharpe "
  "observado contra o máximo esperado por puro acaso, dado o número de tentativas:")

formula("SR_max_esperado(N) = σ_SR · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]",
        note="γ é a constante de Euler-Mascheroni; σ_SR é a dispersão dos Sharpes entre as N configurações testadas.")

p("Complementarmente — e talvez mais importante que a fórmula — o projeto mantém um registro vivo "
  "de todas as configurações já avaliadas, incluindo as que falharam. São 22 entradas ao final "
  "deste ciclo. Esse registro é auditável e está reproduzido na Seção 5.1. Sem ele, o denominador "
  "N do DSR seria uma escolha arbitrária de quem reporta.")

pagebreak()

# =====================================================================
# 5. TRAJETORIA
# =====================================================================
h1("5. A trajetória: o que testamos e por quê")

p("Esta seção percorre o desenvolvimento em ordem cronológica. O objetivo não é documentar apenas "
  "o resultado final, mas o raciocínio — incluindo os caminhos que não deram certo, que são a "
  "maioria.")

h2("5.1 Registro das configurações avaliadas")

table(
    ["#", "Configuração", "Resultado"],
    [
        ("1", "Split único 80/20, notícia bruta, RV em nível", "Preliminar; sem rigor de série temporal"),
        ("2", "CV expansiva 5 folds, notícia suavizada, log-RV", "Refinamento do anterior"),
        ("3", "Walk-forward purgado + embargo (proxy retorno²)", "Primeira avaliação rigorosa"),
        ("4", "Persistência pura como diagnóstico", "Revelou que o problema não era a regressão"),
        ("5", "GARCH(1,1)", "Pior que HAR-RV; descartado"),
        ("6", "HAR-RV com estimador Parkinson", "RMSE menor em todos os folds; adotado"),
        ("7", "Credibilidade Tier 1 (θₜ + dispersão)", "Não ajuda"),
        ("8", "Risco fiscal — nível bruto de atenção", "Não ajuda"),
        ("9", "Risco fiscal — surpresa (z-score, 63d)", "Não ajuda"),
        ("10", "Risco fiscal — surpresa (z-score, 21d)", "Efeito marginal, dentro do ruído"),
        ("11", "Diagnóstico: R² por fold vs. agregado", "ERRO NOSSO corrigido; métrica oficial trocada"),
        ("12", "Varredura de esquemas de walk-forward", "1 config venceu; vizinhas falharam → rejeitada como sorte"),
        ("13", "Leverage via interação multiplicativa", "Sem efeito (multicolinearidade)"),
        ("14", "Risco fiscal v2 (surpresa + FinBERT)", "Parecia ajudar com dado parcial; inverteu com dado completo"),
        ("15", "Overnight + semivariância + ensemble", "INVÁLIDO — testado sobre dado defeituoso (ver 5.4)"),
        ("16", "Risco global exógeno (VIX + DXY)", "Não ajuda; VIX chega a piorar"),
        ("17", "Persistência adotada como previsão oficial", "Único R² positivo — depois revisto"),
        ("18", "Camadas como correção de resíduo da persistência", "Todas as 6 pioram"),
        ("19", "Validação de fonte de preço contra PTAX", "ERRO NOSSO encontrado: defasagem de 1 dia"),
        ("20", "Migração para o futuro de dólar da B3", "R² positivo NÃO sobrevive"),
        ("21", "Varredura de horizonte na fonte correta", "Sinal existe em 1 dia; ausente em 21"),
        ("22", "Cross-check com estimador alternativo (B3)", "Confirma: negativo em ambos os estimadores"),
    ],
    widths=[0.35, 3.2, 2.85],
    size=8.6,
    highlight={10, 14, 18, 19, 20},
)

p("As linhas destacadas marcam os pontos de inflexão do projeto. Três delas são erros nossos "
  "descobertos por verificação; duas são reversões de conclusões anteriores.",
  size=9, color=MUTED)

h2("5.2 Fase inicial: construir a barra a ser superada")

p("O primeiro resultado relevante foi negativo e veio cedo: o HAR-RV ajustado não superava a "
  "persistência pura. Isso poderia significar duas coisas muito diferentes — que a regressão "
  "estava mal especificada, ou que não havia sinal a extrair. O teste com a persistência (item 4 "
  "da tabela) separou as hipóteses: como ela também apresentava desempenho fraco em termos "
  "absolutos, o problema não estava na regressão.")

p("A troca do estimador de retorno ao quadrado pelo Parkinson (item 6) foi a primeira melhoria "
  "genuína, e por isso foi adotada: RMSE menor em todos os folds, não apenas na média — critério "
  "que escolhemos justamente para evitar adotar uma mudança que funcionasse por acaso em um "
  "período específico.")

h2("5.3 Fase das camadas de informação")

p("Aqui estava a aposta central do projeto. Cada camada foi submetida ao mesmo teste: ablação "
  "formal com e sem a camada, nos mesmos folds purgados.")

p("O caso do risco fiscal merece detalhamento porque contém uma lição metodológica. A coleta de "
  "manchetes do GDELT foi repetidamente bloqueada por limite de taxa, e chegamos a ter apenas 12 "
  "das 15 janelas-alvo. Com esses dados parciais, a camada com FinBERT parecia ajudar de forma "
  "consistente — R² melhorando em 3 de 5 folds, com o coeficiente de sentimento aparentemente "
  "estável.")

callout(
    "O resultado que se desfez.",
    "Ao completar a coleta (15 de 15 janelas), o mesmo teste inverteu: de +0,073 para −0,568. "
    "O desvio-padrão do coeficiente praticamente dobrou. Se tivéssemos reportado o resultado "
    "parcial — que era o resultado mais animador de todo o projeto até então —, teríamos publicado "
    "uma conclusão falsa. Foi a decisão de completar os dados antes de concluir que evitou isso.",
    cor=RED,
)

h2("5.4 Os dois erros que encontramos no nosso próprio trabalho")

h3("Erro 1 — a métrica instável")

p("Descrito na Seção 4.2. O R² médio por fold exagerava sistematicamente a gravidade do problema. "
  "Correção: R² agregado.")

h3("Erro 2 — a fonte de preço desalinhada")

p("Este foi mais grave e mais difícil de perceber. Ao validar o proxy de preço (yfinance BRL=X) "
  "contra os preços oficiais do futuro da B3, encontramos uma inconsistência estrutural nas barras "
  "diárias do yfinance:")

bug = R["bug_alinhamento"]
table(
    ["Medida", "Valor", "Interpretação"],
    [
        ("|fechamento − abertura| do mesmo dia", num(bug['yf_intra_medio'], 5), "Praticamente zero"),
        ("|abertura − fechamento anterior|", num(bug['yf_gap_medio'], 5), f"{bug['yf_razao']:.0f}× maior"),
        ("Correlação abertura × fechamento", num(bug['yf_corr_open_close'], 5), "São o mesmo ponto de preço"),
    ],
    widths=[2.6, 1.4, 2.4],
)

image("bug_alinhamento.png", width=5.8,
      caption="Figura 3 — Nas barras do yfinance, o movimento não acontece dentro do dia: aparece entre barras.")

p("Abertura e fechamento são o mesmo instante — não é um fechamento de fim de pregão, é um "
  "snapshot no limite do dia. Para confirmar a direção do problema, usamos o PTAX do Banco Central "
  "como árbitro independente, por ser oficial e datado sem ambiguidade:")

image("validacao_ptax.png", width=5.8,
      caption="Figura 4 — Correlação com o retorno do PTAX. Só o ajuste do futuro da B3 alinha no mesmo dia.")

ptax_v = R["validacao_ptax"]
p(f"O ajuste do futuro da B3 correlaciona {num(ptax_v['futuro_b3']['0'])} com o PTAX no mesmo dia. "
  f"Já o fechamento do yfinance alinha melhor com defasagem de um dia "
  f"({num(ptax_v['yfinance']['1'])}) do que no mesmo dia ({num(ptax_v['yfinance']['0'])}).")

h3("O que esse erro comprometeu — e o que não comprometeu")

table(
    ["Componente", "Situação"],
    [
        ("RV por Parkinson (estimador oficial)", "ÍNTEGRO — usa apenas máxima e mínima, que estão corretamente datadas"),
        ("Feature de gap overnight", "INVÁLIDA — media o retorno do dia inteiro, não o gap (correlação 0,998 com retorno diário²)"),
        ("Sinal da semivariância (leverage)", "COMPROMETIDO — sinal do retorno defasado em relação à variância"),
        ("Preços de entrada e saída do backtest", "DEFASADOS em um dia"),
    ],
    widths=[2.4, 4.0],
)

p("A consequência mais importante: o item 15 da tabela de configurações, que registramos como "
  "“resultado nulo”, era na verdade um teste inválido. Corrigimos o registro. A distinção importa — "
  "um resultado nulo legítimo informa; um teste inválido rotulado como nulo desinforma.")

pagebreak()

h2("5.5 A migração de fonte e a reversão da conclusão")

p("Corrigido o diagnóstico, a decisão seguinte era clara: migrar para o futuro de dólar da B3. "
  "Não por elegância, mas porque é o instrumento economicamente correto — a opção que o VolBoy "
  "negocia é escrita sobre esse futuro, então a volatilidade relevante é a dele.")

amostra = R["amostra"]
p(f"Coletamos {amostra['b3_pregoes']} pregões ({amostra['b3_inicio']} a {amostra['b3_fim']}), "
  f"cobrindo {amostra['b3_contratos']} contratos com {amostra['b3_rolagens']} rolagens mensais, "
  "diretamente dos arquivos diários da B3. A rolagem exigiu tratamento explícito: a série troca de "
  "contrato todo mês, e emendar preços de contratos diferentes criaria saltos artificiais que "
  "entrariam como volatilidade falsa. O estimador de Parkinson, por operar dentro de cada pregão, "
  "é imune a esse problema.")

p("O resultado da migração inverteu a conclusão central do projeto:")

table(
    ["Fonte de preço", "HAR-RV", "Persistência", "Veredito"],
    [
        ("yfinance BRL=X (proxy)", fmt(yf_21["har_r2"]), fmt(yf_21["persist_r2"]), "Persistência vence, R² positivo"),
        ("Futuro de dólar B3 (oficial)", fmt(b3_21["har_r2"]), fmt(b3_21["persist_r2"]), "Ambos negativos"),
    ],
    widths=[2.2, 1.2, 1.4, 2.0],
    highlight={1},
)

p("Um cross-check com estimador completamente diferente (retorno de fechamento sobre o ajuste "
  "oficial) confirmou o resultado no mesmo dado da B3, descartando a hipótese de que fosse "
  "artefato do estimador de Parkinson.")

h2("5.6 O diagnóstico final: descasamento de horizonte")

p("Restava explicar por que a fonte importava tanto. A resposta está na estrutura de memória da "
  "série — precisamente o que a previsão de longo horizonte explora:")

image("autocorrelacao.png", width=5.9,
      caption="Figura 5 — Autocorrelação da variância diária. O spot 24h retém memória em defasagens longas; o futuro, não.")

acf = R["autocorrelacao"]
i5, i21 = acf["lags"].index(5), acf["lags"].index(21)
p(f"Em defasagem 5, a autocorrelação é {num(acf['b3'][i5])} no futuro contra {num(acf['yfinance'][i5])} "
  f"no spot; em defasagem 21, {num(acf['b3'][i21])} contra {num(acf['yfinance'][i21])}. Como o alvo é "
  "a volatilidade dos próximos 21 dias, é essa memória longa que sustenta qualquer previsão — e "
  "ela existe no spot, não no futuro.")

h3("Duas leituras, e o que decide entre elas")

bullets([
    ("Efeito estatístico legítimo. ",
     "O intervalo de 24 horas do spot média mais do processo de volatilidade, reduzindo ruído de "
     "medição. Alvo menos ruidoso permite R² maior. Isso é real."),
    ("Artefato de construção. ",
     "As barras anômalas do yfinance podem inflar artificialmente a autocorrelação."),
])

p("Não conseguimos separar completamente as duas — e registramos isso como limitação. Mas o ponto "
  "decisivo não depende dessa separação: o straddle do VolBoy tem payoff determinado pelo caminho "
  "do preço do futuro. A volatilidade do futuro é o alvo economicamente correto, e é nela que "
  "nenhum método funciona.")

p("A varredura de horizonte na fonte correta fecha o diagnóstico:")

sweep_rows = []
for h in ["1", "3", "5", "10", "15", "21"]:
    d = R["horizonte"]["b3"][h]
    sweep_rows.append((f"{h} dia(s)", fmt(d["har_r2"]), fmt(d["persist_r2"]),
                       "sinal real" if d["har_r2"] > 0 else "sem sinal"))
table(["Horizonte", "HAR-RV", "Persistência", "Leitura"], sweep_rows,
      widths=[1.3, 1.3, 1.5, 2.2], highlight={0, 5})

p("Existe previsibilidade genuína, ainda que modesta, a um dia — e é justamente ali que o HAR-RV "
  "supera a persistência, exatamente como Corsi (2009) previa, já que o modelo foi desenhado para "
  "horizonte curto. Ela decai monotonicamente até se tornar francamente negativa em 21 dias.")

callout(
    "O diagnóstico central do VolBoy.",
    "Depois de 22 configurações, a explicação não é “faltou a feature certa”. É descasamento de "
    "horizonte: o sinal existe onde a estratégia não consegue usá-lo (1 dia) e está ausente no "
    "prazo de que ela precisa (21 dias, vencimento da opção). Nenhuma variável adicional resolve "
    "isso — é uma propriedade do processo de volatilidade neste período e neste instrumento.",
    cor=RED,
)

pagebreak()

# =====================================================================
# 6. RESULTADOS
# =====================================================================
h1("6. Resultados e análise crítica")

h2("6.1 Volatilidade realizada no período")

image("rv_historico.png", width=6.2,
      caption="Figura 6 — RV de 21 dias anualizada, nas duas fontes. Os picos coincidem com episódios de estresse fiscal e cambial.")

rv = R["rv"]
p(f"A volatilidade média do futuro no período foi de {num(rv['b3_media'], 2)}% ao ano, oscilando entre "
  f"{num(rv['b3_min'], 2)}% e {num(rv['b3_max'], 2)}%. As duas fontes correlacionam {num(rv['corr_rv21'])} "
  "no nível de RV de 21 dias — próximas, mas não idênticas, o que já sinalizava que a escolha de "
  "fonte não era indiferente.")

h2("6.2 Ablações: o veredito sobre cada camada")

table(
    ["Camada testada", "Formulação", "Resultado"],
    [
        ("Notícia (tom GDELT)", "Tom médio suavizado em 21 dias", "Não ajuda"),
        ("Credibilidade do BC", "θₜ + dispersão do Focus", "Não ajuda; piora o R²"),
        ("Risco fiscal — atenção", "% da cobertura sobre o tema", "Ganho aparente, mas inconsistente (1 de 5 folds)"),
        ("Risco fiscal — surpresa", "z-score vs. média móvel", "Efeito dentro do ruído"),
        ("Risco fiscal — sentimento", "FinBERT-PT-BR nas manchetes", "Inverteu ao completar os dados"),
        ("Risco global exógeno", "VIX + DXY", "Não ajuda; VIX piora"),
        ("Extensões de literatura", "Overnight, semivariância, ensemble", "Teste invalidado pelo erro de fonte"),
    ],
    widths=[1.9, 2.3, 2.4],
)

p("Um teste adicional merece registro por ser o mais direto de todos. Depois de adotar a "
  "persistência como referência, testamos as seis camadas não como substitutas do modelo, mas como "
  "correção do resíduo da persistência — a pergunta mais justa possível: “esta informação explica "
  "o que o melhor baseline ainda erra?”. Todas as seis pioraram o resultado.")

p("A interpretação é direta e vale explicitar: a persistência não tem nenhum parâmetro estimado, "
  "logo não corre risco de sobreajuste. Qualquer correção por regressão introduz variância de "
  "estimação. Se a informação adicionada não carrega sinal real, o modelo ajusta ruído — e "
  "generaliza pior. A simplicidade, aqui, não é preguiça: é a escolha estatisticamente superior.")

h2("6.3 Backtest de P&L e por que ele não deve ser levado a sério")

callout(
    "Advertência necessária.",
    "A B3 publica apenas o snapshot do dia da superfície de volatilidade, sem histórico para "
    "download. Existe UM único dia de IV real em todo o projeto. A IV de entrada de cada operação "
    "simulada é uma proxy: a RV recente multiplicada por um prêmio de risco fixo calibrado nesse "
    "único dia. Qualquer número de Sharpe ou taxa de acerto daí é ilustrativo do motor, não "
    "evidência de lucro.",
    cor=RED,
)

p("Essa limitação produziu um efeito instrutivo. Ao trocar o modelo de previsão pelo que tinha "
  "melhor R², o backtest piorou — resultado aparentemente contraditório. A investigação revelou a "
  "causa: a previsão por persistência e a IV proxy têm correlação de 0,994, porque a proxy é "
  "construída como múltiplo constante da própria RV recente. O spread que decide comprar ou vender "
  "é dominado por uma constante multiplicativa, e o momento exato em que ele cruza a banda passa a "
  "ser ditado por ruído de curtíssimo prazo.")

p("A conclusão metodológica é que o Sharpe deste backtest não deve ser usado para escolher entre "
  "modelos enquanto não houver IV histórica real. O R², medido contra o valor efetivamente "
  "realizado, é a única métrica confiável do projeto hoje.")

h2("6.4 Por que R² e taxa de acerto podem discordar")

p("Vale explicitar essa distinção, porque ela é fonte recorrente de leitura equivocada em "
  "estratégias quantitativas. As duas métricas são calculadas sobre conjuntos diferentes:")

bullets([
    ("R² — todos os dias da amostra. ", "Mede se, em média ao longo de todo o período, a previsão "
     "ficou próxima do valor real."),
    ("Taxa de acerto — apenas os dias com operação. ", "Um subconjunto pequeno e não aleatório, "
     "selecionado por um limiar. Ser bom na média não garante nada sobre esse recorte específico."),
])

p("Um previsor com boa acurácia média pode disparar operações justamente nos momentos em que a "
  "volatilidade está temporariamente inflada e prestes a reverter. É por isso que um R² melhor "
  "pode conviver com uma taxa de acerto pior — e por que reportar apenas a segunda seria enganoso.")

pagebreak()

# =====================================================================
# 7. LIMITACOES
# =====================================================================
h1("7. Limitações declaradas")

h2("7.1 Ausência de histórico de volatilidade implícita")

p("É a limitação mais séria, e ela é estrutural, não uma falha de execução. A B3 sobrescreve o "
  "arquivo da superfície de volatilidade diariamente. A tese central do VolBoy — comparar RV "
  "prevista contra IV de mercado — não pode ser validada historicamente sem essa série.")

h2("7.2 O mercado de opções de dólar é ilíquido no nível de série")

p("Tentamos reconstruir a IV histórica a partir dos preços de ajuste das opções, invertendo "
  "Black-76 numericamente. A investigação nos arquivos oficiais da B3 mostrou que isso não é "
  "viável, e o motivo é um achado relevante por si só:")

table(
    ["Constatação (data verificada)", "Valor"],
    [
        ("Séries de opções de dólar no arquivo", "2.042"),
        ("Séries com preço de ajuste publicado", "0"),
        ("Séries que efetivamente negociaram", "10 (com 1 a 2 negócios cada)"),
    ],
    widths=[4.0, 2.4],
)

p("Isso explica por que a própria B3 constrói a superfície a partir de um painel de informantes, e "
  "não a partir de negócios: não existe mercado líquido de onde derivá-la. Duas implicações "
  "diretas para o VolBoy:")

bullets([
    "A hipótese de custo de 5% do prêmio é provavelmente otimista para um mercado com essa liquidez.",
    "A execução de um straddle no dinheiro enfrentaria dificuldade real, além do custo — um risco "
    "que a simulação não captura.",
])

h2("7.3 Amostra e período")

p("Cerca de três anos e quatro meses de dados, com aproximadamente 830 pregões. Para um alvo de 21 "
  "dias, isso significa poucas dezenas de janelas verdadeiramente independentes. Conclusões sobre "
  "ausência de sinal valem para este período e este instrumento; não são afirmações universais.")

h2("7.4 Simplificações de apreçamento assumidas")

bullets([
    ("Taxa de juros nula no Black-76. ", "Sem curva de juros própria coletada. Afeta o nível do "
     "prêmio, não a forma do vega usada no dimensionamento."),
    ("Spot como proxy do forward. ", "Pelo mesmo motivo — sem curva, não há como calcular o "
     "forward via paridade coberta de juros."),
    ("Ausência de delta-hedge dinâmico. ", "O payoff terminal do straddle é calculado de forma "
     "exata, o que é adequado para uma estratégia de nível de volatilidade, mas não captura o "
     "resultado de gamma scalping intradiário."),
])

pagebreak()

# =====================================================================
# 8. CONCLUSAO
# =====================================================================
h1("8. Conclusão e próximos passos")

h2("8.1 O projeto atendeu ao que se propôs?")

p("A resposta honesta é parcialmente — e vale separar as duas partes com clareza.")

h3("O que a hipótese central prometia")

p("A tese era que fluxo de notícias e credibilidade institucional melhorariam a previsão de "
  "volatilidade do dólar, gerando vantagem sobre a expectativa embutida no mercado. "
  "Essa hipótese foi testada com rigor e não se sustentou. Nenhuma das camadas produziu ganho "
  "consistente, e o teste mais direto — usá-las para corrigir o resíduo do melhor baseline — "
  "mostrou todas piorando o resultado. Como proposta de gerar alfa, o VolBoy não entrega.")

h3("O que foi efetivamente construído")

bullets([
    ("Infraestrutura completa e reproduzível. ", "Coletores idempotentes para oito fontes públicas, "
     "240 testes automatizados, pipeline modular com separação estrita entre I/O e lógica."),
    ("Avaliação rigorosa. ", "Walk-forward purgado com embargo, R² agregado, Deflated Sharpe "
     "Ratio, custos sempre incluídos, registro auditável de todas as configurações testadas."),
    ("Capacidade de encontrar os próprios erros. ", "Dois defeitos metodológicos silenciosos — que "
     "produziam números plausíveis e errados — foram detectados e corrigidos por verificação "
     "ativa, não por acaso."),
    ("Um diagnóstico específico e testável. ", "Não “nada funcionou”, mas: o sinal existe em "
     "horizonte de um dia e desaparece em 21, e a estratégia precisa dele em 21."),
])

h2("8.2 O que foi mais interessante de analisar")

p("Três episódios se destacam, e todos têm em comum o mesmo padrão: um resultado inicial "
  "favorável que não sobreviveu a escrutínio adicional.")

bullets([
    ("O FinBERT que funcionou e depois não. ", "Com 12 das 15 janelas de dados, a camada de "
     "sentimento fiscal parecia ajudar de forma consistente. Ao completar a coleta, o efeito "
     "inverteu completamente. É uma demonstração viva de armadilha de amostra pequena — e o "
     "resultado parcial era o mais animador que o projeto havia produzido."),
    ("A configuração vencedora que era sorte. ", "Uma combinação de walk-forward superou a "
     "persistência. Testar as configurações vizinhas mostrou que todas falhavam — era seleção por "
     "acaso, exatamente o que o Deflated Sharpe existe para prevenir. Foi rejeitada."),
    ("O R² positivo que a fonte correta desfez. ", "O único resultado positivo do projeto vinha de "
     "uma fonte de preço com defeito estrutural. Ao usar o instrumento efetivamente negociado, ele "
     "desapareceu."),
])

p("Se há uma lição transversal, é que a diferença entre um projeto quantitativo confiável e um "
  "enganoso raramente está no modelo — está na disposição de continuar testando depois que o "
  "resultado já ficou bom.", italic=True)

h2("8.3 Próximos passos")

h3("Prioridade alta — destravam validação")

bullets([
    ("Arquivar a superfície de IV diariamente. ", "Trivial de implementar e já deveria estar "
     "rodando: cada dia sem coletar é um dia de histórico perdido para sempre. É o único caminho "
     "para validar a tese central."),
    ("Reformular o horizonte da estratégia. ", "O diagnóstico da Seção 5.6 aponta diretamente para "
     "isso. Se o sinal está em horizonte curto, a estratégia precisaria operar opções de prazo "
     "curto — o que esbarra na liquidez documentada na Seção 7.2. Avaliar essa tensão é a decisão "
     "estratégica mais importante do projeto."),
])

h3("Prioridade média — melhoram a base")

bullets([
    ("Refazer o teste de overnight com dados corretos. ", "A B3 fornece primeiro e último negócio "
     "do pregão, permitindo um teste legítimo de gap overnight pela primeira vez."),
    ("Migrar o pipeline completo para a fonte da B3. ", "As ablações e o backtest ainda rodam "
     "sobre a fonte antiga; a comparação central já foi migrada."),
    ("Modelar o prêmio de risco de forma variável. ", "A proxy de IV atual usa um multiplicador "
     "constante, que é a origem da colinearidade discutida em 6.3."),
])

h3("Escopo descartado conscientemente")

bullets([
    ("Modelos mais complexos. ", "Gradient boosting ou redes neurais seriam adicionar capacidade "
     "onde já demonstramos ausência de sinal — aumentaria o risco de sobreajuste sem endereçar o "
     "diagnóstico."),
    ("Tier 2 e 3 da camada de credibilidade. ", "Classificador hawkish/dovish, θ como estado "
     "latente, momentos implícitos BKM. Interessantes conceitualmente, mas o Tier 1 já não "
     "mostrou sinal — não há razão para sofisticar a camada antes de haver evidência de que ela "
     "carrega informação."),
])

h2("8.4 Considerações finais")

p("O VolBoy termina este ciclo sem demonstrar vantagem preditiva sobre o mercado — e com uma "
  "compreensão precisa do motivo. Consideramos que a segunda parte tem mais valor que a primeira "
  "teria isoladamente: uma estratégia que parece funcionar sem que se saiba por quê é um passivo; "
  "um resultado negativo bem estabelecido é conhecimento.")

p("O projeto entrega uma infraestrutura reproduzível, um método de avaliação que resistiu a "
  "múltiplas tentativas de encontrar sinal onde não havia, e um registro completo do que foi "
  "tentado — incluindo os erros cometidos e o modo como foram descobertos. Para uma equipe que "
  "vai continuar operando neste espaço, esse conjunto vale mais que um backtest favorável de "
  "origem incerta.")

pagebreak()

# =====================================================================
# 9. REFERENCIAS + ANEXO
# =====================================================================
h1("9. Referências")

bullets([
    "Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. Journal of Financial Econometrics.",
    "Parkinson, M. (1980). The Extreme Value Method for Estimating the Variance of the Rate of Return. Journal of Business.",
    "Black, F. (1976). The Pricing of Commodity Contracts. Journal of Financial Economics.",
    "Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. Journal of Econometrics.",
    "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.",
    "Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio. Journal of Portfolio Management.",
    "Barro, R. J. & Gordon, D. B. (1983). A Positive Theory of Monetary Policy in a Natural Rate Model. Journal of Political Economy.",
    "Gu, S., Kelly, B. & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. Review of Financial Studies.",
    "Barndorff-Nielsen, O., Kinnebrock, S. & Shephard, N. (2010). Measuring Downside Risk: Realised Semivariance.",
    "Kambouroudis, D., McMillan, D. & Tsakou, K. (2021). Forecasting realized volatility. Journal of Futures Markets.",
    "FinBERT-PT-BR — huggingface.co/lucas-leme/FinBERT-PT-BR",
    "B3 — Manual de Apreçamento de Opções; arquivos BVBG-086.",
    "Banco Central do Brasil — API SGS, API Olinda (Focus).",
    "GDELT Project — DOC 2.0 API.",
], size=9.5)

h1("Anexo — mapa do código")

table(
    ["Módulo", "Responsabilidade"],
    [
        ("data/", "Oito coletores idempotentes: B3 (futuros, superfície de IV), BCB (PTAX, Focus, IBC-Br, COPOM), GDELT, RSS, VIX/DXY"),
        ("vol/", "RV realizada (Parkinson), IV implícita interpolada, Black-76 e gregas, previsão HAR-RV e GARCH"),
        ("sentiment/", "FinBERT-PT-BR e agregação em índice diário, com filtro de idioma"),
        ("credibility/", "Índice θₜ de credibilidade, hiato do produto por HP expansivo, ablação própria"),
        ("strategy/", "Regra de sinal com banda morta e dimensionamento por vega"),
        ("backtest/", "Walk-forward purgado, ablações formais, motor de P&L, métricas robustas"),
        ("report/", "Gráficos, resumo e o script único que orquestra a avaliação"),
        ("relatorio/", "Geração deste documento e das figuras, a partir dos dados reais"),
        ("tests/", "240 testes, um arquivo por módulo"),
    ],
    widths=[1.3, 5.1],
)

p("Reprodução completa: `python -m relatorio.gerar_dados` recalcula todos os números e figuras; "
  "`python -m relatorio.gerar_relatorio` reconstrói este documento. Nenhum número citado aqui foi "
  "digitado manualmente.", size=9, color=MUTED, before=8)

doc.save(OUT_PATH)
print("salvo:", OUT_PATH)
