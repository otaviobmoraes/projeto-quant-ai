# CLAUDE.md — Módulo `credibility/` · Fase 2: Credibilidade do BC via Teoria dos Jogos

> Módulo da Fase 2 do projeto. O `CLAUDE.md` da raiz continua valendo integralmente
> (tese de vol, guardrails, convenções). Este arquivo **estende**, não substitui.

## O que esta fase muda na tese

**Antes (Fase 1):** sentimento de notícias era uma *feature solta* alimentando a previsão de RV.

**Agora (Fase 2):** o texto vira **parâmetro estrutural de um jogo**. Estimamos um score de
credibilidade do Banco Central **θt ∈ [0,1]**, onde θt = 1 é compromisso puro (Stackelberg) e
θt = 0 é discrição pura (Nash). Esse θt é o parâmetro que interpola entre os dois equilíbrios
do modelo de Barro-Gordon.

**Por que isso conecta com vol de USD/BRL:** sob discrição, o BC tem incentivo a tolerar
inflação surpresa quando o hiato do produto é negativo. Isso eleva **assimetricamente** o risco de
depreciação do real. Logo, queda de credibilidade deveria alargar a **cauda direita** da distribuição
implícita de USD/BRL (skew), não apenas o nível de vol. **Se θt cai e o skew ainda não reagiu,
há trade.**

A estratégia passa a ter duas camadas:

- **Nível:** previsão de RV (agora aumentada por credibilidade) vs. IV ATM → long/short vol (straddle).
- **Forma:** θt prevê o skew da distribuição implícita → risk reversal / trade de skew.

## Estrutura formal do jogo (Barro-Gordon)

Perda do BC: `L = ½[(π − π*)² + λ(y − y*)²]`
Curva de Phillips: `y = yⁿ + α(π − πᵉ)`

- **Nash (discrição):** BC reotimiza a cada período tomando πᵉ como dado → viés inflacionário sistemático.
- **Stackelberg (compromisso):** BC anuncia e se compromete, πᵉ consistente → sem viés, menos flexibilidade.

Pergunta empírica central do projeto: **qual equilíbrio o mercado está precificando agora?**

## Fontes de dados desta fase (todas gratuitas)

| Insumo | Fonte | Nota |
|---|---|---|
| **πᵉ (expectativa de inflação)** | **Boletim Focus / BCB (API Olinda + SGS)** | **Peça central.** Semanal. É literalmente o πᵉ da curva de Phillips, observado direto. |
| **Dispersão de expectativas** | Focus (desvio-padrão entre ~130 participantes) | Análogo empírico da coordenação de crenças. Alta dispersão = baixa credibilidade. |
| Meta de inflação (π*) | CMN / BCB | Necessária para o desvio πᵉ − π*. |
| Hiato do produto (y − y*) | IBGE/BCB (IBC-Br ou PIB) + filtro HP | Cuidado com o problema de ponta do HP; considerar Hamilton como robustez. |
| Selic e curva DI | BCB (SGS) e B3 | Para taxa esperada e surpresa de decisão. |
| Comunicação do COPOM | Comunicados, atas e discursos (site BCB) | Insumo do classificador de tom. |
| Opções de dólar / IV | **Já existe na Fase 1** (`vol/implied.py`) | Reuso direto da cadeia e do smile. |

**Não usar Bloomberg como dependência.** Regra da raiz continua valendo.

## Papel do ML nesta fase

O ML é **auxiliar mas estrutural**: extrai o input textual que vira parâmetro do jogo.
Não é ML de ponta a ponta, e isso é intencional — o edital não premia complexidade.

### ⚠️ Hawkish/dovish ≠ polaridade de sentimento

**FinBERT-PT-BR NÃO serve para tom de política monetária.** Ele classifica positivo/negativo/neutro.
Uma ata pode ser "negativa" e hawkish simultaneamente. São eixos ortogonais.

**Abordagem correta (nesta ordem):**

1. **Classificação zero-shot/few-shot via LLM** dos parágrafos do COPOM numa escala hawkish–dovish
   (ex.: −1 a +1), com rubrica explícita e exemplos no prompt. É o caminho principal:
   mais simples que fine-tuning **e** conta como uso de IA Generativa (15% da nota).
2. **Baseline por dicionário** (léxico hawkish/dovish adaptado da literatura) para comparação.
   Serve de sanity check e mostra rigor.
3. Fine-tuning: **apenas se sobrar tempo**. Não é prioridade.

**Rigor obrigatório na classificação por LLM:**
- Prompt versionado em arquivo, temperatura 0, saída em JSON estruturado.
- **Cachear toda resposta** (hash do texto → score). Reclassificar é caro e não determinístico.
- Medir **concordância**: reclassificar uma amostra N vezes e reportar a variância; classificar
  manualmente ~50 parágrafos como *gold set* e reportar a concordância com o LLM.
- Registrar modelo e versão do prompt junto do score.

### Do tom para θt

O sinal não é o tom em si, é a **surpresa de tom**: tom observado menos tom esperado
(esperado = previsão a partir da regra de Taylor / consenso Focus / tom da reunião anterior).
θt é estimado relacionando surpresa de tom → reação da distribuição implícita, condicionado ao
hiato do produto. Preferir uma formulação de **estado latente** (filtro de Kalman ou bayesiano
simples) a uma regressão solta: credibilidade é persistente, não ruído i.i.d.

## Roadmap em tiers (respeitar os cortes)

**Prazos: pré-relatório 31/07/2026 · entrega final 17/08/2026.** O tempo é o constraint dominante.

### Tier 1 — obrigatório, tem que estar de pé no pré-relatório
- Coletor do Focus (πᵉ, dispersão) + meta + hiato do produto.
- **Índice de credibilidade baseline:** desvio de πᵉ vs. meta, normalizado. Sem NLP, sem jogo ainda.
- Alimentar θt_baseline e dispersão como features na previsão de RV existente.
- **Rodar a ablação:** previsão de RV com vs. sem credibilidade. Esse é o resultado mínimo publicável.

Tier 1 sozinho já entrega narrativa de teoria dos jogos com risco técnico baixo. **Não avance sem ele fechado.**

### Tier 2 — alvo realista para a entrega final
- Classificador hawkish/dovish via LLM + baseline de dicionário + gold set.
- Surpresa de tom em janela de evento ao redor do COPOM.
- Calibração do Barro-Gordon (λ, viés inflacionário) e θt como estado latente interpolando Nash–Stackelberg.
- Comparar θt do modelo com índices de credibilidade da literatura brasileira (benchmark de validação).

### Tier 3 — stretch, cortar sem dó se o prazo apertar
- **Momentos BKM (Bakshi-Kapadia-Madan)**: skew e curtose implícitas model-free da cadeia de opções.
  **Fazer BKM ANTES de tentar RND completa** — é muito mais robusto com strikes esparsos.
- RND completa via Breeden-Litzenberger: só depois de BKM, e só com smile arbitrage-free (SVI)
  diferenciado analiticamente. **Nunca** diferenciar duas vezes um smile interpolado cru.
- Trade de skew (risk reversal) além do trade de nível (straddle).

## Riscos conhecidos e mitigações

| Risco | Mitigação |
|---|---|
| **RND instável** (strikes esparsos na B3 → densidade com massa negativa) | Usar BKM em vez de RND completa. Se for para RND: SVI arbitrage-free + derivada analítica. Sempre validar densidade ≥ 0 e integral ≈ 1. |
| **Amostra pequena** (COPOM = 8 reuniões/ano) | Focus é semanal → usar como espinha dorsal de frequência. Incluir discursos, não só atas. Reportar intervalo de confiança e ser explícito sobre n. |
| **Endogeneidade** (tom e preço se movem juntos) | Janela de evento estreita ao redor do anúncio. Reportar como limitação, não como causalidade estabelecida. |
| **Overfitting** com muitos parâmetros novos | Guardrail da raiz continua: walk-forward purgado, Deflated Sharpe, contabilizar nº de configurações testadas. |
| **Scope creep afundar a entrega** | Respeitar os tiers. Tier 3 é descartável por design. |
| **Vantagem que NÃO herdamos** | O modelo original assume opções americanas líquidas. Opção de dólar da B3 não é. Assumir isso explicitamente no relatório e explicar a compensação via Focus. |

## Estrutura do módulo

```
credibility/
  data_focus.py      # Focus/BCB: πᵉ, dispersão, meta (API Olinda/SGS) -> parquet
  data_gap.py        # hiato do produto (HP + Hamilton como robustez)
  copom_text.py      # scraping/parse de comunicados, atas e discursos
  tone.py            # classificação hawkish/dovish (LLM zero-shot + baseline dicionário + cache)
  credibility.py     # θt: baseline (desvio Focus) e versão estrutural (estado latente)
  barro_gordon.py    # calibração do jogo; equilíbrios Nash vs Stackelberg
  rnd.py             # Tier 3: momentos BKM; RND via Breeden-Litzenberger (opcional)
  prompts/           # prompts versionados do classificador
```

Testes deste módulo seguem a convenção da raiz: ficam no `tests/` de nível superior do
projeto (não em `credibility/tests/`), um arquivo por módulo (`test_data_focus.py`, etc.).

## Convenções específicas desta fase

- Séries macro têm **defasagem de publicação**. Usar sempre a data de *divulgação*, nunca a data de
  *referência*, ao montar features. Violar isso é look-ahead bias disfarçado — o erro mais provável desta fase.
- Filtro HP tem problema de ponta severo: **nunca** rodar HP no full sample e usar como se fosse
  informação disponível em tempo real. Recalcular expanding-window.
- Focus tem revisões: preservar o vintage original.
- Todo score de LLM vai para cache versionado em `data/processed/tone_cache/`.
- Manter a ablação como padrão: **toda métrica nova é reportada com vs. sem a camada de credibilidade.**

## Referências

- Barro & Gordon (1983) e extensões recentes de reputação/credibilidade.
- Bakshi, Kapadia & Madan (2003) — momentos model-free implícitos em opções.
- Breeden & Litzenberger (1978) — densidade risco-neutro (Tier 3).
- Nakamura & Steinsson — identificação de alta frequência de choques de política monetária.
- Literatura brasileira de índices de credibilidade do BCB baseados em Focus vs. meta
  (de Mendonça é a referência clássica) — usar como benchmark de validação de θt.
- Comunicação de BC e NLP: "hawkish dovish classification FOMC", "Copom comunicação análise textual".
- Hamilton (2018) — alternativa ao filtro HP para hiato do produto.
- Repositórios: BDTD, RePEc/IDEAS, SSRN, arXiv q-fin, FGV EPGE/EESP-USP.
