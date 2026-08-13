# CLAUDE.md — Robô [definir nome] · Desafio Itaú Asset Quant AI 2026

## Visão geral

Estratégia quantitativa de **volatilidade em opções de dólar (USD/BRL)**.

**Tese central:** o mercado precifica a vol futura do dólar na **volatilidade implícita (IV)** das opções.
Construímos uma **previsão de volatilidade realizada (RV)** superior incorporando fluxo de notícias
(NLP / transfer learning). Operamos o *spread*: **compramos vol** quando previsão de RV > IV de mercado,
e **vendemos vol** quando previsão de RV < IV.

**Objetivo do projeto: rigor metodológico e clareza, NÃO retorno máximo.** O edital não pontua
desempenho histórico — pontua conceito, modelagem, mitigação de vieses e clareza. Priorize sempre
correção metodológica sobre "número bonito de backtest".

## Contexto da equipe (importante para decisões técnicas)

- Forças: **Python/engenharia** (forte), **quant/opções — gregas, vol** (bom).
- Ponto mais fraco: **Machine Learning**.
- Implicação prática: prefira **soluções prontas / transfer learning** (FinBERT-PT-BR via `pipeline`)
  a treinar modelo do zero. Comece com modelos de previsão **simples e defensáveis** (HAR-RV, GARCH)
  antes de considerar gradient boosting. Não introduza deep learning sem necessidade clara.

## Stack

- Python 3.11+ (venv ou uv)
- pandas, numpy, scipy, statsmodels, `arch` (GARCH), scikit-learn, lightgbm (opcional)
- transformers + torch (FinBERT-PT-BR)
- Dados persistidos em **parquet** (pyarrow)
- Testes: pytest

## Estrutura do repositório

```
data/        # coletores de dados -> parquet limpo (raw/ e processed/)
sentiment/   # wrapper FinBERT-PT-BR -> índice diário de sentimento/estresse
vol/         # implied.py (IV), realized.py (RV), forecast.py (HAR/GARCH + versão c/ notícia)
strategy/    # regra de sinal (RV_forecast vs IV) + sizing por vega + estrutura de opção
backtest/    # walk-forward purgado, custos/slippage, métricas robustas
report/      # gráficos e métricas para pré-relatório e entrega final
tests/       # pytest por módulo
```

## Fontes de dados (todas gratuitas)

- **Futuro de dólar (DOL) — FONTE OFICIAL DE PREÇO:** arquivos BVBG-086 da B3 (`data/b3_futures.py`),
  2.135 pregões de 2018-01 a 2026-08. É o instrumento sobre o qual a opção é escrita.
- **NÃO usar `yfinance` para USD/BRL.** O `close` de `BRL=X` é snapshot no limite do dia (correlação
  0,99998 com a abertura), não fechamento de pregão — o retorno sai defasado 1 dia. Validado contra o
  PTAX. Só `high`/`low` é confiável (usado pelo Parkinson). Vale só para VIX/DXY.
- **Câmbio/PTAX:** API SGS do Banco Central — usado como árbitro independente de validação.
- **Superfície de IV de dólar:** publicada pela B3, sobrescrita diariamente. **Só existe 1 snapshot
  arquivado (2026-07-21) + 1 recuperado do Internet Archive (2026-04-29).** Arquivar diariamente é
  ganho puro e barato; cada dia sem coletar é perdido para sempre.
- **IV própria (implementado):** `data/b3_options.py` + `vol/black76.implied_vol`.
  **ATENÇÃO — o caminho originalmente previsto está FECHADO:** a B3 publica preço de ajuste (`AdjstdQt`)
  para **zero** das milhares de séries de opção de dólar, verificado em 2019, 2022 e 2026. Não é mudança
  recente, é estrutural. O caminho que funciona é inverter Black-76 sobre o **preço negociado** —
  7.346 negócios em 244 pregões (2018-2023), validado contra a superfície oficial dentro de 0,25-0,78 p.p.
  A liquidez caiu 15× no período (45 séries negociadas/dia em 2019 → 3 em 2026).
- **Notícias (global):** GDELT (API gratuita) — cobertura de 2023-07 a 2026-07.
- **Notícias (BR):** atas/comunicados do COPOM e BCB; RSS de veículos financeiros.
- **Bloomberg:** NÃO fazer scraping (viola ToS). Usar apenas export do terminal, quando disponível na
  competição, como enriquecimento/validação — nunca como dependência central.
- **Lead não explorado:** a CME lista opção de real (contrato 6L) e publica ajuste para todas as séries,
  inclusive as que não negociam — seria IV de série longa, sem a limitação da B3.

## Conceitos de domínio

- **IV** = expectativa de vol do mercado (implícita nas opções). **RV** = vol realizada (histórica/futura).
- Opção de dólar é opção **sobre futuro** → apreçar com **Black-76**, não Black-Scholes spot.
- Inverter a IV numericamente (Brent/bisseção) a partir do preço de ajuste.
- Isolar exposição a vol via **delta-hedge**; dimensionar posição por **vega**.
- Estrutura de opção inicial: **straddle ATM** (mais simples). Evoluir depois (strangle, condor).

## Regras invioláveis (guardrails)

- **Sem look-ahead bias.** A decisão de hoje usa apenas dados até o fechamento de hoje; ordem executada em D+1.
- **Walk-forward purgado com embargo** (López de Prado). NUNCA usar train/test split aleatório em série temporal.
- **Custos realistas sempre.** Opção de dólar tem spread bid-ask largo — incluir custos e slippage em todo backtest.
- **Controle de overfitting.** Registrar o nº de configurações testadas e usar **Deflated Sharpe Ratio**.
  Não repetir o grid search cego da referência de 2023.
- **Ablação obrigatória.** Toda avaliação da previsão de RV compara **com notícia vs baseline sem notícia**.
  Esse contraste é a prova causal do valor da IA — é central para a nota.
- Séries temporais com timezone explícito (`America/Sao_Paulo`) e alinhadas por dia útil da B3.

## Convenções de código

- Identificadores em inglês; docstrings/comentários podem ser em português.
- Cada coletor é **idempotente e cacheável**: salva em `data/raw/`, versão limpa em `data/processed/`.
- Separar I/O de lógica; funções puras onde possível. Type hints. Formatação com black/ruff.
- `pytest` verde por módulo antes de avançar para o próximo.
- Nenhum segredo no repo. Chaves/tokens via `.env` (python-dotenv), com `.env.example` versionado.

## Ordem de execução (fases)

1. `data/` — PTAX + futuro + IV + notícias, com testes.
2. `vol/` — `implied` e `realized` funcionando (IV pronta da B3 primeiro; Black-76 próprio depois).
3. `sentiment/` — índice diário via FinBERT-PT-BR.
4. `vol/forecast` — baseline HAR/GARCH → versão com notícia → **estudo de ablação**.
5. `strategy/` — regra de sinal + sizing por vega.
6. `backtest/` — walk-forward purgado, custos, métricas.
7. `report/` — gráficos e métricas.

Não construir tudo de uma vez. Validar módulo a módulo.

## Estado dos achados (ler antes de propor experimento novo)

- **Diagnóstico central: descasamento de horizonte.** Previsibilidade existe em h=1 e decai até
  desaparecer em h=21, o prazo de que a estratégia precisa. Autocorrelação da variância em lag 21:
  0,025 no futuro da B3. Não é falta de feature.
- **Único resultado positivo:** HAR-RV puro em h=1 bate a persistência com R² absoluto positivo em
  2021-2022 (4/5 folds) e 2023-2026 (5/5 folds). Diebold-Mariano p=0,17 e p=0,11 — **sugestivo, não
  estabelecido**. Nenhuma camada de informação melhora isso.
- **Todas as camadas falharam**, testadas em h=21 **e** reavaliadas em h=1 e h=5: notícia (GDELT),
  risco fiscal, FinBERT, credibilidade (Barro-Gordon), risco global, correção de rolagem.
- **XGBoost perde do HAR em todo horizonte ≥ 3**, com a distância crescendo conforme as observações
  independentes caem. Desenho e referências em `vol/ml_forecast.py`.
- **Consultar `report/run_report.py:CONFIGS_TESTED` (31 registros) antes de propor qualquer teste** —
  cada entrada traz veredito, magnitude, consistência entre folds e as ressalvas.

### Armadilhas já encontradas (todas custaram um resultado falso)

1. **Dados parciais** — concluir antes de a coleta terminar inverteu um veredito.
2. **Janelas sobrepostas** — inflam p-valor; sempre reportar a versão independente.
3. **Seleção pós-hoc** — janela/configuração escolhida depois de ver o resultado. Testar a vizinhança.
4. **R² entre amostras diferentes** — `ss_tot` cresce com a heterogeneidade da janela; comparação
   entre modelos exige **período de teste idêntico**.
5. **Agregação criando padrão** — monotonia na amostra completa que não existe em nenhum fold.
6. **Escala em modelo de gradiente** — `gblinear` com features de ordem 1e-5 sai subajustado e dá R²
   negativo; parece resultado, é artefato. Padronizar (ajustando o scaler só no treino).

### Critério de três portões (aplicar a qualquer camada nova)

Δ R² > 0 **e** ≥4/5 folds melhorando **e** Diebold-Mariano p < 0,05 em janelas **independentes**.
Abaixo disso: "não estabelecido". Declarar o critério **antes** de rodar.

## Comandos

- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Testes: `pytest -q` (372 testes)
- Lint: `python -m ruff check .`
- Veredito: `python -m report.run_report`
- Relatório: `python -m relatorio.gerar_dados && python -m relatorio.gerar_relatorio`

## Datas do desafio

- Pré-relatório: **31/07/2026** · Entrega final: **17/08/2026**
- Quartas: 31/08 · Semifinal: 09/09 · Final (presencial, Faria Lima): 26/09

## Referências

- FinBERT-PT-BR: https://huggingface.co/lucas-leme/FinBERT-PT-BR
- López de Prado, *Advances in Financial Machine Learning* (walk-forward purgado, embargo, Deflated Sharpe)
- B3 — Manual de Apreçamento de Opções (metodologia de vol de dólar) e boletins diários de mercado
- GDELT Project — base global de notícias (API gratuita)
- Banco Central — API SGS (PTAX) e comunicados/atas do COPOM

<!-- Nota p/ mantenedores: se colocarem o doc da proposta no repo, referenciar com @Proposta_QuantAI_Ultron.docx -->
