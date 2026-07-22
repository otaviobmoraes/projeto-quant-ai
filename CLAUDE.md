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

- **Câmbio/PTAX:** API SGS do Banco Central (série PTAX) — oficial e estável. Localizar o código da série.
- **Futuro de dólar (DOL/WDO):** market data público da B3; `yfinance` (`BRL=X`) como apoio.
- **Superfície de IV de dólar:** publicada pela B3 (preços referenciais, pool de informantes às 18h). Baseline pronto.
- **IV própria (diferencial):** boletins diários da B3 (arquivo **BD_Arbit**) → preços de ajuste das opções →
  **inverter via Black-76** (numérico). Fazer como diferencial de rigor, além do baseline pronto.
- **Notícias (global):** GDELT (API gratuita) — milhares de manchetes, já traz score de tom.
- **Notícias (BR):** atas/comunicados do COPOM e BCB; RSS de veículos financeiros.
- **Bloomberg:** NÃO fazer scraping (viola ToS). Usar apenas export do terminal, quando disponível na
  competição, como enriquecimento/validação — nunca como dependência central.

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

## Comandos (preencher conforme o repo evolui)

- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Testes: `pytest -q`
- (adicionar lint/run conforme surgirem)

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
