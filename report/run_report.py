"""Script de conveniencia: roda a avaliacao final (walk-forward purgado),
gera todos os graficos/resumo do relatorio de uma vez e imprime um veredito
em linguagem simples sobre o modelo.

Uso:
    .venv\\Scripts\\python.exe -m report.run_report   (Windows)
    .venv/bin/python -m report.run_report             (bash)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew

from backtest import ablation, engine
from backtest.metrics import deflated_sharpe_ratio, pooled_oos_metrics
from backtest.walk_forward import purged_walk_forward_splits, purged_walk_forward_splits_by_step
from credibility import ablation as credibility_ablation
from data.gdelt_news import (
    TONE_PROCESSED_PATH,
    VOLUME_PROCESSED_PATH,
    fiscal_risk_surprise,
    load_fiscal_risk_series,
)
from data.iv_surface import PROCESSED_PATH as IV_PROCESSED_PATH
from data.ptax import PROCESSED_PATH as PTAX_PROCESSED_PATH
from report import plots, summary
from sentiment.daily_index import FISCAL_SENTIMENT_PROCESSED_PATH
from strategy import signal, sizing
from vol import implied, realized
from vol.forecast import BASELINE_FEATURES, NEWS_FEATURES, build_dataset, fit_har, persistence_forecast

OUT_DIR = Path(__file__).resolve().parent / "output"

CONFIGS_TESTED = [
    "Fase 4 (preliminar): split unico 80/20, noticia bruta, RV em nivel (proxy retorno^2)",
    "Fase 4 (refinado): CV expansiva 5 folds, noticia suavizada 21d, log-RV (proxy retorno^2)",
    "Fase 6 (final v1): walk-forward PURGADO 5 folds + embargo 5d, noticia suavizada 21d, log-RV (proxy retorno^2)",
    "Diagnostico: persistencia pura (sem modelo) -- tambem R2 negativo, confirma que o problema nao e a regressao",
    "Diagnostico: GARCH(1,1) em retorno^2 -- pior que HAR-RV simples",
    "Diagnostico: HAR-RV com estimador Parkinson (OHLC) -- RMSE menor em TODOS os folds, adotado como baseline oficial",
    "Fase 6 (final v2, oficial): walk-forward PURGADO 5 folds + embargo 5d, noticia suavizada 21d, log-RV, variancia Parkinson",
    "Credibilidade Tier 1: theta_baseline + dispersao (Focus/meta), log-RV, variancia Parkinson",
    "Risco fiscal (GDELT, % de cobertura, nivel bruto) -- nao ajuda",
    "Risco fiscal (surpresa/z-score, janela 63d) -- nao ajuda",
    "Risco fiscal (surpresa/z-score, janela 21d) -- efeito marginal, dentro do ruido",
    "Diagnostico R2: media por fold vs POOLED -- media por fold e instavel com "
    "folds pequenos (explode negativo); adotado R2 pooled como metrica oficial",
    "Sweep de esquema de walk-forward (5/8/10 folds fixos + 6 variantes de "
    "step_size) com R2 pooled -- so 1 config bateu persistencia (378d/42d), "
    "vizinhos proximos NAO bateram -- rejeitado como selecao de config por sorte, "
    "mantido 5 folds fixos como esquema oficial",
    "HAR-RV + termo de leverage (rv_d x indicador de retorno negativo) -- sem "
    "efeito (multicolinearidade com rv_d/rv_w/rv_m)",
    "Risco fiscal REFINADO v2 (surpresa + sentimento FinBERT-PT-BR, 15/15 "
    "janelas coletadas, 22 dias com manchete real) -- com dado parcial (12/15) "
    "parecia ajudar (R2 +0.073, 3 de 5 folds), mas com a coleta completa "
    "INVERTEU (R2 -0.568) -- artefato de amostra pequena, descartado",
    "Pesquisa na literatura (Kambouroudis et al. 2021; Barndorff-Nielsen, "
    "Kinnebrock & Shephard 2010) + 3 extensoes testadas com R2 pooled: "
    "overnight return (gap fechamento->abertura), leverage via semivariancia "
    "(rv_d_pos/rv_d_neg, decomposicao aditiva em vez da interacao "
    "multiplicativa) e ensemble (media das 3 variantes) -- nenhuma moveu o "
    "R2 de forma perceptivel (delta entre +0.004 e -0.001, dentro do ruido); "
    "todas continuam perdendo pra persistencia pura. "
    "*** RESSALVA (descoberta depois): esse teste de overnight/leverage e "
    "INVALIDO. Os bars de BRL=X do yfinance tem open ~= close (mesmo "
    "snapshot: |close-open| medio = 0.00043 vs gap entre barras = 0.03082, "
    "72x maior), entao a feature 'overnight' media o retorno do DIA (corr "
    "0.998 com retorno diario ao quadrado), nao um gap overnight -- e o "
    "sinal do retorno usado na semivariancia vem defasado 1 dia. Refazer "
    "com OHLC real do futuro B3 (data/b3_futures.py). ***",
    "Risco global exogeno (VIX + DXY via yfinance, primeira feature que NAO "
    "deriva do proprio preco/imprensa do USD/BRL) -- R2 pooled: baseline "
    "-0.011, so VIX -0.020 (pior), so DXY -0.012 (sem efeito), VIX+DXY "
    "-0.020 -- nenhuma ajuda; persistencia continua em +0.134 no mesmo "
    "periodo. 12a tentativa consecutiva sem melhorar o R2 nesse dataset.",
    "DECISAO: persistencia pura (rv_m) adotada como previsao OFICIAL da "
    "estrategia (secao 5) no lugar do HAR-RV -- bateu o HAR-RV em TODAS as "
    "13 comparacoes de R2 pooled feitas neste projeto (R2 +0.126 vs -0.510 "
    "no esquema oficial). O backtest ilustrativo (Sharpe) PIOROU com essa "
    "troca (1.087 -> 0.014) apesar do R2 melhorar -- diagnostico: "
    "RV_previsto (persistencia) e IV_proxy tem correlacao ~0.994 (IV_proxy "
    "= RV_trailing x premio fixo, persistencia = RV_trailing quase igual), "
    "entao o spread que decide compra/venda e dominado por uma constante "
    "multiplicativa, nao por sinal -- R2 mede acerto medio em TODOS os "
    "dias, taxa de acerto mede so o subconjunto pequeno e nao-aleatorio de "
    "dias em que esse spread quase-constante cruzou a banda por ruido de "
    "curto prazo. R2 (pooled, medido contra o valor real) continua a "
    "metrica confiavel; Sharpe do backtest ilustrativo nao deveria ser "
    "usado pra escolher entre modelos ate haver IV real historica.",
    "*** CORRECAO IMPORTANTE (descoberta na revisao final): a afirmacao "
    "'persistencia bate o HAR-RV', repetida nos itens acima, era ARTEFATO DA "
    "FONTE DEFEITUOSA. Na fonte correta (futuro da B3) o HAR-RV vence a "
    "persistencia em 5 dos 6 horizontes (h=1,3,10,15,21); a persistencia so "
    "vence em h=5. Na fonte yfinance ela vencia em 6 de 6. Consequencia: a "
    "decisao de adotar persistencia como previsao oficial e o experimento de "
    "correcao de residuo (item seguinte) foram construidos sobre a base "
    "errada -- refazer sobre HAR-RV. ***",
    "Persistencia + CORRECAO DE RESIDUO (regressao no residuo target-"
    "persistencia, ver backtest.engine.generate_residual_corrected_forecast) "
    "-- testadas as 6 camadas ja avaliadas contra o HAR-RV (so rv_d/rv_w/rv_m, "
    "noticia, credibilidade, risco fiscal v2, overnight, leverage, VIX+DXY) "
    "como corretoras da persistencia em vez de substitutas dela. TODAS "
    "pioraram o R2 pooled vs persistencia pura, sem excecao (delta entre "
    "-0.133 e -1.851) -- persistencia nao tem parametro nenhum (zero risco "
    "de overfitting); qualquer correcao via regressao introduz variancia de "
    "estimacao sem sinal real pra compensar, piorando a previsao. 19a "
    "tentativa consecutiva sem melhorar o modelo nesse dataset -- "
    "simplicidade (persistencia pura, sem nenhuma camada) e o resultado "
    "mais robusto encontrado.",
    "VALIDACAO DE FONTE DE PRECO (futuro de dolar da B3, BVBG-086, vs "
    "yfinance BRL=X, arbitrado pelo PTAX do BCB): o AJUSTE do futuro B3 bate "
    "com o PTAX no MESMO dia (corr 0.673 na amostra completa de 830 pregoes; "
    "0.768 na janela inicial de 78 dias, k=0); ja o `close` do yfinance so "
    "alinha com defasagem de 1 dia (corr 0.528 em k=+1 vs 0.391 em k=0). "
    "Causa: os bars de FX do yfinance tem open ~= close (correlacao 0.99998 "
    "entre eles) -- nao e fechamento de fim de pregao, e um snapshot no "
    "limite do dia. IMPACTO: (a) o baseline OFICIAL usa Parkinson, que so "
    "depende de high/low, e ESSES estao corretamente datados (PTAX cai "
    "dentro do range do mesmo dia em 98.9% dos pregoes) -- as conclusoes "
    "centrais do projeto sobrevivem; (b) features derivadas de `close` "
    "(overnight, sinal da semivariancia) e os precos de entrada/saida do "
    "backtest estavam defasados 1 dia. Migracao para o futuro da B3 "
    "(instrumento efetivamente negociado, com OHLC e ajuste oficiais).",
    "*** RESULTADO CENTRAL REVISTO apos migrar pro futuro da B3 (830 "
    "pregoes, 2023-04 a 2026-08): o R2 POSITIVO NAO SOBREVIVEU. Com "
    "Parkinson -- yfinance: HAR-RV -0.066 / persistencia +0.126 (persistencia "
    "ganha); futuro B3: HAR-RV -0.255 / persistencia -0.367 (AMBOS "
    "negativos). Cross-check com estimador close-to-close sobre o ajuste "
    "oficial confirma: HAR-RV -0.155 / persistencia -0.766. Causa "
    "diagnosticada: autocorrelacao da variancia diaria em lag 5/21 e de "
    "0.096/0.025 no futuro B3 contra 0.190/0.074 no yfinance -- o alvo de "
    "21 dias depende dessa memoria longa, que existe no spot 24h mas nao no "
    "futuro. Duas leituras nao totalmente separaveis: (a) o range de 24h "
    "media mais do processo e reduz ruido de medicao (efeito estatistico "
    "legitimo); (b) os bars anomalos do yfinance inflam a autocorrelacao "
    "(artefato). Como a estrategia negocia opcao SOBRE O FUTURO, o alvo "
    "economicamente correto e a RV do futuro -- e nela nada funciona. ***",
    "Varredura de horizonte na fonte B3 (R2 pooled, 5 folds purgados): "
    "h=1 HAR-RV +0.049 / persistencia -0.022; h=3 -0.027/-0.037; h=5 "
    "-0.102/-0.075; h=10 -0.155/-0.158; h=15 -0.197/-0.245; h=21 "
    "-0.255/-0.367. Ou seja: existe previsibilidade fraca porem REAL em "
    "h=1 (e o HAR-RV bate a persistencia la), decaindo monotonicamente ate "
    "ficar bem negativa em h=21. Consistente com Corsi (2009), que desenhou "
    "o HAR-RV para horizonte curto. DIAGNOSTICO: ha descasamento entre onde "
    "existe sinal (1 dia) e o horizonte que a estrategia precisa (21 dias, "
    "prazo da opcao) -- nao e falta de feature, e o horizonte.",
    "CREDIBILIDADE testada na dimensao que a teoria PREVE (assimetria, nao "
    "nivel): a ablacao anterior testou theta_t contra RV futura, um alvo de "
    "NIVEL, mas Barro-Gordon preve efeito ASSIMETRICO (alargamento da cauda "
    "direita/skew). O nulo anterior era evidencia fraca contra a teoria, nao "
    "refutacao. O teste correto exigiria skew IMPLICITO -- inviavel, so ha 1 "
    "dia de superficie de IV. Usada a contraparte realizada "
    "(vol.realized.forward_realized_skewness). Direcao esperada: theta baixo "
    "-> skew alta, ou seja correlacao NEGATIVA. Resultado com janelas "
    "INDEPENDENTES (nao sobrepostas): h=10 r=+0.172 p=0.140; h=21 r=-0.002 "
    "p=0.993; h=42 r=-0.150 p=0.552 -- nenhum significativo, e o sinal nem e "
    "consistente entre horizontes. NOTA METODOLOGICA: com janelas SOBREPOSTAS "
    "os MESMOS dados dariam p<0.001 em h=10 e h=42 -- p-valor inflado por "
    "observacoes que compartilham quase todos os retornos. Status da camada: "
    "de 'refutada' para 'testada na dimensao certa, sem evidencia detectavel "
    "nesta amostra'.",
    "CICLO DE ROLAGEM do futuro (vol/roll.py, backtest/roll_ablation.py). "
    "Motivacao: os contratos rolam a cada ~20,2 pregoes, praticamente o mesmo "
    "periodo do horizonte alvo (21 dias), e a vol de Parkinson medida cai "
    "~15% conforme o contrato envelhece -- sentido INVERSO ao efeito "
    "Samuelson, sugerindo artefato de medicao (a liquidez migra para o "
    "vencimento seguinte, menos negocios exploram o range, ln(high/low) "
    "encolhe). Hipotese: dente de serra deterministico com periodo ~= "
    "horizonte, que o HAR nao captura. TRES RESULTADOS, todos negativos: "
    "(a) DESCRITIVO -- dessazonalizar quase nao move a autocorrelacao "
    "(lag 1: 0.2988 -> 0.2982; lag 21: 0.0248 -> 0.0222), ou seja a "
    "sazonalidade de rolagem NAO explica a memoria ausente; "
    "(b) PREVISAO (walk-forward purgado, fator estimado so no treino, alvo "
    "sempre da variancia ORIGINAL para manter o R2 comparavel) -- piora em "
    "todos os horizontes: variancia dessazonalizada da delta de -0.045 (h=1), "
    "-0.083 (h=5), -0.079 (h=10), -0.184 (h=21); posicao no ciclo como "
    "feature explicita da -0.031 (h=21) e +0.0018 (h=1, 4/5 folds, magnitude "
    "dentro do ruido -- NAO tratado como achado); "
    "(c) DIAGNOSTICO -- o fator sazonal NAO e estavel entre folds (faixa "
    "0-6d varia de 1.112 a 0.860, amplitude 0.253; nao e monotonico em "
    "NENHUM fold isolado). A monotonia bonita da amostra completa "
    "(0.884 -> 1.205) e artefato de agregacao. So a faixa '28+ dias' "
    "(contrato recem-promovido) e consistente: fator 1.28-1.47 nos 5 folds. "
    "Explica por que a correcao piora: dividir por um fator que oscila "
    "+-0.25 entre folds injeta variancia de estimacao sem remover sinal "
    "real -- mesmo mecanismo ja observado na correcao de residuo. "
    "CONCLUSAO: a rolagem e um artefato de medicao REAL e documentado, mas "
    "NAO e a causa do descasamento de horizonte. Hipotese eliminada; reforca "
    "a explicacao alternativa (ruido do estimador de Parkinson sobre OHLC "
    "diario), sustentada pela autocorrelacao de lag 1 = 0.30 quando RV limpa "
    "de cambio, medida com dados intraday, costuma ficar em 0.6-0.8.",
    "EXTENSAO DA AMOSTRA: 830 -> 2135 pregoes (2018-01-02 a 2026-08-05, "
    "coleta retroativa dos boletins BVBG-086 da B3). Janelas independentes "
    "de 21 dias: 39 -> 101. Validacao do merge: a janela 2023-04-10 em "
    "diante reproduz EXATAMENTE o -0.2550 ja publicado. "
    "*** ARMADILHA EVITADA -- LER ANTES DE CITAR QUALQUER NUMERO DAQUI: *** "
    "na amostra completa o R2 pooled em h=21 salta de -0.255 para +0.381 e "
    "a autocorrelacao em lag 21 de 0.025 para 0.140. NENHUM DOS DOIS E "
    "MELHORA DE PREVISAO -- os dois sao artefato de MISTURA DE REGIMES. "
    "Tres diagnosticos independentes: "
    "(a) a PERSISTENCIA, que nao tem nenhum parametro, tambem saltou "
    "(-0.367 -> +0.365). Um modelo sem parametros nao aprende nada; se o R2 "
    "dele muda 0.73, o que mudou foi o denominador, nao a previsao. "
    "(b) Fixando o periodo de TESTE (485 pontos, 2024-07 a 2026-07, mesmos "
    "folds) e variando so o TREINO: treinar com 8.5 anos em vez de 1 da "
    "RMSE 2.7429 vs 2.7808 (-1.4%), MAE PIOR (2.2478 vs 2.0910) e R2 "
    "-0.1221 vs -0.1533 -- continua NEGATIVO. O ganho real de mais treino e "
    "marginal. (O MAE piorar enquanto o RMSE melhora sugere que treinar num "
    "periodo de vol mais alta -- alvo medio 13.1/12.4 em 2018-22 contra "
    "11.1 em 2023-26 -- vicia a previsao para cima numa epoca mais calma.) "
    "(c) A autocorrelacao de CADA subperiodo e MENOR que a da amostra "
    "completa (lag 21: 2018-19 +0.110; 2021-22 -0.018; 2023-26 +0.025; "
    "completa +0.140). Diferenca de nivel entre regimes gera memoria longa "
    "espuria. "
    "CONCLUSAO: o diagnostico de descasamento de horizonte SOBREVIVE e fica "
    "MAIS FORTE -- agora sustentado por 101 janelas independentes em vez de "
    "39. LICAO METODOLOGICA para o relatorio: R2 nao e comparavel entre "
    "amostras de composicao diferente, porque o denominador (ss_tot) cresce "
    "com a heterogeneidade da janela. Comparacao entre modelos exige "
    "periodo de teste identico.",
    "IV PROPRIA RECONSTRUIDA DE NEGOCIOS (data/b3_options.py + "
    "vol.black76.implied_vol). O caminho previsto no CLAUDE.md -- inverter "
    "Black-76 do preco de AJUSTE das opcoes -- esta FECHADO: verificado em "
    "2019, 2022 e 2026, a B3 publica ajuste para ZERO das milhares de series "
    "de opcao de dolar (nao e mudanca recente, e estrutural). Caminho que "
    "funcionou: inverter sobre o preco EFETIVAMENTE NEGOCIADO. "
    "VALIDACAO contra a superficie oficial da B3 (unico dia disponivel, "
    "2026-07-21, negocios de 2026-07-23): 3 series batem dentro de 0.25 a "
    "0.78 ponto percentual, com estrutura a termo coerente (10.6% em 40 "
    "dias -> 13.0% em 193). "
    "AMOSTRA FINAL: 7.346 negocios em 244 pregoes (2018-01 a 2023-03), "
    "coletados com passo de 21 pregoes -- desenho escolhido para maximizar "
    "JANELAS INDEPENDENTES por hora de coleta, ja que dias consecutivos "
    "produzem janelas sobrepostas (armadilha ja registrada neste projeto). "
    "Filtros pre-fixados: >=2 negocios na serie, IV entre 4% e 60%, "
    "vencimento 15-60 dias corridos, |moneyness| <=3%. "
    "*** FALSO POSITIVO NUMERO 4 -- REGISTRADO PARA NAO SE REPETIR: *** "
    "com apenas o bloco denso de 2018 (n=6 janelas independentes) a IV "
    "aparecia MUITO superior (R2 +0.0903 contra -0.5473 da persistencia) e "
    "isso foi reportado como se invertesse a premissa da tese. NAO "
    "SOBREVIVEU. Com 57 janelas independentes cobrindo 2018-2023: IV "
    "-0.0683 (RMSE 3.918) contra persistencia +0.1455 (RMSE 3.504) -- a "
    "PERSISTENCIA GANHA. E ganha em TODOS os subperiodos: 2018 (n=10) IV "
    "-1.086 vs +0.092; 2019-2020 (n=21) IV +0.239 vs +0.262; 2021-2023 "
    "(n=26) IV -0.763 vs -0.464. "
    "DIAGNOSTICO DA INSTABILIDADE: para o MESMO periodo de 2018, selecionar "
    "as janelas independentes por POSICAO (n=6) da IV +0.09 e por DATA "
    "(n=10) da IV -1.09. Com n~10 o R2 oscila mais de um ponto inteiro "
    "conforme a escolha dos pontos -- nesse regime de amostra o R2 nao e "
    "estatistica estavel e nao deve ser usado para decidir nada. "
    "O QUE SOBREVIVE (e e o resultado de fato): a IV CARREGA INFORMACAO mas "
    "e ENVIESADA. Correlacao IV x RV futura +0.568 na amostra toda e "
    "estavel entre subperiodos (+0.591 / +0.611 / +0.361). A regressao "
    "RV_futura = 6.92 + 0.504 x IV tem R2 in-sample 0.3221 -- coeficiente "
    "0.50 em vez de 1 significa que quando a IV sobe 1 ponto a RV sobe "
    "meio. E o achado classico da literatura (Christensen & Prabhala 1998; "
    "Poon & Granger 2003): IV informativa, com vies de nivel. Por isso a IV "
    "CRUA perde da persistencia como previsao pontual apesar de ter "
    "conteudo preditivo -- usa-la exigiria recalibrar, o que com 57 pontos "
    "e fragil. "
    "PREMIO DE RISCO medido (este SIM se sustenta e importa): "
    "IV/RV_futura = 1.079 e IV/RV_trailing = 1.129, com IV > RV futura em "
    "59% dos dias -- contra o multiplicador 1.29 CONSTANTE que o backtest "
    "ilustrativo assume, calibrado num unico dia de 2026. A calibragem de "
    "um dia superestima o premio em ~15%.",
    "BACKTEST COM IV REAL (primeira vez possivel no projeto). 9 configuracoes "
    "(IV real / proxy 1.29 / proxy 1.08, x bandas 0.5/1.0/2.0): TODAS com "
    "Sharpe negativo (-0.007 a -0.378). Ganho QUALITATIVO: com IV real o "
    "sinal e bilateral (4 long / 17 short); com o proxy e degenerado "
    "(0 long / 28 short), confirmando a colinearidade de 0.994 ja documentada. "
    "DECOMPOSICAO DE CUSTO (o diagnostico que importa): a estrategia perde "
    "ANTES dos custos -- PnL bruto -45.024 com spread_pct=0. A hipotese de "
    "que o premio de 1.08 nao cobriria o spread de 5% estava ERRADA: os "
    "custos agravam (-62.796 liquido) mas nao explicam. O sinal erra o lado: "
    "as 17 vendas de vol perderam -55.307 brutos e as 4 compras ganharam "
    "+10.283. Coerente com uma previsao de RV de R2 negativo comparada "
    "contra uma IV informativa. LIMITE: 21 trades (sobreposicao entre "
    "previsao OOS e IV real e de so 46 dias).",
    "MACHINE LEARNING (XGBoost) -- vol/ml_forecast.py, backtest/ml_ablation.py. "
    "REFERENCIAS que fundamentam o desenho: Christensen, Siggaard & Veliyev "
    "(2023), J. of Financial Econometrics 21(5):1680-1727 -- ML supera a "
    "linhagem HAR mesmo usando so as defasagens diaria/semanal/mensal da RV, "
    "e com AJUSTE MINIMO de hiperparametros; Teller, Pigorsch & Pigorsch "
    "(SSRN 4267541) -- XGBoost supera HAR e LSTM em 1 passo a frente, mas em "
    "horizontes longos os base learners LINEARES superam os de arvore. "
    "DESENHO: mesmas features do HAR (isola a forma funcional), mesmo alvo "
    "em log, mesmos folds purgados, hiperparametros FIXOS pre-registrados "
    "(sem grid search -- protege o DSR), dois base learners. "
    "BUG ENCONTRADO E CORRIGIDO: gblinear ajusta por descida de gradiente e e "
    "sensivel a escala; com features de variancia diaria na ordem de 1e-5 "
    "saia gravemente subajustado (R2 negativo em TODOS os horizontes). Com "
    "StandardScaler ajustado so no treino, passa a 0.25-0.33. O R2 negativo "
    "inicial era artefato de escala, nao resultado. "
    "RESULTADO: o HAR-RV (OLS) vence em TODOS os horizontes >= 3. R2 pooled "
    "em h=21: HAR 0.3810, XGB linear 0.2915, XGB arvore 0.2316, persistencia "
    "0.3654. Em h=1 o XGB linear fica marginalmente a frente (0.2510 vs "
    "0.2433) mas em apenas 2 de 5 folds -- NAO tratado como achado. "
    "PADRAO INFORMATIVO: a distancia do XGB-arvore para o HAR CRESCE "
    "monotonicamente com o horizonte (-0.024 em h=1 ate -0.149 em h=21), "
    "acompanhando a queda de observacoes independentes de 1.760 para ~101. "
    "Isso CONTRARIA Christensen et al. (que acham ganhos MAIORES em "
    "horizontes longos) e a explicacao e o contexto: eles usam painel de "
    "constituintes do Dow Jones com RV intradiaria -- dezenas de milhares de "
    "observacoes; aqui e um ativo, Parkinson sobre OHLC diario. "
    "CONCLUSAO: a Secao 8.3 do relatorio afirmava que gradient boosting "
    "'seria adicionar capacidade onde ja demonstramos ausencia de sinal'. "
    "Isso deixa de ser afirmacao e vira DEMONSTRACAO medida.",
    "PREVISIBILIDADE EM h=1 (o unico resultado POSITIVO do projeto). Em h=1 o "
    "alvo NAO se sobrepoe, entao as observacoes sao independentes de fato. "
    "HAR vs persistencia, folds purgados, por subperiodo: 2021-2022 (n=395) "
    "HAR +0.0382 vs -0.0193, 4/5 folds; 2023-2026 (n=725) HAR +0.0407 vs "
    "-0.0227, 5/5 folds; 2018-2019 (n=390) HAR -0.2862 vs +0.0468, direcao "
    "CONTRARIA; completa (n=1760) +0.2433 vs +0.2182. "
    "Diebold-Mariano (H0: mesma acuracia): 2023-2026 p=0.1075; 2021-2022 "
    "p=0.1719; 2018-2019 p=0.2275; COMPLETA p=0.4213. "
    "*** ARMADILHA DECLARADA: a janela 2021-2026 da p=0.0017, mas foi "
    "escolhida DEPOIS de ver que 2018-2019 falhava -- selecao pos-hoc, a "
    "mesma armadilha nº 3 que ja derrubou um resultado neste projeto. NAO "
    "deve ser reportada como achado. *** "
    "STATUS HONESTO: direcao consistente e replicada em dois periodos "
    "independentes recentes, com R2 absoluto POSITIVO (bate a media "
    "incondicional) e apoio de 4/5 e 5/5 folds, mas NENHUMA janela "
    "pre-registrada atinge significancia a 5%. Evidencia SUGESTIVA, nao "
    "estabelecida. Nota: 2023-2026 e a amostra ORIGINAL do projeto, entao "
    "isso nao depende da extensao -- e confirma com quantificacao e teste "
    "formal o que a Secao 5.7 ja afirmava qualitativamente ('existe "
    "previsibilidade genuina, ainda que modesta, a um dia'), exatamente como "
    "Corsi (2009) desenhou o HAR para fazer.",
    "CAMADAS DE INFORMACAO REAVALIADAS EM HORIZONTE CURTO (h=1 e h=5). "
    "MOTIVACAO, formulada a priori: todas as camadas (notícia, risco fiscal, "
    "credibilidade) tinham sido avaliadas SO em h=21 -- um regime onde nem o "
    "baseline funciona (R2 -0.41), ou seja sem sinal para melhorar. E notícia "
    "e choque de CURTO prazo: a media dos proximos 21 dias e o instrumento "
    "errado para detecta-la. O descasamento de horizonte se aplicaria a "
    "FEATURE, nao so ao modelo. "
    "DESENHO PRE-REGISTRADO: 4 camadas x 2 horizontes (10 testes, contando o "
    "tom do GDELT em 2 variantes -- bruto e suavizado 21d, porque a "
    "suavizacao de 21 dias foi escolhida para casar com alvo de 21 dias e "
    "destroi justamente o choque curto). Criterio de sucesso declarado ANTES "
    "de rodar: delta R2 > 0 E >=4/5 folds E Diebold-Mariano p < 0.05. "
    "RESULTADO: os 10 testes FALHAM. "
    "A HIPOTESE FOI REFUTADA de forma direta -- a notícia, que era a "
    "candidata teorica, e a PIOR camada em h=1 (tom bruto: -0.0083, 1/5 "
    "folds; tom 21d: -0.0007, 3/5). Risco fiscal surpresa +0.0165 mas 3/5. "
    "Risco fiscal v2 (FinBERT) -0.0855. "
    "UNICO CANDIDATO: credibilidade em h=5 (+0.0609, 4/5 folds), que passou "
    "os dois primeiros portoes E sobreviveu ao teste de vizinhanca (h=3 "
    "+0.0542, h=4 +0.0593, h=6 +0.0531, todos 4/5) -- diferente do caso ja "
    "rejeitado neste projeto, em que os vizinhos falhavam todos. "
    "MAS REPROVOU NO TERCEIRO PORTAO: DM p=0.2166 com janelas sobrepostas e "
    "p=0.9531 com janelas INDEPENDENTES (t=-0.059, efeito nulo). "
    "DIAGNOSTICO do porque o R2 subia sem haver efeito: o erro absoluto medio "
    "e IDENTICO (baseline 2.7750 vs credibilidade 2.7758) e o vies fica PIOR "
    "(+0.0355 -> +0.2475). O ganho de R2 vinha da ponderacao quadratica "
    "reduzindo alguns erros grandes, nao de prever melhor. "
    "ARITMETICA DE TESTES MULTIPLOS: P(>=4 de 5 folds por acaso) = 0.1875; "
    "em 10 testes esperavam-se 1.88 casos assim por puro acaso, e P(pelo "
    "menos um) = 0.875. Encontrar exatamente um e o que o ruido preve. "
    "CONCLUSAO: os vereditos de h=21 das camadas ficam CONFIRMADOS, agora "
    "tambem em horizonte curto. A questao fecha -- nao resta 'talvez em "
    "outro horizonte'.",
    "GARCH(1,1) REAVALIADO -- o veredito antigo (config 5) estava sobre base "
    "invalida: rodou na fonte yfinance (depois descoberta defeituosa), em "
    "h=21 (onde nem o baseline funciona) e com 830 pregoes. Refeito na fonte "
    "B3 com 2.135 pregoes, comparado contra HAR sobre o MESMO input "
    "(retorno^2), mesmos folds purgados, mesmo alvo: "
    "h=1  GARCH -0.0532 vs HAR -0.0955 -> GARCH GANHA; "
    "h=5  GARCH +0.0774 vs HAR +0.1477 -> HAR ganha; "
    "h=21 GARCH +0.1359 vs HAR +0.2879 -> HAR ganha. "
    "CORRECAO DO REGISTRO: 'GARCH pior que HAR' e FALSO em h=1. O GARCH(1,1) "
    "e um modelo de variancia condicional de UM passo a frente -- h=1 e a "
    "casa dele, e la ele bate o HAR com o mesmo input. Em horizonte longo o "
    "HAR ganha, como esperado: o GARCH reverte a variancia incondicional na "
    "taxa (alpha+beta)^h, enquanto os tres componentes do HAR aproximam "
    "memoria longa. "
    "RESSALVA METODOLOGICA: a comparacao contra HAR-Parkinson (0.2433 / "
    "0.3660 / 0.3810) NAO e valida -- o alvo e construido a partir da serie "
    "de variancia correspondente, entao Parkinson e retorno^2 tem alvos "
    "DIFERENTES e denominadores diferentes (armadilha 4). Fica como "
    "contexto, nao como comparacao. Quantificar quanto o ESTIMADOR vale "
    "contra quanto o MODELO vale exige um teste que ainda nao foi feito. "
    "LIMITE ESTRUTURAL: o GARCH consome RETORNOS, entao nao consegue usar o "
    "estimador de Parkinson (que precisa de high/low). Remover essa "
    "desvantagem exigiria Realized GARCH / GARCH-X com a medida realizada "
    "como exogena -- nao implementado.",
    "ESTIMADORES DE VARIANCIA (a hipotese central nunca atacada de frente). "
    "Ate aqui o projeto so tinha Parkinson, que usa apenas high/low e ASSUME "
    "DRIFT ZERO -- ruim para o USD/BRL, cujo settlement vai de ~3.150 a "
    "~6.220 na amostra. Implementados Garman-Klass (1980), Rogers-Satchell "
    "(1991, tolera drift), overnight, full-day (overnight + RS) e Yang-Zhang "
    "(2000, forma nativa de janela). Motivacao: o diagnostico do projeto "
    "aponta RUIDO DE MEDICAO como causa provavel, e as colunas `open` e "
    "`last` da B3 (primeiro e ultimo negocio reais) estavam coletadas em "
    "2.135 pregoes e NAO ENTRAVAM no pipeline. "
    "RESULTADO 1 -- AUTOCORRELACAO por subperiodo (nao so amostra completa, "
    "para nao repetir a armadilha 4). Em lag 1 a melhora e CONSISTENTE nos "
    "tres subperiodos: Parkinson 0.330/0.160/0.296; Garman-Klass "
    "0.364/0.214/0.349; full-day 0.326/0.303/0.379. O gap overnight carrega "
    "sinal real -- descarta-lo custava caro. "
    "RESULTADO 2 -- em lag 21 NENHUM estimador restaura memoria longa "
    "(todos entre -0.02 e +0.11 em todo subperiodo). Isso ELIMINA 'era ruido "
    "de medicao' como explicacao para o fracasso em h=21: o diagnostico de "
    "descasamento de horizonte fica sem hipotese alternativa viva. "
    "ARMADILHA DE DESENHO ENCONTRADA E CORRIGIDA: o primeiro teste de "
    "previsao usou alvo medido por PARKINSON, e ai todo estimador alternativo "
    "perdia. Nao era qualidade -- e que o alvo FAVORECE o estimador que o "
    "produziu (variante da armadilha 4). Refeito com arbitro imparcial: alvo "
    "close-to-close futuro, ruidoso mas nao-enviesado e independente do ruido "
    "de medicao dos estimadores de range. "
    "RESULTADO 3 -- com o arbitro, R2 pooled (alvo e folds identicos): "
    "h=1 Parkinson -0.0712, Garman-Klass -0.0550, Rogers-Satchell -0.0484; "
    "h=5 Parkinson 0.2102, GK 0.2463, RS 0.2536; "
    "h=21 praticamente empatados (delta -0.006 e -0.007). "
    "RS ganha do Parkinson em +0.0228 (h=1) e +0.0434 (h=5) -- e RS e "
    "justamente o unico que tolera drift, entao a previsao teorica e o "
    "resultado coincidem. full-day e MUITO pior na previsao (-0.16 a -0.30) "
    "apesar de ter a melhor autocorrelacao: o gap overnight e persistente mas "
    "seu ruido domina quando vira feature. "
    "TRES PORTOES: NAO PASSA pelo criterio declarado. RS vs Parkinson em h=1 "
    "da 2/5 folds (exigido >=4/5), embora o Diebold-Mariano com n=1760 "
    "observacoes INDEPENDENTES (em h=1 o alvo nao se sobrepoe) de t=+3.149, "
    "p=0.0017 -- e GK da p=0.0056. Os dois portoes medem coisas diferentes e "
    "AMBOS os resultados sao reportados: a melhora MEDIA e real e "
    "estatisticamente forte (o p mais baixo ja produzido neste projeto), mas "
    "NAO e consistente entre subperiodos. 2/5 e o resultado mediano esperado "
    "por acaso, ou seja o teste de folds com apenas 5 comparacoes nao rejeita "
    "nem confirma. O criterio NAO foi afrouxado depois de ver o resultado.",
    "YANG-ZHANG (2000) AVALIADO -- nao entra no despachante `ESTIMATORS` "
    "porque e estimador de JANELA, nao por dia, entao nao substitui o `rv_d` "
    "do HAR. O lugar certo dele sao os componentes SEMANAL e MENSAL, que ja "
    "sao medidas de janela: testado com rv_d = Rogers-Satchell (identico nos "
    "dois lados) e rv_w/rv_m vindos de YZ(5)/YZ(22) contra media movel de RS. "
    "Alvo arbitro close-to-close, mesmos folds. "
    "RESULTADO: PIOR em todos os horizontes -- h=1 -0.0622 vs -0.0484; "
    "h=5 0.2219 vs 0.2536; h=21 0.2797 vs 0.2964. Deltas -0.0138, -0.0317, "
    "-0.0167, com 2/5, 1/5 e 1/5 folds. Em h=5 e SIGNIFICATIVAMENTE pior "
    "(DM t=-2.113, p=0.0353). "
    "EXPLICACAO: apesar da eficiencia teorica ~14x, o YZ estima a VARIANCIA "
    "DOS RETORNOS DENTRO da janela -- subtraindo a media deles. Os componentes "
    "do HAR querem outra coisa: o NIVEL MEDIO da variancia diaria no periodo, "
    "e a media movel de um estimador por dia e o estimador direto disso. "
    "Efeito colateral relevante para cambio: ao remover a media do gap "
    "overnight, o YZ descarta o componente sistematico de CARRY, que no "
    "USD/BRL e grande dado o diferencial de juros. Correto para medir "
    "volatilidade, mas muda o objeto medido. "
    "BUG CORRIGIDO: sem `min_periods` no rolling, os dias de rolagem "
    "mascarados zeravam a serie inteira para window=22 -- o mesmo erro que "
    "quase eliminou o full_day, e que eu havia corrigido la mas nao aqui. "
    "Num estimador de janela a resposta certa e diferente: um dia "
    "inobservavel REDUZ a amostra efetiva (exigimos 80% da janela), nao "
    "anula a estimativa.",
    "COMBINACAO DE PREVISOES (Bates & Granger 1969; Timmermann 2006; "
    "'forecast combination puzzle' -- pesos IGUAIS costumam bater pesos "
    "estimados, porque estimar pesos adiciona variancia). "
    "DIAGNOSTICO PREVIO (feito ANTES do teste, para decidir se valia gastar "
    "configuracao): correlacao entre os ERROS dos modelos. Combinar so ajuda "
    "se os modelos errarem de formas diferentes. Medido em h=21: HAR x "
    "persistencia 0.754 (a menor entre modelos de qualidade parecida); HAR x "
    "GARCH 0.861; XGB x persistencia 0.618. Variantes do mesmo HAR ficam em "
    "0.94-0.98 -- e por isso que o ensemble da config 16 (media de 3 "
    "variantes do HAR) nao tinha o que extrair. "
    "PAR ESCOLHIDO PELO DIAGNOSTICO: HAR + persistencia, pesos iguais. "
    "RESULTADO NA AMOSTRA COMPLETA -- parecia o melhor achado do projeto: "
    "bate AMBOS os individuais em TODO horizonte. h=1 0.3043 vs 0.2433 "
    "(+0.061, DM p=0.0060); h=5 0.4030 vs 0.3660 (+0.037); h=21 0.4504 vs "
    "0.3810 (+0.069, 4/5 folds). Mecanismo plausivel: combinar com a "
    "persistencia e SHRINKAGE do HAR na direcao de um benchmark de zero "
    "parametros -- reduz variancia de estimacao. "
    "*** NAO SOBREVIVEU AO TESTE POR SUBPERIODO. *** "
    "h=1: deltas +0.028 / +0.008 / +0.011 (direcao positiva nos tres, mas "
    "magnitude minuscula e DM p = 0.73 / 0.70 / 0.68 -- nenhum perto de "
    "significancia). h=21: -0.109 / +0.202 / -0.009 -- INCONSISTENTE, ajuda "
    "em um periodo e atrapalha em dois. "
    "STATUS: em h=1, direcao consistente nos 3 subperiodos mas efeito fraco e "
    "nunca significativo -- mesmo status do resultado de h=1 do HAR: "
    "SUGESTIVO, nao estabelecido. Em h=21, REJEITADO. "
    "*** CONCLUSAO METODOLOGICA, que vale mais que o teste: esta e a QUINTA "
    "vez nesta linha de trabalho em que uma estatistica agregada da amostra "
    "completa produz um achado que a analise por subperiodo mata (as outras: "
    "extensao da amostra, autocorrelacao, monotonia do ciclo de rolagem, IV "
    "como previsor). Deixou de ser coincidencia: a mistura de regimes "
    "(2018-19, 2020, 2021-22, 2023-26 tem niveis de vol muito diferentes) "
    "infla sistematicamente qualquer metrica calculada no POOL. Neste "
    "dataset, R2 pooled e DM pooled NAO sao evidencia primaria confiavel -- "
    "o subperiodo deveria ser o resultado principal e o agregado apenas "
    "contexto, o inverso do que o relatorio faz hoje. ***",
]


def _fold_comparison(
    per_fold_baseline: list[dict], per_fold_other: list[dict], metric: str = "r2_oos"
) -> tuple[int, int]:
    """Conta em quantos folds a versao aumentada melhorou vs piorou o `metric`."""
    melhorou = sum(1 for b, o in zip(per_fold_baseline, per_fold_other) if o[metric] > b[metric])
    piorou = sum(1 for b, o in zip(per_fold_baseline, per_fold_other) if o[metric] < b[metric])
    return melhorou, piorou


def build_verdict(result: dict, comparison_key: str = "com_noticia", comparison_label: str = "notícia (GDELT)") -> str:
    """Veredito em linguagem simples: o modelo previu bem? A camada extra
    (noticia ou credibilidade) ajudou? Pura funcao de dados -- nao faz I/O,
    so interpreta o resultado da ablacao.

    Le o R2 POOLED (backtest.metrics.pooled_oos_metrics, concatena as
    previsoes de todos os folds antes de calcular) como numero principal --
    o R2 medio por fold (`result["baseline"]`) fica instavel com folds
    pequenos (a media de cada fold, usada como referencia do R2, tem alta
    variancia amostral) e pode exagerar o quao ruim o modelo parece. A
    contagem de "melhora em N de M folds" continua usando o R2 por fold, que
    ainda serve bem pra esse proposito especifico (robustez/consistencia).
    """
    baseline_r2 = result["baseline_pooled"]["r2_oos"]
    other_r2 = result[f"{comparison_key}_pooled"]["r2_oos"]
    delta = other_r2 - baseline_r2

    if baseline_r2 < 0:
        forecast_line = (
            f"NAO -- o HAR-RV baseline tem R2 fora da amostra negativo ({baseline_r2:.3f}). "
            "Isso quer dizer que a previsao erra MAIS do que simplesmente usar a media "
            "historica de RV como \"previsao\" -- o modelo, do jeito que esta hoje, nao "
            "tem poder preditivo real pra RV de 21 dias nesse periodo."
        )
    elif baseline_r2 < 0.10:
        forecast_line = f"FRACO -- R2 fora da amostra positivo mas baixo ({baseline_r2:.3f})."
    else:
        forecast_line = f"RAZOAVEL -- R2 fora da amostra de {baseline_r2:.3f}."

    melhorou, piorou = _fold_comparison(result["per_fold"]["baseline"], result["per_fold"][comparison_key])
    n_folds = len(result["per_fold"]["baseline"])
    if delta > 0.02 and melhorou > piorou:
        other_line = (
            f"Ajuda de forma perceptivel na media (R2 {delta:+.3f}) e melhora "
            f"em {melhorou} de {n_folds} folds -- efeito consistente."
        )
    elif delta > 0:
        other_line = (
            f"Ajuda pouco na media (R2 {delta:+.3f}), mas de forma INCONSISTENTE: "
            f"melhora em {melhorou} de {n_folds} folds e piora em {piorou} -- nao da pra "
            "confiar que o efeito e real, pode ser ruido de um fold especifico."
        )
    else:
        other_line = (
            f"NAO ajuda na media (R2 {delta:+.3f}) -- melhora em {melhorou} de {n_folds} "
            f"folds, piora em {piorou}."
        )

    return (
        f"O modelo consegue prever bem a RV futura? {forecast_line}\n\n"
        f"A camada de {comparison_label} melhora a previsao? {other_line}"
    )


def _print_header(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _print_header("1. Avaliacao final do modelo (walk-forward purgado, baseline Parkinson)")
    result = ablation.load_and_run_purged_ablation(
        horizon=21, n_splits=5, embargo_days=5, source="b3"
    )
    print(f"Baseline (HAR-RV, Parkinson):  {summary.format_metrics(result['baseline_pooled'])}")
    print(f"Com noticia:                   {summary.format_metrics(result['com_noticia_pooled'])}")

    # Persistencia pura (rv_m, sem nenhum parametro ajustado) como segundo
    # baseline de referencia -- em TODAS as comparacoes feitas neste projeto
    # (13 tentativas: HAR-RV puro, com noticia/credibilidade/risco fiscal em
    # varias formas, extensoes de literatura, risco global exogeno), a
    # persistencia bateu o HAR-RV. Por isso a secao 5 (backtest) usa
    # persistencia como previsao oficial em vez do HAR-RV -- essa linha so
    # documenta a comparacao no dataset Parkinson padrao (nao exatamente o
    # mesmo recorte trimado por disponibilidade de noticia do `result`
    # acima, mas o mesmo esquema oficial de 5 folds).
    _persist_close, _persist_daily_variance = realized.load_parkinson_prices_and_variance()
    _persist_dataset = build_dataset(_persist_close, horizon=21, daily_variance=_persist_daily_variance)
    _persist_folds = purged_walk_forward_splits(_persist_dataset, n_splits=5, horizon=21, embargo_days=5)
    _persist_forecast = persistence_forecast(_persist_dataset).reindex(
        pd.concat([test["target"] for _, test in _persist_folds]).index
    )
    _persist_metrics = pooled_oos_metrics(_persist_dataset["target"], _persist_forecast)
    print(f"Persistencia pura (rv_m):      {summary.format_metrics(_persist_metrics)}")

    md = summary.ablation_summary_md(result, CONFIGS_TESTED)
    (OUT_DIR / "ablation_summary.md").write_text(md, encoding="utf-8")

    fig = plots.plot_ablation_folds(result["per_fold"]["baseline"], result["per_fold"]["com_noticia"])
    fig.savefig(OUT_DIR / "ablation_folds.png", dpi=150)

    _print_header("2. Veredito -- noticia (GDELT)")
    print(build_verdict(result, comparison_key="com_noticia", comparison_label="notícia (GDELT)"))

    _print_header("2b. Veredito -- credibilidade (Focus/meta, Tier 1)")
    try:
        cred_result = credibility_ablation.load_and_run_credibility_ablation(
            horizon=21, n_splits=5, embargo_days=5, source="b3"
        )
        print(f"Baseline (HAR-RV, Parkinson):  {summary.format_metrics(cred_result['baseline_pooled'])}")
        print(f"Com credibilidade:             {summary.format_metrics(cred_result['com_credibilidade_pooled'])}")
        print()
        print(
            build_verdict(
                cred_result, comparison_key="com_credibilidade", comparison_label="credibilidade (theta_t + dispersao)"
            )
        )

        fig = plots.plot_ablation_folds(
            cred_result["per_fold"]["baseline"], cred_result["per_fold"]["com_credibilidade"]
        )
        fig.savefig(OUT_DIR / "ablation_folds_credibility.png", dpi=150)
    except FileNotFoundError as e:
        print(f"Credibilidade ainda nao coletada -- {e}")

    _print_header("2c. Veredito -- risco fiscal (GDELT, atencao da imprensa)")
    if VOLUME_PROCESSED_PATH.exists():
        try:
            fiscal_result = ablation.load_and_run_fiscal_risk_ablation(
                horizon=21, n_splits=5, embargo_days=5, source="b3"
            )
            print(f"Baseline (HAR-RV, Parkinson):  {summary.format_metrics(fiscal_result['baseline_pooled'])}")
            print(f"Com risco fiscal:              {summary.format_metrics(fiscal_result['com_noticia_pooled'])}")
            print()
            print(
                build_verdict(
                    fiscal_result, comparison_key="com_noticia", comparison_label="risco fiscal (GDELT, % de cobertura)"
                )
            )

            fig = plots.plot_ablation_folds(
                fiscal_result["per_fold"]["baseline"], fiscal_result["per_fold"]["com_noticia"]
            )
            fig.savefig(OUT_DIR / "ablation_folds_fiscal_risk.png", dpi=150)
        except FileNotFoundError as e:
            print(f"Risco fiscal ainda nao coletado -- {e}")
    else:
        print(
            "Risco fiscal (GDELT) ainda nao coletado -- rode "
            "data.gdelt_news.load_gdelt_volume_processed primeiro."
        )

    _print_header("2d. Veredito -- risco fiscal REFINADO (surpresa + sentimento FinBERT-PT-BR)")
    if FISCAL_SENTIMENT_PROCESSED_PATH.exists():
        try:
            fiscal_v2_result = ablation.load_and_run_fiscal_risk_ablation_v2(
                horizon=21, n_splits=5, embargo_days=5, source="b3"
            )
            print(f"Baseline (HAR-RV, Parkinson):     {summary.format_metrics(fiscal_v2_result['baseline_pooled'])}")
            print(f"Com risco fiscal refinado:        {summary.format_metrics(fiscal_v2_result['com_risco_fiscal_v2_pooled'])}")
            print()
            print(
                build_verdict(
                    fiscal_v2_result,
                    comparison_key="com_risco_fiscal_v2",
                    comparison_label="risco fiscal refinado (surpresa + sentimento FinBERT nas manchetes em portugues)",
                )
            )

            fig = plots.plot_ablation_folds(
                fiscal_v2_result["per_fold"]["baseline"], fiscal_v2_result["per_fold"]["com_risco_fiscal_v2"]
            )
            fig.savefig(OUT_DIR / "ablation_folds_fiscal_risk_v2.png", dpi=150)
        except FileNotFoundError as e:
            print(f"Risco fiscal refinado ainda nao coletado -- {e}")
    else:
        print(
            "Sentimento fiscal (FinBERT) ainda nao coletado -- rode "
            "sentiment.daily_index.load_fiscal_risk_sentiment_index() primeiro "
            "(precisa de data.gdelt_news.load_gdelt_headlines_processed_chunked antes)."
        )

    _print_header("3. Graficos historicos")
    fx_df = pd.read_parquet(realized.FX_SPOT_PROCESSED_PATH).set_index("date").sort_index()
    close = fx_df["close"]
    daily_variance = realized.parkinson_daily_variance(fx_df["high"], fx_df["low"])
    rv_series = realized.parkinson_vol(fx_df["high"], fx_df["low"], window=21).dropna()
    fig = plots.plot_series(
        rv_series,
        title="RV realizada USD/BRL (Parkinson, janela 21d, anualizada)",
        ylabel="RV (% a.a.)",
        color=plots.SERIES_GREEN,
    )
    fig.savefig(OUT_DIR / "rv_history.png", dpi=150)
    print(f"rv_history.png ({len(rv_series)} pontos)")

    if TONE_PROCESSED_PATH.exists():
        tone_series = pd.read_parquet(TONE_PROCESSED_PATH).set_index("date")["tone"].sort_index()
        fig = plots.plot_series(tone_series, title="Tom medio diario de noticias (GDELT)", ylabel="Tom", color=plots.SERIES_BLUE)
        fig.savefig(OUT_DIR / "gdelt_tone.png", dpi=150)
        print(f"gdelt_tone.png ({len(tone_series)} pontos)")

    _print_header("4. Sinal de hoje (estrategia)")
    if IV_PROCESSED_PATH.exists():
        iv_df = implied.load_iv_atm_processed(target_days=21)
        iv_today = iv_df.iloc[-1]
        rv_today = rv_series.iloc[-1]
        spot = pd.read_parquet(PTAX_PROCESSED_PATH)
        spot_today = spot[spot["tipo"] == "venda"].sort_values("date").iloc[-1]["value"]

        sig = signal.generate_signal(rv_today, iv_today["iv_atm_pct"], band_pct=1.0)
        label = {signal.LONG_VOL: "COMPRAR VOL", signal.SHORT_VOL: "VENDER VOL", signal.NO_TRADE: "NAO OPERAR"}[sig]
        n = sizing.size_straddle(target_vega=1000, spot=spot_today, ttm_days=21, iv_pct=iv_today["iv_atm_pct"])

        print(f"RV realizada (21d):  {rv_today:.2f}%")
        print(f"IV ATM (21d):        {iv_today['iv_atm_pct']:.2f}%")
        print(f"Sinal:               {label}")
        print(f"Straddles p/ vega alvo R$1000/ponto: {n:,.0f}")
    else:
        print("Superficie de IV ainda nao coletada -- rode data.iv_surface primeiro.")

    _print_header("5. Backtest de P&L (ILUSTRATIVO -- ver aviso abaixo)")
    if IV_PROCESSED_PATH.exists():
        print(
            "AVISO: a B3 so publica o snapshot do dia da superficie de IV (sem historico\n"
            "pra download) -- so ha 1 dia real de IV conhecido. A IV de ENTRADA de cada\n"
            "trade abaixo e uma PROXY (RV Parkinson trailing x premio de risco fixo,\n"
            "calibrado nesse unico dia real). O preco de entrada/saida e o payoff\n"
            "terminal usam dado 100% real. Resultado ILUSTRATIVO -- testa o motor e da\n"
            "uma nocao de ordem de grandeza, NAO e evidencia de lucro real.\n"
        )

        dataset = build_dataset(close, horizon=21, daily_variance=daily_variance)
        # Persistencia pura (rv_m) em vez de HAR-RV: em TODAS as 13
        # comparacoes feitas neste projeto (HAR-RV puro, com camadas de
        # noticia/credibilidade/risco fiscal, extensoes de literatura,
        # risco global exogeno), a persistencia bateu o HAR-RV no R2 pooled
        # -- ver secao 1 (+0.126 vs -0.510 no esquema oficial). Sem
        # coeficiente nenhum pra ajustar, entao nao precisa de walk-forward
        # (nao ha risco de overfitting num modelo sem parametros).
        rv_forecast = persistence_forecast(dataset)

        iv_df = implied.load_iv_atm_processed(target_days=21)
        calib_date = iv_df["refdate"].iloc[-1]
        iv_real = iv_df["iv_atm_pct"].iloc[-1]
        rv_calib_idx = rv_series.index.asof(calib_date)
        risk_premium = engine.calibrate_risk_premium(rv_series.loc[rv_calib_idx], iv_real)
        iv_proxy_series = engine.proxy_iv(rv_series, risk_premium)

        print(f"Calibracao: {calib_date.date()}  IV_real={iv_real:.2f}%  "
              f"RV_trailing={rv_series.loc[rv_calib_idx]:.2f}%  premio_de_risco={risk_premium:.3f}x")

        # Diagnostico: RV_previsto (persistencia) e IV_proxy sao QUASE A
        # MESMA SERIE -- IV_proxy = RV_trailing_21d x 0.847 (premio de risco
        # fixo), e persistencia = RV_trailing_22d (rv_m). Correlacao ~0.994
        # nesse periodo. Isso significa que o spread que decide compra/venda
        # e dominado por uma constante multiplicativa (~1.18x), nao por
        # sinal genuino -- o momento exato em que ele cruza a banda de 1.0
        # e ditado por ruido de curtissimo prazo na vol recente, nao pela
        # qualidade da previsao. E por isso que trocar HAR-RV por
        # persistencia MELHOROU o R2 (+0.126 vs -0.510, secao 1) mas
        # PIOROU o backtest (Sharpe 1.087->0.014): R2 mede acerto medio em
        # TODOS os dias da amostra; taxa de acerto mede so os poucos dias
        # em que o spread cruzou a banda -- um subconjunto pequeno e
        # nao-aleatorio, escolhido por um limiar sensivel a ruido quando a
        # previsao e quase colinear com a propria proxy de IV. Reforca por
        # que a secao 5 e ILUSTRATIVA: o problema maior nao e qual modelo
        # usamos, e a IV-proxy ser um multiplo constante da RV trailing.
        _corr = rv_forecast.corr(iv_proxy_series.reindex(rv_forecast.index))
        print(f"Diagnostico: correlacao RV_previsto x IV_proxy = {_corr:.3f} "
              "(quase colineares -- ver nota no codigo sobre por que isso "
              "desconecta R2 de taxa de acerto)")

        trades = engine.run_backtest(
            close, rv_forecast, iv_proxy_series, horizon=21, band_pct=1.0, target_vega=1000.0, spread_pct=0.05
        )
        stats = engine.summarize_backtest(trades, horizon=21)

        if stats["n_trades"] > 0:
            print()
            print(f"Trades: {stats['n_trades']} ({stats['n_long_vol']} compra vol, {stats['n_short_vol']} venda vol)")
            print(f"Taxa de acerto: {stats['win_rate']:.1%}")
            print(f"PnL total (unidades do modelo): {stats['total_pnl']:,.0f}")
            print(f"PnL medio por trade: {stats['avg_pnl']:,.0f}  (desvio: {stats['pnl_std']:,.0f})")
            print(f"Sharpe (anualizado): {stats['sharpe']:.3f}")

            # Deflated Sharpe Ratio: varia a banda morta (band_pct) -- o
            # hiperparametro mais natural da estrategia -- como familia de
            # tentativas, e ajusta o Sharpe da config escolhida (1.0) pelo
            # numero de variacoes testadas (guardrail do CLAUDE.md: registrar
            # configuracoes testadas + DSR, nao so reportar a melhor sem
            # disclosure).
            band_pcts_tested = [0.5, 1.0, 1.5, 2.0]
            trial_sharpes = [stats["sharpe"]]
            for bp in band_pcts_tested:
                if bp == 1.0:
                    continue
                t = engine.run_backtest(
                    close, rv_forecast, iv_proxy_series, horizon=21, band_pct=bp,
                    target_vega=1000.0, spread_pct=0.05,
                )
                s = engine.summarize_backtest(t, horizon=21)
                if s["n_trades"] > 1:
                    trial_sharpes.append(s["sharpe"])

            if len(trial_sharpes) >= 2 and stats["n_trades"] > 1:
                capital_at_risk = (trades["n_contracts"] * trades["premium"]).abs()
                trade_returns = (trades["pnl_net"] / capital_at_risk).to_numpy()
                dsr = deflated_sharpe_ratio(
                    sr_hat=stats["sharpe"],
                    sr_trials_std=float(np.std(trial_sharpes, ddof=1)),
                    n_trials=len(trial_sharpes),
                    n_obs=stats["n_trades"],
                    skew=float(_skew(trade_returns)),
                    kurtosis=float(_kurtosis(trade_returns, fisher=False)),
                )
                print(f"Deflated Sharpe Ratio: {dsr:.3f}  "
                      f"(band_pct=1.0 escolhido entre {len(trial_sharpes)} variacoes testadas "
                      f"{band_pcts_tested}, Sharpes={[round(s, 2) for s in trial_sharpes]})")

            trades.to_csv(OUT_DIR / "backtest_trades.csv", index=False)
            fig = plots.plot_cumulative_pnl(trades)
            fig.savefig(OUT_DIR / "backtest_pnl.png", dpi=150)
            print(f"\nbacktest_trades.csv ({len(trades)} trades) e backtest_pnl.png salvos.")
        else:
            print("Nenhum trade gerado com esses parametros.")

        _print_header("5b. Risco fiscal (GDELT) na previsao de RV -- compara rendimento dos trades")
        print(
            "Nota: esta secao (e a 5c) testam se a camada ajuda um HAR-RV AJUSTADO -- "
            "pergunta de pesquisa diferente da secao 5 acima, que ja usa persistencia "
            "(sem parametros) como previsao oficial por bater o HAR-RV em toda comparacao feita.\n"
        )
        if VOLUME_PROCESSED_PATH.exists():
            fiscal_news = load_fiscal_risk_series()
            dataset_fiscal = build_dataset(
                close, news=fiscal_news, horizon=21, news_smooth_window=21, daily_variance=daily_variance
            )
            # Mesmos folds (mesmo periodo de teste) para as duas versoes do
            # modelo -- comparacao justa. O dataset com noticia comeca mais
            # tarde (so a partir da 1a data com cobertura de risco fiscal no
            # GDELT), entao esse "baseline" tem MENOS trades que o da secao 5.
            fiscal_folds = purged_walk_forward_splits_by_step(
                dataset_fiscal, min_train_size=252, step_size=21, horizon=21, embargo_days=5
            )

            rv_forecast_baseline_aligned = engine.generate_oos_rv_forecast(
                dataset_fiscal, BASELINE_FEATURES, horizon=21, n_splits=5, embargo_days=5, folds=fiscal_folds
            )
            rv_forecast_fiscal = engine.generate_oos_rv_forecast(
                dataset_fiscal, NEWS_FEATURES, horizon=21, n_splits=5, embargo_days=5, folds=fiscal_folds
            )

            # Peso (coeficiente OLS) que o modelo atribui a noticia de risco
            # fiscal em cada fold -- estimado a partir dos dados, nao
            # escolhido a dedo (o time e mais forte em quant/eng que em ML;
            # regressao simples e defensavel > blend arbitrario).
            news_weights = [
                fit_har(train, NEWS_FEATURES, log_target=True).params.get("news", float("nan"))
                for train, _ in fiscal_folds
            ]

            trades_baseline_aligned = engine.run_backtest(
                close, rv_forecast_baseline_aligned, iv_proxy_series, horizon=21, band_pct=1.0,
                target_vega=1000.0, spread_pct=0.05,
            )
            trades_fiscal = engine.run_backtest(
                close, rv_forecast_fiscal, iv_proxy_series, horizon=21, band_pct=1.0,
                target_vega=1000.0, spread_pct=0.05,
            )
            stats_baseline_aligned = engine.summarize_backtest(trades_baseline_aligned, horizon=21)
            stats_fiscal = engine.summarize_backtest(trades_fiscal, horizon=21)

            print(f"Peso medio da noticia de risco fiscal (coef. OLS, log-RV) nos "
                  f"{len(fiscal_folds)} folds: {np.nanmean(news_weights):+.4f} "
                  f"(desvio: {np.nanstd(news_weights):.4f})")
            print()
            print(f"{'':22s}{'SEM risco fiscal':>20s}{'COM risco fiscal':>20s}")
            for label, key in [
                ("Trades", "n_trades"), ("Taxa de acerto", "win_rate"),
                ("PnL total", "total_pnl"), ("PnL medio", "avg_pnl"), ("Sharpe", "sharpe"),
            ]:
                v_base = stats_baseline_aligned.get(key, float("nan"))
                v_fiscal = stats_fiscal.get(key, float("nan"))
                if key == "win_rate":
                    print(f"{label:22s}{v_base:>19.1%} {v_fiscal:>19.1%}")
                elif key in ("total_pnl", "avg_pnl"):
                    print(f"{label:22s}{v_base:>20,.0f}{v_fiscal:>20,.0f}")
                elif key == "sharpe":
                    print(f"{label:22s}{v_base:>20.3f}{v_fiscal:>20.3f}")
                else:
                    print(f"{label:22s}{v_base:>20.0f}{v_fiscal:>20.0f}")

            trades_baseline_aligned.to_csv(OUT_DIR / "backtest_trades_baseline_aligned.csv", index=False)
            trades_fiscal.to_csv(OUT_DIR / "backtest_trades_fiscal_risk.csv", index=False)
            print(f"\nbacktest_trades_baseline_aligned.csv ({len(trades_baseline_aligned)} trades) e "
                  f"backtest_trades_fiscal_risk.csv ({len(trades_fiscal)} trades) salvos.")
        else:
            print(
                "Risco fiscal (GDELT) ainda nao coletado -- rode "
                "data.gdelt_news.load_gdelt_volume_processed primeiro."
            )

        _print_header("5c. Risco fiscal REFINADO (surpresa + sentimento) -- compara rendimento dos trades")
        if FISCAL_SENTIMENT_PROCESSED_PATH.exists():
            fiscal_surprise_series = fiscal_risk_surprise(load_fiscal_risk_series(), window=63)
            fiscal_sentiment_df = pd.read_parquet(FISCAL_SENTIMENT_PROCESSED_PATH)
            fiscal_sentiment_series = fiscal_sentiment_df.set_index("date")["sentiment_mean"].sort_index()

            dataset_fiscal_v2 = ablation.build_dataset_with_fiscal_risk(
                close, fiscal_surprise_series, fiscal_sentiment_series,
                horizon=21, daily_variance=daily_variance, sentiment_smooth_window=5,
            )
            fiscal_v2_folds = purged_walk_forward_splits_by_step(
                dataset_fiscal_v2, min_train_size=252, step_size=21, horizon=21, embargo_days=5
            )

            rv_forecast_baseline_v2 = engine.generate_oos_rv_forecast(
                dataset_fiscal_v2, BASELINE_FEATURES, horizon=21, n_splits=5, embargo_days=5, folds=fiscal_v2_folds
            )
            rv_forecast_fiscal_v2 = engine.generate_oos_rv_forecast(
                dataset_fiscal_v2, ablation.FISCAL_RISK_FEATURES, horizon=21, n_splits=5, embargo_days=5,
                folds=fiscal_v2_folds,
            )

            v2_weights = {"fiscal_surprise": [], "fiscal_sentiment": []}
            for train, _ in fiscal_v2_folds:
                model = fit_har(train, ablation.FISCAL_RISK_FEATURES, log_target=True)
                v2_weights["fiscal_surprise"].append(model.params.get("fiscal_surprise", float("nan")))
                v2_weights["fiscal_sentiment"].append(model.params.get("fiscal_sentiment", float("nan")))

            trades_baseline_v2 = engine.run_backtest(
                close, rv_forecast_baseline_v2, iv_proxy_series, horizon=21, band_pct=1.0,
                target_vega=1000.0, spread_pct=0.05,
            )
            trades_fiscal_v2 = engine.run_backtest(
                close, rv_forecast_fiscal_v2, iv_proxy_series, horizon=21, band_pct=1.0,
                target_vega=1000.0, spread_pct=0.05,
            )
            stats_baseline_v2 = engine.summarize_backtest(trades_baseline_v2, horizon=21)
            stats_fiscal_v2 = engine.summarize_backtest(trades_fiscal_v2, horizon=21)

            for feat_name, weights in v2_weights.items():
                print(f"Peso medio de '{feat_name}' (coef. OLS, log-RV) nos {len(fiscal_v2_folds)} folds: "
                      f"{np.nanmean(weights):+.4f} (desvio: {np.nanstd(weights):.4f})")
            print()
            print(f"{'':22s}{'SEM risco fiscal':>20s}{'COM risco fiscal v2':>22s}")
            for label, key in [
                ("Trades", "n_trades"), ("Taxa de acerto", "win_rate"),
                ("PnL total", "total_pnl"), ("PnL medio", "avg_pnl"), ("Sharpe", "sharpe"),
            ]:
                v_base = stats_baseline_v2.get(key, float("nan"))
                v_fiscal = stats_fiscal_v2.get(key, float("nan"))
                if key == "win_rate":
                    print(f"{label:22s}{v_base:>19.1%} {v_fiscal:>21.1%}")
                elif key in ("total_pnl", "avg_pnl"):
                    print(f"{label:22s}{v_base:>20,.0f}{v_fiscal:>22,.0f}")
                elif key == "sharpe":
                    print(f"{label:22s}{v_base:>20.3f}{v_fiscal:>22.3f}")
                else:
                    print(f"{label:22s}{v_base:>20.0f}{v_fiscal:>22.0f}")

            trades_baseline_v2.to_csv(OUT_DIR / "backtest_trades_baseline_v2_aligned.csv", index=False)
            trades_fiscal_v2.to_csv(OUT_DIR / "backtest_trades_fiscal_risk_v2.csv", index=False)
            print(f"\nbacktest_trades_baseline_v2_aligned.csv ({len(trades_baseline_v2)} trades) e "
                  f"backtest_trades_fiscal_risk_v2.csv ({len(trades_fiscal_v2)} trades) salvos.")
        else:
            print(
                "Sentimento fiscal (FinBERT) ainda nao coletado -- rode "
                "sentiment.daily_index.load_fiscal_risk_sentiment_index() primeiro."
            )
    else:
        print("Superficie de IV ainda nao coletada -- rode data.iv_surface primeiro.")

    print()
    print(f"Arquivos gerados em: {OUT_DIR}")


if __name__ == "__main__":
    main()
