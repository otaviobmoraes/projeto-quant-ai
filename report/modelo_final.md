# VolBoy — especificação do modelo final

Documento de referência da versão que produziu o melhor backtest do projeto.
Todos os números aqui têm entrada correspondente em `report/run_report.py:CONFIGS_TESTED`
(52 registros).

---

## 1. Resposta curta

O modelo final **não usa machine learning**. É uma regressão linear de três
parâmetros, estimada por mínimos quadrados sobre duas features adimensionais.

Isso não é uma escolha por conveniência: gradient boosting foi implementado,
testado sob o mesmo protocolo e **perdeu**. O limite não é a forma funcional —
é a ausência de sinal no horizonte, e isso foi medido, não suposto (seção 7).

---

## 2. O que entra

**Fonte de preço.** Futuro de dólar da B3 (DOL), arquivos BVBG-086 —
2.135 pregões de 2018-01 a 2026-08. É o instrumento sobre o qual a opção é
escrita, então é a RV dele que a estratégia precisa prever.

**Estimador de variância diária.** Parkinson (1980), sobre o range do pregão:

```
                ln(H_t / L_t)²
      RV_t  =  ────────────────
                   4 · ln(2)

   H_t = máxima do pregão      L_t = mínima do pregão
```

Escolhido por medição, não por convenção: RMSE menor que o retorno de
fechamento ao quadrado em todos os folds testados. Rogers-Satchell e
Garman-Klass ficam marginalmente à frente em h=1 (DM p=0,0017 e p=0,0056), mas
**não passam** no portão de consistência entre folds (2/5), então o baseline
oficial segue Parkinson.

**Componentes HAR** (Corsi, 2009) — médias móveis da variância diária:

```
   RV_d(t)  =  RV_t                              (diário)

                 1                                (semanal)
   RV_w(t)  =  ───  ·  Σ RV_(t-i)   ,  i = 0..4
                 5

                 1                                (mensal)
   RV_m(t)  =  ────  ·  Σ RV_(t-i)  ,  i = 0..21
                 22
```

---

## 3. As features: adimensionais, de propósito

O modelo **não** usa `RV_d`, `RV_w`, `RV_m` em nível. Usa duas razões:

```
              RV_d(t)                     RV_w(t)
   x₁(t)  =  ─────────         x₂(t)  =  ─────────
              RV_m(t)                     RV_m(t)
```

Elas codificam a **forma** da estrutura a termo de volatilidade (curto acima ou
abaixo do médio prazo), e não o **nível**, que não é estacionário.

**Por que isso importa, e é a correção central desta versão.** O HAR em nível
tem viés que troca de sinal entre folds — medido: −8,15%, −9,27%, +1,53%,
+11,38%, +11,02%. O treino é de janela expansiva, então o modelo carrega a
média de regimes antigos: sub-prevê em 2019-2020 (vol alta) e sobre-prevê em
2023-2026 (vol calma). Na janela em que o backtest opera o viés agregado é
**−7,03%**, contra um prêmio de risco medido de **+7,90%** — o erro de nível do
modelo era da mesma ordem do prêmio que a estratégia tenta capturar, e
empurrava o sinal para o lado vendido.

O R² **não detecta isso**: remover o viés move o R² pooled de 0,3810 para
0,3809. Por isso `vol.forecast.forecast_metrics` passou a reportar `vies_pct`
em toda avaliação.

---

## 4. O alvo: uma razão, não um nível

```
                    ⎛  RV_futura(t,h)  ⎞
   y(t,h)  =   ln   ⎜ ──────────────── ⎟
                    ⎝  RV_corrente(t)  ⎠
```

onde ambos são volatilidades anualizadas em pontos percentuais:

```
                            ______________________
                           ╱  252
   RV_futura(t,h) = 100 · ╱  ─────  ·  Σ RV_(t+i)     ,  i = 1..h
                        ╲╱      h

                            _____________________
   RV_corrente(t)  = 100 · ╱  252 · RV_m(t)
                         ╲╱
```

Em forma logarítmica isso é o log-HAR com o coeficiente do componente mensal
**restrito a 1** — uma restrição, portanto reduz variância de estimação ao
custo de viés se a restrição for falsa.

---

## 5. A equação estimada

```
   ┌────────────────────────────────────────────────────────────┐
   │                                                            │
   │    ln( RV_futura(t,h) / RV_corrente(t) )                   │
   │                                                            │
   │              =   α  +  β₁ · x₁(t)  +  β₂ · x₂(t)  +  ε(t)  │
   │                                                            │
   └────────────────────────────────────────────────────────────┘
```

Ajuste por mínimos quadrados. Coeficientes do último fold
(treino de 2.097 pregões, h=5):

| parâmetro | valor | t |
|---|---|---|
| α (intercepto) | −0,1959 | −13,53 |
| β₁ (razão diária) | +0,0432 | +5,85 |
| β₂ (razão semanal) | +0,1129 | +7,23 |

Interpretação: α negativo é **reversão à média** — na ausência de sinal a vol
futura fica ~18% abaixo da corrente. β₁ e β₂ positivos dizem que vol de curto
prazo acima da de médio prazo antecipa vol futura mais alta, e o componente
semanal pesa ~2,6× mais que o diário (o diário é mais ruidoso).

**Previsão, de volta à escala de nível:**

```
   RV_prevista(t,h) = RV_corrente(t) · exp( α + β₁·x₁ + β₂·x₂ ) · exp( σ² / 2 )
                      └──── nível ────┘   └──── forma da reversão ────┘  └ Jensen ┘
```

O último fator não é cosmético: `exp( E[ln Y] )` é a **mediana** de Y, não a
média; para uma variável log-normal a média é `exp( μ + σ²/2 )`. Sem ele a
previsão de nível sai sistematicamente baixa. Medido em h=5: σ² = 0,0634,
logo o fator é **1,0322**. O `predict` em log do projeto omitia isso desde
sempre.

σ² vem do resíduo de **treino**, nunca do teste.

---

## 6. Horizontes: onde funciona e onde não

O modelo é o mesmo em todo horizonte; o que muda é o alvo. O desempenho **não**
é uniforme, e a versão em razão só é superior no regime curto:

| h | ΔR² vs HAR em nível | folds | DM p (indep.) | veredito |
|---|---|---|---|---|
| **1** | **+0,0738** | **4/5** | **0,0032** | **passa nos 3 portões** |
| 2 | +0,0556 | 4/5 | 0,0782 | 2 de 3 |
| 3 | +0,0488 | 4/5 | 0,1114 | 2 de 3 |
| 5 | +0,0311 | 4/5 | 0,8908 | 2 de 3 |
| 10 | −0,0138 | 2/5 | — | pior |
| 21 | −0,0513 | 2/5 | — | pior |

**Correção a uma leitura tentadora:** este não é "o modelo com a melhor
previsibilidade em tudo". Em h≥10 ele **perde** para o HAR em nível. Ele é o
melhor no regime curto — que é justamente onde a estratégia passou a operar, e
por isso é o modelo final.

O que é consistente em **todo** horizonte e **todo** subperíodo é a calibragem:
o |viés| médio cai de 6,52% para 1,59% e para de trocar de sinal.

**Horizonte operacional: h = 5 dias úteis.**

---

## 7. Onde o machine learning entra (e por que sai)

Foi implementado e avaliado sob protocolo idêntico — mesmas features, mesmo
alvo em log, mesmos folds purgados, hiperparâmetros pré-registrados sem busca
em grade (`vol/ml_forecast.py`).

**XGBoost perde do HAR em todo horizonte ≥ 3**, e a distância cresce
monotonicamente com o horizonte (−0,024 em h=1 até −0,149 em h=21),
acompanhando a queda de observações independentes de 1.760 para ~101.

E o limite foi **medido**, não inferido por eliminação (`backtest/capacity.py`):

**Teto em amostra** — ajustando *e* avaliando no mesmo bloco, isto é com
permissão para colar:

```
   h = 21  →  R² = 0,0377        h = 1  →  R² = 0,0481
```

Nenhum ajuste honesto supera o teto de uma versão desonesta.

**Varredura de capacidade** — XGBoost com regularização desligada:

```
   capacidade          R² dentro     R² fora
   ─────────────────────────────────────────
   prof. 2,  100 árv.    +0,524      −0,323
   prof. 4,  300 árv.    +0,750      −0,846
   prof. 8,  600 árv.    +0,998      −0,924
   prof. 12, 1500 árv.   +0,9999     −0,973
                         ▲            ▲
                    memoriza      generaliza
                    perfeito       pior
```

Essa divergência é a definição operacional de "não há sinal": o gargalo não é
capacidade de representação.

**Curva de aprendizado** — em h=21 não converge para nada útil. Em h=1 ainda
sobe (+0,0197 no último dobro de dados): é o único lugar do projeto onde
"mais dados ajudariam" é afirmação sustentada por medição.

Também testados e rejeitados: GARCH(1,1) (ganha em h=1, perde em h≥5), modelo
global em painel de 8 moedas emergentes (reprova nos três portões), e todas as
camadas de informação exógena — notícia via GDELT + FinBERT-PT-BR,
credibilidade de Barro-Gordon via Focus/COPOM, risco global via VIX+DXY.

---

## 8. Como o modelo decide operar

### Passo 1 — IV de mercado

Invertida numericamente (Brent) de negócios reais de opção de dólar da B3 via
**Black-76** — opção sobre futuro, não sobre spot:

```
   C  =  exp(−r·T) · [ F · N(d₁)  −  K · N(d₂) ]

            ln(F/K)  +  ½ · σ² · T
   d₁  =  ──────────────────────────        d₂  =  d₁  −  σ·√T
                  σ · √T

   F = futuro     K = strike     T = prazo em anos     N = normal acumulada
```

Filtros de qualidade pré-fixados: ≥2 negócios na série, IV entre 4% e 60%,
|moneyness| ≤ 3%, e **vencimento casado com o horizonte** (3–12 dias corridos
para h=5).

### Passo 2 — sinal

```
   spread(t)  =  RV_prevista(t,h)  −  IV(t)

                  ┌  +1   (compra vol)   se  spread >  b
   sinal(t)  =    │   0   (fora)         se  |spread| ≤ b
                  └  −1   (vende vol)    se  spread < −b
```

A banda morta `b` existe porque o spread bid-ask da opção é largo; operar
diferenças pequenas demais não paga o custo de execução.

### Passo 3 — estrutura e tamanho

Straddle ATM sobre o futuro (K = F_t), dimensionado por vega alvo constante:

```
                    vega_alvo
   n_contratos  =  ─────────────         vega_straddle = 2·exp(−r·T)·F·φ(d₁)·√T
                   vega_straddle
```

Existe também dimensionamento por **risco** constante (`size_by_risk_target`),
que divide pela dispersão móvel da RV. Não melhora o Sharpe — Harvey et al.
(2018) já preveem isso para moedas — mas corta a perda máxima em 41% e o
drawdown pela metade. Adotado por gestão de risco, não como melhora de retorno.

### Passo 4 — delta-hedge diário

A posição é rebalanceada para delta-neutro todo pregão, na IV de entrada:

```
   Δ_straddle  =  exp(−r·T) · [ 2·N(d₁)  −  1 ]

   posição_futuro(t)  =  − sinal · n_contratos · Δ_straddle(t)
```

Sem hedge o P&L é `|S_T − K|` contra o prêmio: mede **direção mais ruído**, não
volatilidade. Com hedge, o resultado converge para

```
        ⌠
        ⎮  ½ · Γ(t) · F(t)²  ·  ( IV²  −  RV(t)² )  dt
        ⌡
```

que é exatamente o spread que a estratégia diz operar. Efeito medido: a
dispersão do P&L por operação cai **61–68%** em todas as configurações.

Foi o hedge que revelou que os Sharpes positivos do backtest anterior
(+0,301 e +0,495) eram **sorte direcional**: removido o ruído de direção, o
sinal do HAR em nível é consistentemente negativo.

---

## 9. Validação

- **Walk-forward purgado com embargo** de 5 dias (López de Prado). Nunca split
  aleatório; o treino descarta as linhas cujo rótulo se sobrepõe ao teste.
- **Três portões pré-registrados** para qualquer camada nova:
  ΔR² > 0 **e** ≥4/5 folds **e** Diebold-Mariano p < 0,05 em janelas
  **independentes**. Declarados antes de rodar; nunca afrouxados depois.
- **Subperíodo como resultado principal**, pooled apenas como contexto — a
  mistura de regimes infla sistematicamente qualquer métrica agregada, e isso
  já derrubou seis achados aparentes neste projeto.
- **Bootstrap de bloco estudentizado** (Ledoit & Wolf, 2008) para inferência
  sobre o Sharpe com operações sobrepostas.
- **Deflated Sharpe** descontando as 52 configurações testadas.

---

## 10. Desempenho, com as ressalvas

Straddle ATM delta-neutro, operações sobrepostas, IC95 por bootstrap de bloco:

| h | spread medido | Sharpe | IC95 | p |
|---|---|---|---|---|
| **5** | **8,51%** | **+1,691** | [−0,23, +3,51] | 0,079 |
| 10 | 5,60% | +0,030 | [−0,69, +0,78] | 0,914 |
| 21 | 6,94% | −0,729 | [−1,30, −0,21] | 0,012 |

Sem custo de execução o gradiente fica limpo: **+2,823** (h=5, p=0,008),
+0,712 (h=10, p=0,041), +0,015 (h=21, p=0,956).

Em h=5: 55 operações, acerto 56,4%, assimetria **+0,49** — a patologia de cauda
esquerda que dominava h=21 não aparece. Os 3 maiores ganhos somam 61,8% do P&L,
e removendo-os o Sharpe ainda é +0,823.

**O que impede chamar isso de resultado estabelecido:**

1. O portão declarado era IC95 excluindo zero. O IC vai de −0,23 a +3,51;
   p = 0,079 não é p < 0,05. **Não foi afrouxado depois de ver o número.**
2. A anualização supõe 50 operações/ano; a cobertura de IV sustenta 13,4. Com a
   frequência **realizável** o Sharpe é **+0,865**, não +1,691.
3. O spread tem 29 observações; usando o p75 delas (19,71%) o Sharpe cai a
   +0,22.
4. 2023 colapsa (n=5).
5. Retorno percentual **não é calculável**: a estratégia opera majoritariamente
   vendida, a perda de uma venda de straddle não tem teto, e o capital exigido é
   a margem — que não está nos dados. Reportamos Sharpe (invariante a escala),
   PSR, DSR e drawdown em unidades absolutas.

---

## 11. Reprodução

```bash
python -m report.run_report          # todos os vereditos
python -m report.figuras_vereditos   # as 7 figuras
python -m report.ml_diagram          # diagrama de blocos do pipeline
pytest -q                            # 504 testes
```
