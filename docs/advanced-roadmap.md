# @columna/advanced — план догона Minitab

## Границы модулей

| Модуль | Что содержит | Правило приёма |
|---|---|---|
| **standard** — `columna`, `@columna/core`, `@columna/runtime` | DataFrame-движок: IO, Expr DSL, план, бэкенды; всё, что есть в pandas/polars: `rank`, broadcast-агрегаты, `over`, `corr/cov`, математика в Expr, fused kernel, radix sort | Примитив общего назначения, не привязанный к статистической методологии |
| **advanced** — `@columna/advanced`, `columna/advanced` | Всё, что нужно «догнать Minitab»: распределения, тесты гипотез, ANOVA-сравнения, дисперсии, нормальность, непараметрика и всё последующее из этого плана | Импорт `columna/advanced` подключает и функции, и методы `df.*`/`lazy.*` через module augmentation; standard от advanced не зависит |

Уже в advanced: `dist` (normal, t, χ², F, стьюдентизированный размах, Даннет), `ttest1/2/Paired`, `anova`,
`tukeyHSD`, `fisherLSD`, `dunnett`, `hsuMCB`, `levene`, `bartlett`, `bonett/bonett2`, `varTest2`,
`chi2test/chi2gof/crosstab`, `andersonDarling`, `ryanJoiner`, `kolmogorovSmirnov`, `shapiroWilk`,
`mannWhitney`, `kruskal`, ярусы 2–5 (Basic Statistics → Predictive).

## Протокол тестирования каждой новой функции

Каждое добавление закрывается **тремя слоями** тестов, и функция не считается готовой, пока все три не зелёные:

1. **Эталон (fixture)** — `packages/advanced/tests/fixtures/<topic>-scipy.json`, сгенерированный скриптом `packages/advanced/tests/refs/<topic>_ref.py` из scipy / statsmodels / R (nortest, multcomp) или прямого расчёта формул Minitab в Python. Допуски: 1e-12 для замкнутых формул, 1e-7…1e-9 для квадратур, явный комментарий, если эталон сам неточен (как `scipy.t.ppf`).
2. **Свойства** — инварианты, которые верны независимо от эталона: симметрии (`cdf(−x) = sf(x)`), вырождения (Даннет с одним сравнением = t), согласованность p-value и ДИ (при confidence = 1 − p граница касается H₀), монотонность, обработка null/NaN, ошибки на некорректном входе.
3. **Синтетические данные (Монте-Карло, seeded)** — генератор `rng(seed)` из `tests/bonett.test.ts` (mulberry32 → uniform → Box–Muller → t₃):
   - **калибровка размера**: под H₀ доля p < 0.05 в диапазоне [0.035, 0.065] при ≥ 3000 повторах (биномиальная σ ≈ 0.4 %);
   - **мощность**: под заданной альтернативой отвержение ≥ ожидаемого (например, ≥ 0.8 при стандартизованном эффекте 1 и n = 20);
   - **покрытие ДИ**: доля интервалов, накрывающих истинный параметр, в [0.93, 0.97] для 95 %;
   - **робастность там, где она заявлена**: тяжёлые хвосты (t₃), асимметрия (логнормальное), связки (округлённые данные) — статистика не должна разваливаться, как Бартлетт при куртозисе.
   Бюджет: один Monte-Carlo-тест ≤ 1 с, иначе уменьшить n или повторы и увеличить допуск.

Плюс на каждый метод: строка в `docs/pandas-to-columna.md`, пример в README и обновление отчёта покрытия Minitab.

## Ярус 2 — довести Stat › Basic Statistics и Nonparametrics — ✅ реализован

Файлы: `packages/advanced/src/{basic,nonparametric2,equivalence,power}.ts` + дополнения в `dist.ts`
(`binomial`, `poisson`, `hypergeomPmf`, `nctCdf`, `ncfCdf`, `ncChi2Cdf`). Эталон: `tests/refs/tier2_ref.py`
→ `fixtures/tier2-scipy.json`; тесты `tests/tier2-{basic,nonparametric,power}.test.ts` (40 тестов, ~3 с).
statsmodels в окружении нет — там, где он планировался эталоном, использована прямая формула (проверяемая
в Python независимо от TypeScript-кода).

| # | Функция Minitab | Реализация | Эталон (слой 1) | Синтетика (слой 3) — что подтверждено |
|---|---|---|---|---|
| 2.1 | 1-Sample Z | `ztest1(x, { sigma, mu, alternative, confidence })`, `df.ztest` | формула (z, p, ДИ) | размер 5 % ± 1.5, покрытие 95 % ± 1.5 |
| 2.2 | 1 Proportion / 2 Proportions | `propTest1` (точный биномиальный + Клоппер–Пирсон / нормальный), `propTest2` (pooled z, unpooled ДИ, Fisher exact) | `scipy.binomtest` + `proportion_ci`, `fisher_exact`, формула z | точные тесты консервативны (≤ 5.5 %), z в [3, 7] %, Клоппер–Пирсон ≥ 94.5 %, мощность 0.6 vs 0.3 при n = 25 ≥ 0.75 |
| 2.3 | 1-Sample / 2-Sample Poisson Rate | `poissonRateTest1/2` (точный: Пуассон / условный биномиальный; нормальный; ДИ Гарвуда) | `scipy.poisson`, `chi2.ppf`, `binomtest`, формула z | точные ≤ 5.5 %, покрытие Гарвуда ≥ 94.5 %, мощность λ = 4 vs 1.5 ≥ 0.75 |
| 2.4 | 1 Variance (χ², Бонетт) | `varTest1(x, { sigma0, method })`, `df.varTest`; Бонетт 2006: ДИ exp[ln(c·s²) ± z·c·se], p — наименьший α, исключающий σ₀² | `scipy.chi2` (χ²); Бонетт — инверсия ДИ ↔ p проверяется свойством | χ²: 5 % на нормальных, ≈ 19 % на Лапласе; Бонетт: ≈ 5 % на нормальных, < 10 % на Лапласе при n = 40 (→ 5 % при n → ∞; на t₃ четвёртого момента нет — не калибруется никем) |
| 2.5 | Correlation с p-value и ДИ | `corrTest(a, b, { method, alternative, confidence })`, `df.corrTest`: t-тест, Фишер z-ДИ (Спирмен: Бонетт–Райт) | `scipy.pearsonr` (+ `confidence_interval`), `spearmanr` (со связками) | размер 5 % ± 1.5 (Пирсон/Спирмен), покрытие при ρ = 0.5 в [0.93, 0.97] |
| 2.6 | Outlier Test (Граббс, Диксон) | `grubbs(x, { alpha, alternative })`, `dixon(x)` (r10/r11/r21/r22 по n, нулевое распределение — 20 000 seeded-выборок), `df.outlierTest` | Граббс — формула на примере NIST; Диксон — таблицы Dean–Dixon 1951 / Rorabacher 1991 (± 0.02) | ложные выбросы 5 % ± 1.5; выброс 5σ при n = 12: Граббс ≈ 0.93, Диксон ≈ 0.79 |
| 2.7 | 1-Sample Sign / Wilcoxon | `signTest` (точный биномиальный, ДИ по порядковым статистикам + NLI Хеттманспергера–Шизера), `wilcoxonSigned` (точный DP до n = 50, нормальный с поправками на связки/непрерывность, оценка и ДИ по средним Уолша), `df.signTest` / `df.wilcoxon` | `binomtest`, `scipy.wilcoxon` (exact / approx, связки и нули), формула ДИ Уолша | знаковый: размер ≤ 5.5 % на t₃, покрытие [0.93, 0.98]; Уилкоксон: размер [2.5, 5.5] %, покрытие, мощность > знакового в 1.15+ раз при сдвиге 0.6σ |
| 2.8 | Mood's Median, Friedman, Runs test | `moodMedian` (связки — вниз), `friedman` (поправка на связки), `runsTest` (около среднего, ±0.5 по опции), `df.mood` / `df.friedman` (long-format) / `df.runsTest` | `scipy.median_test(ties='below', correction=False)`, `friedmanchisquare`, формула runs | Муд: размер [2, 6.5] % на t₃, мощность сдвига 1.2 ≥ 0.7; Фридман: [3, 7] %, мощность ≥ 0.7; runs: [3, 7] %, AR(1) φ = 0.7 → ≥ 0.8 |
| 2.9 | Equivalence tests (TOST) | `tost1`, `tost2` (Уэлч / pooled), `tostPaired`; `df.equivalence('y', { limits, by | paired })` | формула (t, p, 90 % ДИ) | на границе — 5 % ± 1.5; внутри (σ/2) — ≥ 95 %; «эквивалентно» ⇔ ДИ(1 − 2α) внутри границ во всех 3000 повторах |
| 2.10 | Power and Sample Size | `power({ test, effect, n, power, alpha, alternative, sigma, p0/p1, groups })` — 9 тестов Minitab, решает любую из трёх величин | `scipy.nct`, `ncf`, `ncx2`, `chi2`, `f` (1e-7); табличные n = 34 / 64 / 32 при d = 0.5, 80 % | эмпирическая мощность реальных тестов (t, 2-t, ANOVA, 1 Variance, 1 Proportion) = расчётной ± 0.025 при 4000 повторах |

## Ярус 3 — Регрессия и модели — ✅ реализован

Файлы: `packages/advanced/src/{linalg,regression,stepwise,glm,lm,nls,pls}.ts`. Эталон: `tests/refs/tier3_ref.py`
→ `fixtures/tier3-scipy.json` (statsmodels/sklearn в окружении нет: OLS — numpy lstsq + замкнутые формулы;
логистика / Пуассон / порядковая / номинальная — прямая максимизация правдоподобия `scipy.optimize` (другой
алгоритм, чем IRLS/Ньютон в TypeScript); Type III — SSE редуцированных моделей; нелинейная — `curve_fit`
+ сертифицированные значения NIST StRD Misra1a; ортогональная — `scipy.odr`; PLS — независимый NIPALS в numpy).
Тесты `tests/tier3-{regression,glm,nls-pls}.test.ts` (32 теста, ~2.5 с).

| # | Функция | Реализация | Эталон (слой 1) | Синтетика (слой 3) — что подтверждено |
|---|---|---|---|---|
| 3.1 | Fitted Line Plot / Regression (OLS) | `ols(y, X, { intercept, weights, confidence })`, `fittedLine(x, y, { degree, logX, logY })`, `df.regress` / `df.fittedLine`: коэффициенты, SE, t, p, ДИ, VIF; S, R², R²(adj), R²(pred), PRESS, AIC/AICc/BIC; ANOVA с последовательными и скорректированными SS; `predict()` с ДИ и интервалом предсказания | numpy (1e-9…1e-10) | β несмещён (2000 повторов), покрытие ДИ β и интервала предсказания в [0.93, 0.97], p при β = 0 → размер 5 % ± 1.5, R²pred < R² всегда |
| 3.2 | Диагностика | в `ols`: leverage, стандартизованные и удалённые (t) остатки, Cook's D, DFFITS, Durbin–Watson, таблица «необычных наблюдений» (R / X по правилам Minitab), aliased-столбцы (ранг через QR) | numpy (1e-9) | — |
| 3.3 | Stepwise / Best Subsets | `stepwise(y, X, { method, alphaIn, alphaOut, include })` с историей шагов; `bestSubsets(y, X, { maxK, nBest, include })`: R², R²(adj), R²(pred), Cp Мэллоуза, S | полный перебор и p-шаги в numpy (точное совпадение путей и величин) | из 2 истинных + 4 шумовых предикторов: истинные входят в 99 %+, ровно они — в ≥ 75 % (α = 0.05) |
| 3.4 | Binary / Ordinal / Nominal Logistic | `glm(family: 'binomial', link: 'logit' \| 'probit' \| 'cloglog', trials, offset)`, `logit`; `ologit` (пропорциональные шансы, знак как в Minitab: logit P(Y ≤ k) = θ_k + x'β); `mlogit` (референс — первый уровень или заданный); odds ratios с ДИ, G-тест, deviance/Pearson GOF, Хосмер–Лемешоу, AIC/BIC, VIF, `predict()` | scipy.optimize (1e-7 коэффициенты, 1e-5 SE порядковой через численный гессиан) | β ≈ несмещён при n = 200, покрытие Wald-ДИ [0.92, 0.98], размер Wald для шумового предиктора [2.5, 8] % |
| 3.5 | Poisson Regression | `poissonRegression(y, X, { offset: log(exposure) })`; rate ratios | scipy.optimize (1e-7) | β = 0.5 восстановлен ± 0.02, покрытие ≥ 92 %, deviance-GOF при средних ≈ 12 отвергает < 10 % |
| 3.6 | General Linear Model | `linearModel(data, 'y ~ a*b + x^2', { factors })`, `df.linearModel('y ~ …')`: эффектное кодирование (−1/0/+1, как Minitab), скорректированные (Type III) и последовательные SS, F, p, коэффициенты, средние по уровням, диагностика; несбалансированные и неполноранговые дизайны | numpy (1e-8) | один фактор = One-Way ANOVA (1e-10); сбалансированный 2×2: Type III = Type I; шумовой фактор в несбалансированном дизайне — размер [3.5, 7] % |
| 3.7 | Nonlinear regression | `nls(model, x, y, { start, names, jacobian?, bounds?, weights? })`: Левенберг–Марквардт, SE/t/p/ДИ из (JᵀJ)⁻¹, корреляция параметров, история итераций, `predict()` с ДИ/PI | NIST StRD Misra1a (сертифицированные b, SE, SSE: 1e-8 / 1e-7 / 1e-9 из «трудного» старта), `curve_fit` | параметры экспоненциального затухания восстанавливаются, покрытие ДИ [0.92, 0.98]; линейная в параметрах модель = OLS (1e-7) |
| 3.8 | Orthogonal / PLS | `orthogonalRegression(x, y, { errorVarianceRatio })` (Деминг, замкнутая форма; SE — jackknife, тесты slope = 1 / intercept = 0); `pls(y, X, { components, standardize, crossValidate })`: NIPALS, R²X/R²Y по компонентам, LOO predicted R² и выбор числа компонент, коэффициенты в исходной шкале, веса/нагрузки/счёты, leverage, несколько откликов | `scipy.odr` (1e-4, SE ± 25 %), NIPALS numpy (1e-8), PLS со всеми компонентами = OLS | ошибки в обеих переменных: ортогональный наклон несмещён (± 0.03), OLS занижен; коллинеарные предикторы: PLS c CV предсказывает лучше OLS в ≥ 80 % выборок |

## Ярус 4 — Quality Tools и Control Charts (SPC) — ✅ реализован

Файлы: `packages/advanced/src/{spc,capability,msa,quality}.ts`. Эталон: `tests/refs/tier4_ref.py`
→ `fixtures/tier4-scipy.json` (ASTM-константы + формулы Minitab / Howe; Box–Cox / Weibull — scipy;
Cohen / Fleiss κ — прямая формула; OC — `scipy.binom`). Тесты `tests/tier4-{spc,capability,msa-quality}.test.ts`
(32 теста, ~1.3 с).

| # | Функция | Реализация | Эталон (слой 1) | Синтетика (слой 3) — что подтверждено |
|---|---|---|---|---|
| 4.1 | Константы контрольных карт | `spcConstants(n)`: A2, A3, B3, B4, D3, D4, d2, c4 | ASTM-таблица + c₄ через Γ (1e-12) | — |
| 4.2 | I-MR, X̄-R, X̄-S, Z-MR | `controlChart(x, { type, subgroup })`, Nelson 1–8, `df.controlChart` | формула MR/d₂, A₂ R̄ (1e-10) | ARL₀ (rule 1) ≫ 200; сдвиг 1σ → ARL₁ < 80 |
| 4.3 | P, NP, C, U, Laney P′/U′ | атрибутивные карты с переменным n | биномиальные пределы P (1e-10) | — |
| 4.4 | EWMA, CUSUM, MA | `ewma(λ, L)`, `cusum(h, k)` (+ V-mask params), `movingAverage` | рекурсия Lucas–Saccucci (1e-10) | EWMA под H₀: доля сигналов в 100 ≈ 0.05…0.35 |
| 4.5 | Capability (normal) | `capability` / `df.capability`: Cp, Cpk, Pp, Ppk, Cpm, PPM, Z.Bench, ДИ Cpk | формулы Minitab на фикстуре | Pp ≈ 1 на N(0,1) ±3; покрытие ДИ Cpk ≥ 90 % |
| 4.6 | Capability nonnormal | `boxCoxLambda`, `johnsonFit`, `weibullFit` (MLE) | scipy boxcox / weibull_min.fit | восстановление λ≈0.5 и Weibull(1.8, 12) |
| 4.7 | Tolerance intervals | `toleranceInterval` (Howe normal + Wilks nonparametric) | Howe k (1e-10) | покрытие ≥ 95 % популяции ≈ 90…99 % при 95/95 |
| 4.8 | Gage R&R / Linearity / Type 1 | `gageRR` (crossed/nested ANOVA), `gageLinearity`, `gageType1` | синтетика с известными σ | σ_repeat ≈ 0.3, σ_part > 2, NDC ≥ 3 |
| 4.9 | Attribute Agreement | `attributeAgreement` — Cohen / Fleiss κ, Kendall W / τ | confusion-matrix / Fleiss формула | — |
| 4.10 | Acceptance sampling | `acceptanceSampling` — OC, AOQ, ATI (attributes + variables k) | `scipy.binom.cdf` | симуляция приёмки = Pa ± 0.02 |
| 4.11 | Pareto, Run chart, Multi-vari, Symmetry, IDI | `pareto`, `runChart`, `multiVari`, `symmetryTest`, `individualDistributionID` | счётчики Pareto; AD по семействам | нормальные данные → normal / Box–Cox(λ≈1) |

## Ярус 5 — Time Series, DOE, Reliability, Multivariate, Predictive — ✅ реализован (+ углубления)

Файлы: `packages/advanced/src/{timeseries,doe,reliability,multivariate,predictive}.ts`.
Эталон: `tests/refs/tier5_ref.py` → `fixtures/tier5-scipy.json`. Тесты `tests/tier5.test.ts` + `tests/tier5-deepen.test.ts`.

| # | Область | Реализация | Эталон (слой 1) | Синтетика (слой 3) |
|---|---|---|---|---|
| 5.1 | Time Series | `trendAnalysis`, `decompose`, **`stl`**, `ets` (+ PI), `acf`/`pacf`/`ccf`, `ljungBox`, `arima` (CSS-ML + **SARIMA/ARIMAX**) | OLS тренд; ACF₁; SES; scipy Weibull MLE | AR(1)/MA(1); seasonal Φ; PI ширится с горизонтом |
| 5.2 | DOE | `fullFactorial`…`taguchi` (L4–L27 OA), `analyzeEffects`, `analyzeDoe`, **`analyzeTaguchi`** (S/N + response tables) | 2³ эффекты; L9 larger S/N | — |
| 5.3 | Reliability | `reliabilityFit` (SE/CI), `kaplanMeier`, **`logRank`**, `probabilityPlot`, `warrantyPrediction` | scipy `weibull_min.fit` | 90 % CI covers truth ≥ 85 % MC; log-rank power |
| 5.4 | Multivariate | `pca`, `factorAnalysis` (PCA + **ML**/varimax), `kmeans`, `hclust`, `discriminant`, `correspondence` | — | PCA ρ=0.8; ML FA loadings; LDA > 0.9 |
| 5.5 | Predictive | `cart`, `randomForest`, `treeNet` (reg/clf), `mars` (**GCV prune**) | — | TreeNet clf ≥ 0.9; MARS prune `nTerms` < max |

## Порядок и оценка

1. ~~Ярус 2 целиком~~ — сделано.
2. ~~Ярус 3 целиком~~ — сделано.
3. ~~Ярус 4 целиком~~ — сделано.
4. ~~Ярус 5 целиком~~ — сделано: Time Series, DOE, Reliability, Multivariate, Predictive (CART/RF/TreeNet/MARS).
5. ~~Углубления яруса 5~~ — сделано: CSS-ML ARIMA + PI, TreeNet classification, Taguchi OA + `analyzeTaguchi`, Reliability SE/CI, ETS PI.
6. ~~Пять пробелов~~ — сделано: SARIMA/ARIMAX, STL, ML FA, log-rank, MARS GCV prune.
7. ~~Highest-value gaps (без charting)~~ — сделано: coxPH + weighted logRank, autoArima, mixture/optimizer, Taguchi L32/L36 + DSD, mixedModel RI.
8. ~~GLMM + Cox extensions~~ — сделано: `glmm` (binomial/Poisson PQL), LMM random slope, Cox strata / shared frailty / counting-process.
9. ~~Ярус 8: все оставшиеся «нет»~~ — сделано (см. таблицу выше). Вне области библиотеки остаются только продуктовые фичи Minitab (Assistant, Session/ReportPad, Worksheet/Project, macros) и полноценный рендерер графиков (данные для всех графиков отдаются).

## Ярус 8 — закрытие оставшихся разрывов — ✅ реализован

Все строки матрицы покрытия со статусом **нет** (кроме продуктовых: Assistant, Session/ReportPad, Worksheet, macros)
и главные **MVP**-ограничения закрыты. Файлы: `packages/advanced/src/{descriptive,stability,spc2,spectral,alias,reliability2,multivariate2,automl,random,optim}.ts`
+ дополнения в `dist.ts` (beta, gamma, weibull, lognormal, exponential, logistic, SEV), `manova.ts` (`manovaModel`, точные собственные значения E⁻¹H),
`multivariate.ts` (исправлена шкала сингулярных чисел CA), `reliability.ts` (MLE дополирован Нелдером–Мидом).
Эталон: `tests/refs/tier8_ref.py` → `fixtures/tier8-scipy.json`; тесты `tests/tier8-gaps-{a,b}.test.ts` (38 тестов, ~4 с).

| Разрыв Minitab | Реализация | Эталон (слой 1) | Синтетика (слой 3) |
|---|---|---|---|
| Display / Store Descriptive Statistics, Graphical Summary | `descriptiveStats` (N, N*, mean, SE, sd, var, CoefVar, sum, SS, min/Q1/med/Q3/max, range, IQR, mode, skew, kurt, MSSD, trimmed; `by`), `graphicalSummary` (A-D, ДИ mean/median/sd, гистограмма, boxplot) | numpy/scipy: квартили type 6, skew/kurt bias=False, trim_mean, ДИ (1e-10) | skew/kurt несмещены при нормальности; ДИ σ покрывает 95 % ± 2 |
| Goodness-of-Fit Test for Poisson | `poissonGof` (пул хвостов до ожидаемых ≥ 5, df = k − 2) | χ² по той же схеме в numpy (1e-8) | размер [2.5, 8] % на Пуассоне; мощность ≥ 90 % против NB |
| Main Effects / Interaction / Interval plots, Boxplot, Dotplot, ECDF | `mainEffectsPlot`, `interactionPlot`, `intervalPlot` (pooled/individual t-ДИ), `boxplotStats` (1.5·IQR, выбросы), `dotplot`, `ecdf` | numpy квартили / замкнутые формы | — |
| Cause-and-Effect | `causeAndEffect` — структура + раскладка + SVG | — | — |
| Stability Study | `stabilityStudy` (batch*time → batch + time → time при α = 0.25; срок годности по односторонней/двусторонней границе на LSL/USL, по партиям и общий) | numpy: p-значения отбора и срок годности бисекцией (1e-6) | одинаковые партии → общая линия; крутой наклон → короче; USL |
| Rare Event: G / T charts | `gChart` (геометрические вероятностные или 3σ пределы, MLE/MVUE p̂), `tChart` (Weibull/экспоненциальные квантили) | scipy geom / weibull_min.fit | ложные тревоги G при известном p < 0.6 % |
| Multivariate: T², MEWMA, Generalized Variance | `t2Chart` (individuals/subgroups, фазы I/II, декомпозиция вкладов), `mewma` (точная ковариация Zᵢ, h калибруется по ARL₀ = 200 симуляцией), `generalizedVarianceChart` | scipy beta/F пределы, numpy T² (1e-8) | доля точек T² за пределом фазы I = α ± 1.5 % |
| Spectral Analysis | `periodogram` (detrend, split-cosine taper, modified Daniell), `cumulativePeriodogram` (тест Бартлетта) | scipy.signal.periodogram (1e-7), FFT | размер теста белого шума [2, 8] %; AR(1) 0.6 → ≥ 95 % |
| Alias Structure | `aliasStructure` (определяющее соотношение, разрешение, цепочки алиасов) | ручные 2⁵⁻¹, 2⁶⁻² | — |
| Regression with Life Data / ALT | `lifeRegression` (AFT: Weibull, lognormal, loglogistic, exponential, SEV/normal/logistic; right/left/interval; перцентили с дельта-ДИ), `altRegression` (Arrhenius / inverse power / exponential / linear, AF, перцентили при рабочем стрессе) | scipy.optimize прямое правдоподобие (1e-6 коэфф., 1e-4 SE) | β, σ восстановлены при 30 % цензурирования; покрытие Wald ≥ 90 % |
| Test Plans | `demonstrationTestPlan` (n или T при c допустимых отказах), `estimationTestPlan` (n по информации Фишера при цензурировании типа I) | замкнутая форма при c = 0 | планируемое n даёт отношение ДИ ≈ цель (MC) |
| Repairable Systems | `powerLawNHPP` (Crow–AMSAA MLE, несколько систем, ДИ, MTBF, тесты тренда Лапласа и MIL-HDBK-189, TTT) | замкнутые формы | размер Лапласа [2.5, 8] %; β = 2 → ≥ 90 % |
| Probit Analysis | `probitAnalysis` (normal/logistic, natural response по Эбботту, ED_p с интервалами Филлера, log-stress) | scipy.optimize (1e-6) | — |
| Cluster Variables | `clusterVariables` (1 − r / 1 − |r|, single/complete/average/ward, `cutTree`) | scipy linkage average (1e-10) | — |
| Multiple Correspondence | `multipleCorrespondence` (indicator / Burt) | numpy SVD (1e-8) | — |
| Item Analysis | `itemAnalysis` (α, стандартизованное α, item-total, SMC, α if deleted) | формулы numpy (1e-10) | — |
| promax | `promax(loadings, { power })` | numpy Hendrickson–White (1e-8) | простая структура сохраняется |
| General MANOVA | `manovaModel(data, responses, 'a*b + x')` — Type III SSCP по термам, 4 статистики, одномерные F | scipy.linalg.eigh(H, E) (1e-9) | размер Pillai для нулевого фактора [3, 7.5] % |
| AutoML / validation | `crossValidate` (k-fold, R²/RMSE/MAE или accuracy/log-loss/confusion), `autoModel` (ols/cart/RF/TreeNet/MARS или logistic/cart/RF/TreeNet) | — | OLS выигрывает на линейных данных, деревья — на ступенчатой функции |
| Random Data / Patterned Data | `random(seed)` — 16 распределений, sample/shuffle; `patterned` | — | моменты 20 000 выборок |
| Квантили по определению Minitab | `col('x').quantile(p, 'minitab')` / `median('minitab')` во всём движке (agg, groupBy, over, broadcast, `describe({ quantileMethod })`), `quantile(x, p, { method })` в advanced; тип 6 — позиция p(n + 1) с усечением к выборке, по умолчанию остаётся тип 7 (pandas / polars) | numpy `method='weibull'` / `'linear'` (1e-10) на всех путях: гистограмма целых, quickselect, сортировка, boxed | монотонность по p, min/max на концах, Q1 ≤ / Q3 ≥ pandas-определения; заодно исправлен `describe` при n > 10 000 и целой позиции ранга (возвращал 0) |

## Скорость

Бенчмарк каждой функции advanced (172 кейса) со сравнением с scipy / numpy / pandas / polars на данных того же
размера и распределения: `pnpm bench:advanced` → [advanced-benchmarks.md](advanced-benchmarks.md)
(`packages/bench/src/advanced-bench.ts`, `python/advanced_compare.py`, `src/advanced-report.ts`).

По итогам первого прогона исправлены алгоритмические дефекты (все — со сверкой с эталонами после правки):

| Функция | Было | Стало | Что изменено |
|---|---|---|---|
| `hclust` n = 1500 | 20 с (O(n⁴)) | 37 мс (= scipy) | NN-chain + Ланс–Уильямс, сортировка слияний как в scipy; добавлен fixture-тест по 4 linkage |
| `stl` n = 10 000 | 40 с | 3.7 с | LOESS: окно из q соседей двумя указателями вместо сортировки всех n точек |
| `coxPH` n = 10 000 | 17 с | 39 мс | частичное правдоподобие одним проходом по убыванию времени (O(n log n)) |
| `kaplanMeier` / `logRank` / `fineGray` | O(n²) | O(n log n) | счётчики по временам вместо фильтров на каждом шаге |
| `mannWhitney` / `wilcoxonSigned` | n₁n₂ / n² разностей и средних Уолша | O(n log n) | k-я порядковая статистика неявного множества (бисекция по значению + точный финиш, `kth.ts`) |
| `periodogram` n = 20 000 | O(n²) ДПФ | FFT (Bluestein для любого n) | `fft.ts` |
| `lifeRegression` n = 10 000 | 877 мс | 49 мс (= scipy.optimize) | стандартизация предикторов + аналитический score |
| `orthogonalRegression` n = 5000 | 176 мс (n подгонок) | 1 мс | jackknife через leave-one-out суммы за O(n) |
| `pca` n = 100 000 | 229 мс | 45 мс | собственные векторы p×p матрицы Грама вместо SVD n×p |
| `nestedAnova` n = 10 000 | 793 мс | 3 мс | ключи уровней строились через `Array.from` внутри цикла |
| `ols` / `linearModel` n = 100 000 | 1.0 / 2.0 с | 0.14 / 0.23 с | VIF из (XᵀX)⁻¹, последовательные SS из Qᵀy, Type III через общую линейную гипотезу — без повторных подгонок |
| `dist.*.ppf` | ~10 мкс | ~7 мкс | грубая бисекция + Ньютон с контролем брекета |
| `powerLawNHPP` | 17 мс | 1.3 мс | наблюдённая информация в замкнутой форме |
| `descriptiveStats` | 400 мс / 1M | 130 мс | мода по отсортированному массиву, группировка одним проходом |

Матрица меню Minitab → статус (**есть** / **MVP** / **нет**): [docs/minitab-coverage.md](minitab-coverage.md).
