# Measurement error and predictive robustness: completed historical extension

## Design

Six configurations: original QMP, BJZZ on common valid-quote eligible prints, disagreement-weighted QMP, QMP after removal of the highest-disagreement day, QMP after a seeded random day removal, and an equal QMP/BJZZ proxy blend. All use the complete matched formation/control sample; available outcomes are matched separately.

The original universe is reconstituted monthly using point-in-time information. There are 258 eligible weekly formation windows over January 3, 2020–December 19, 2025, partitioned into 131 development, 88 inherited historical-holdout and 39 post-pilot windows. Forecasts require at least 52 matured training windows. The expanding ridge model has fixed penalty 1, original controls and industry indicators, with no outcome-selected hyperparameters.

Twenty-four main tests are four alternatives versus QMP × three periods × two metrics (rank IC and forecast MSE). Holm adjustment covers all 24; paired circular block intervals use four windows, with eight-window sensitivity. Three secondary targeted-versus-random deletion comparisons use their own Holm family.

## Findings and mechanisms

| Period | BJZZ/QMP mean rank correlation | Reweighted/original QMP mean rank correlation | Original QMP mean rank IC | QMP HAC p |
|---|---:|---:|---:|---:|
| Development | 0.4973 | 0.9985 | −0.0011 | 0.761 |
| Historical holdout | 0.5319 | 0.9987 | −0.0084 | 0.161 |
| Post-pilot | 0.4981 | 0.9992 | −0.0042 | 0.580 |

These are period averages of within-window stock-ranking correlations, not the range of every individual window. The trade membership of the two rules differs, so the first column captures more than just sign changes on identical trades.

A common positive constant weight over a stock's aggregation window cancels in normalized order imbalance. Actual day-varying weights do not cancel exactly. The exact adjustment identity is `weighted OIB − original OIB = Cov_p(weight, daily OIB) / E_p(weight)`, where `p` is proportional to daily signed volume. This outcome-free identity was added after the predictive results; it is an explanatory diagnostic, not an independent prediction of every null result.

Under independent sign flips with probability `e` within an aggregation unit, conditional noise variance is `4e(1−e)/n_eff`, with `n_eff = 1/sum(w_i²)` for normalized trade-volume weights. The simulation varies trade counts, volume concentration, flip rates and stock-dependent rates. A positive common rescaling preserves ranks; random noise can change ranks. Rank IC must not simply be multiplied by `1−2e`.

See `results/` for aggregate metrics. No new untouched period, true trade-direction labels, trade-level agreement-only signal, or cost-adjusted trading benefit was established.

## Source

The `code/` directory contains the numerical implementation and substantive tests. Public path configuration is adapted from the recorded run; numerical definitions are unchanged. `protocols/` preserves the historical design and outcome-free addendum with private paths removed. Empirical reproduction requires privately supplied licensed inputs; see the repository's `DATA.md`.
