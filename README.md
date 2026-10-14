# Retail Order Flow: Measurement Error in Trade Signing

**How much does signing-rule choice alter a retail-flow signal, and do disagreement-based adjustments improve its predictions?**

The research compares BJZZ subpenny and quote-midpoint-based signals in a point-in-time, monthly-reconstituted 1,000-stock US universe over 2020–2025. Six signal variants share matched stock/window samples across 258 non-overlapping weekly formation windows. A weekly formation schedule does not imply every calendar week is retained.

## Recorded findings

- **Measurement sensitivity:** BJZZ and quote-midpoint signals had mean within-window cross-sectional Spearman correlations of **0.50–0.53 across sample periods**. Disagreement-based reweighting, in contrast, barely changed the original quote-midpoint rankings (**ρ = 0.9985–0.9992**).
- **Predictive validation:** Disagreement-based adjustments produced no reliable rank-IC or expanding-ridge MSE improvement across **24 Holm-adjusted comparisons**. Targeted stock-day exclusion did not consistently beat **20 seeded random-deletion controls**.
- **Mechanism checks:** Independent-flip simulations examine how effective trade count and volume concentration affect aggregation noise. The variance identity `4e(1−e)/n_eff` was checked across **72 settings and 576 runs**. This is an assumption-based sensitivity calculation, not a novel variance theorem or an estimate of actual signing accuracy.

The baseline signal's mean rank IC was not significant in the current matched-sample periods. No improvement from a correction does not establish robustness of a profitable baseline.

## Start here

- [Detailed research report and interpretation](research/2026-10-13/README.md)
- [Recorded exploratory design](research/2026-10-13/protocols/exploratory_design.json)
- [Data contract and reproduction](DATA.md)
- [Aggregate comparisons](research/2026-10-13/results/primary_contrasts.csv)
- [Numerical method source](research/2026-10-13/code/experiment.py)

Run the mathematical and temporal method checks:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s research/2026-10-13/code -p 'test_*.py' -v
```

These tests use constructed examples. The historical forecast-fence check is skipped unless its local forecast artifact is supplied; data-independent tests do not reproduce market findings.

## Interpretation boundaries

Both signing rules are proxies, not observed true retail direction. Disagreement is not a known error probability. Stock samples are matched, but the BJZZ and quote-midpoint signed-trade sets can differ; the ranking comparison does not isolate direction from trade coverage. Exact jointly-signed buy/sell volume was unavailable in this completed study.

Historical development/holdout/post-pilot labels are inherited from earlier work; every period had already been inspected before this extension. Forecast-loss resampling is conditional on fitted models, without full training refits. Cost-adjusted trading gains, a causal measurement-error effect, and new-period confirmation are not established. A subsequent trade-coverage decomposition was still incomplete in the reviewed record and is not published as a result here.

## Attribution

Project owner: [Linfeng Zhao (@lz3256)](https://github.com/lz3256). Developed with substantial assistance from OpenAI Codex for implementation, analysis, debugging and documentation. Empirical inputs use WRDS CRSP and TAQ; raw records, processed panels, per-stock predictions and credentials are not distributed.

The methodological context includes [Barber et al., *A (Sub)penny for Your Thoughts*](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13334) and [Ardia, Aymard and Cenesizoglu, *Revisiting Boehmer et al. (2021)*](https://arxiv.org/abs/2403.17095). Published error rates are not imported as truth for this sample.
