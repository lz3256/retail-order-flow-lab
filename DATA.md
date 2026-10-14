# Inputs and reproduction

The public repository includes code and aggregate results only. Use your own authorized CRSP/TAQ-derived inputs. Do not redistribute market panels or credentials.

The measurement runner expects `RETAIL_FLOW_INPUT_ROOT` to contain:

- `calendar.csv`, with the trading `date` calendar.
- `linked_manifest.json`, with `complete=true` and SHA256 fingerprints keyed by daily file name.
- `linked/YYYY-MM-DD.parquet`, with `date`, `permno`, `both_signed`, `direction_disagreements`, `qmp_buy_volume`, `qmp_sell_volume`, `qmp_signed_trades`, `bjzz_common_buy_volume`, `bjzz_common_sell_volume`, `bjzz_common_signed_trades`, `clean_volume`, and `mean_quote_age_seconds`.
- `analysis/{development,holdout,post_pilot}/weekly_panel.parquet` plus `run_manifest.json` for each period. The panels contain `signal_date`, `entry_date`, `exit_date`, `permno`, `industry`, `delisted_at_signal`, `qmp`, `bjzz_common`, `week1`, and controls `past5`, `past20`, `logcap`, `vol20`, `logadv20`, `relative_spread`, `beta`. See the numerical implementation for full required columns and inherited-original-summary checks.

The inherited panel itself must be built using a historical monthly universe and outcome timing consistent with the recorded protocol. Supplying unrelated panels is a new experiment, not reproduction of the published result.

From a fresh copy of the research directory, set `RETAIL_FLOW_INPUT_ROOT` to the prepared-input directory, then run:

```bash
python code/configure.py
python code/experiment.py
python -m unittest discover -s code -p 'test_*.py' -v
```

`configure.py` creates a new local protocol and receipt and refuses to overwrite them. That new timestamp records a reproduction attempt; it does not make the historical sample unseen. Cached derived panels and per-stock forecasts stay in ignored `work/` directories. The published numerical tests run without these inputs.
