# Categorical Analysis Guide

## Goal
Evaluate how categorical features are related to fraud target (`target` = 1).

## Required inputs
- Train parts: `src/data/train_part_*.parquet`
- Labels file with columns `event_id,target`:
  - default path: `src/data/train_labels.csv`
  - also supports parquet

## Run
```bash
python3 src/features/categorical_eda.py \
  --train-glob "src/data/train_part_*.parquet" \
  --labels-path "src/data/train_labels.csv" \
  --out-dir "outputs/categorical_analysis"
```

If your labels file is parquet:
```bash
python3 src/features/categorical_eda.py \
  --labels-path "src/data/train_target.parquet"
```

## Outputs
- `outputs/categorical_analysis/categorical_feature_metrics.csv`
  - one row per feature
  - includes:
    - `mutual_info`
    - `chi2_stat`, `chi2_p_value`
    - `cramers_v`
    - `kruskal_h`, `kruskal_p_value`
- `outputs/categorical_analysis/feature_summaries/<feature>_summary.csv`
  - category-level frequency and target rate stats

## Notes
- `Kruskal-Wallis` is included because you asked for it, but for categorical vs binary target
  the more informative checks are usually:
  - `chi2_p_value`
  - `cramers_v`
  - `mutual_info`
- Missing values are treated as a separate category (`__MISSING__`).
