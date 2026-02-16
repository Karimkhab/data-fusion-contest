#!/usr/bin/env python3
"""Categorical feature analysis for Data Fusion Contest train data.

The script reads large parquet train parts in batches, keeps only rows that
have labels, and computes per-feature statistics for binary target influence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import chi2_contingency, kruskal
from sklearn.feature_selection import mutual_info_classif


MISSING_TOKEN = "__MISSING__"
DEFAULT_TARGET_COL = "target"
DEFAULT_EVENT_COL = "event_id"

DEFAULT_CATEGORICAL_COLUMNS = [
    "event_type_nm",
    "event_desc",
    "channel_indicator_type",
    "channel_indicator_sub_type",
    "currency_iso_cd",
    "mcc_code",
    "pos_cd",
    "accept_language",
    "browser_language",
    "timezone",
    "operating_system_type",
    "device_system_version",
    "screen_size",
    "developer_tools",
    "phone_voip_call_state",
    "web_rdp_connection",
    "compromised",
]


def _load_labels(
    labels_path: Path,
    event_col: str = DEFAULT_EVENT_COL,
    target_col: str = DEFAULT_TARGET_COL,
) -> pd.DataFrame:
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}. "
            "Put labels into src/data and pass --labels-path."
        )

    if labels_path.suffix.lower() == ".parquet":
        labels = pd.read_parquet(labels_path, columns=[event_col, target_col])
    else:
        labels = pd.read_csv(labels_path, usecols=[event_col, target_col])

    labels = labels.dropna(subset=[event_col, target_col]).copy()
    labels[event_col] = pd.to_numeric(labels[event_col], errors="coerce")
    labels[target_col] = pd.to_numeric(labels[target_col], errors="coerce")
    labels = labels.dropna(subset=[event_col, target_col]).copy()
    labels[event_col] = labels[event_col].astype("int64")
    labels[target_col] = labels[target_col].astype("int8")
    labels = labels.drop_duplicates(subset=[event_col], keep="last")
    return labels


def _collect_labeled_rows(
    train_files: Iterable[Path],
    label_event_ids: set[int],
    columns: list[str],
    batch_size: int = 500_000,
    event_col: str = DEFAULT_EVENT_COL,
) -> pd.DataFrame:
    out_chunks: list[pd.DataFrame] = []
    read_cols = [event_col, *columns]

    for train_file in train_files:
        parquet_file = pq.ParquetFile(train_file)
        for batch in parquet_file.iter_batches(columns=read_cols, batch_size=batch_size):
            chunk = batch.to_pandas()
            mask = chunk[event_col].isin(label_event_ids)
            if mask.any():
                out_chunks.append(chunk.loc[mask, read_cols].copy())

    if not out_chunks:
        return pd.DataFrame(columns=read_cols)

    df = pd.concat(out_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=[event_col], keep="last")
    return df


def _prepare_cat(series: pd.Series) -> pd.Series:
    as_str = series.astype("string")
    return as_str.fillna(MISSING_TOKEN)


def ordinal_check(
    df: pd.DataFrame,
    feature_name: str,
    target_col: str = DEFAULT_TARGET_COL,
) -> pd.DataFrame:
    """Category ranking by target rate/median (for notebook quick checks)."""
    s = _prepare_cat(df[feature_name])
    ranked = (
        pd.DataFrame({"feature": s, target_col: df[target_col].values})
        .groupby("feature", dropna=False)[target_col]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "target_rate", "median": "target_median"})
        .sort_values("target_rate")
    )
    return ranked


def feature_counts(df: pd.DataFrame, feature_name: str, normalize: bool = False) -> pd.Series:
    s = _prepare_cat(df[feature_name])
    return s.value_counts(normalize=normalize, dropna=False)


def summarize_categorical(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str = DEFAULT_TARGET_COL,
) -> pd.DataFrame:
    s = _prepare_cat(df[cat_col])
    counts = s.value_counts(dropna=False).rename("count")
    grouped = (
        pd.DataFrame({"cat": s, target_col: df[target_col].values})
        .groupby("cat", dropna=False)[target_col]
        .agg(["mean", "median", "std", "sum"])
        .rename(columns={"mean": "target_rate", "sum": "target_positives"})
    )
    summary = pd.concat([counts, grouped], axis=1)
    summary["target_negatives"] = summary["count"] - summary["target_positives"]
    summary["freq"] = summary["count"] / len(df)
    return summary.sort_values("target_rate", ascending=False)


def cat_mutual_info(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str = DEFAULT_TARGET_COL,
) -> float:
    s = _prepare_cat(df[cat_col])
    encoded, _ = pd.factorize(s, sort=False)
    x = encoded.reshape(-1, 1)
    y = df[target_col].values
    return float(mutual_info_classif(x, y, discrete_features=True, random_state=42)[0])


def _chi2_cramers_v(cat: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    contingency = pd.crosstab(cat, y)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return np.nan, np.nan, np.nan

    chi2, p_value, _, _ = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    if n <= 1:
        return float(chi2), float(p_value), np.nan

    phi2 = chi2 / n
    r, k = contingency.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    cramers_v = np.sqrt(phi2_corr / denom) if denom > 0 else np.nan
    return float(chi2), float(p_value), float(cramers_v)


def _kruskal_for_categorical(cat: pd.Series, y: pd.Series) -> tuple[float, float]:
    groups = [y[cat == val].to_numpy() for val in cat.unique()]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return np.nan, np.nan
    h_stat, p_value = kruskal(*groups, nan_policy="omit")
    return float(h_stat), float(p_value)


def run_categorical_analysis(
    train_glob: str,
    labels_path: str,
    out_dir: str,
    categorical_columns: list[str] | None = None,
    event_col: str = DEFAULT_EVENT_COL,
    target_col: str = DEFAULT_TARGET_COL,
    batch_size: int = 500_000,
) -> pd.DataFrame:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    feature_out = out_path / "feature_summaries"
    feature_out.mkdir(parents=True, exist_ok=True)

    train_files = sorted(Path().glob(train_glob))
    if not train_files:
        raise FileNotFoundError(f"No train files found by glob: {train_glob}")

    labels = _load_labels(Path(labels_path), event_col=event_col, target_col=target_col)
    label_event_ids = set(labels[event_col].tolist())
    if not label_event_ids:
        raise ValueError("Labels are empty after cleaning.")

    columns = categorical_columns or DEFAULT_CATEGORICAL_COLUMNS
    columns = list(dict.fromkeys(columns))

    labeled_part = _collect_labeled_rows(
        train_files=train_files,
        label_event_ids=label_event_ids,
        columns=columns,
        batch_size=batch_size,
        event_col=event_col,
    )
    if labeled_part.empty:
        raise RuntimeError("No labeled events were found inside train parquet files.")

    df = labels.merge(labeled_part, on=event_col, how="inner")
    if df.empty:
        raise RuntimeError("Join between labels and train data produced 0 rows.")

    available_cols = [c for c in columns if c in df.columns]
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        print("Warning: missing feature columns:", ", ".join(missing_cols))

    metrics_rows = []
    y = df[target_col].astype("int8")

    for col in available_cols:
        cat = _prepare_cat(df[col])
        chi2, chi2_p, cramers_v = _chi2_cramers_v(cat, y)
        kr_h, kr_p = _kruskal_for_categorical(cat, y)
        mi = cat_mutual_info(df, col, target_col=target_col)
        n_unique = int(cat.nunique(dropna=False))

        feature_summary = summarize_categorical(df, col, target_col=target_col)
        feature_summary.to_csv(feature_out / f"{col}_summary.csv")

        metrics_rows.append(
            {
                "feature": col,
                "n_unique": n_unique,
                "mutual_info": mi,
                "chi2_stat": chi2,
                "chi2_p_value": chi2_p,
                "cramers_v": cramers_v,
                "kruskal_h": kr_h,
                "kruskal_p_value": kr_p,
            }
        )

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["mutual_info", "cramers_v"], ascending=False
    )
    metrics.to_csv(out_path / "categorical_feature_metrics.csv", index=False)

    df[["event_id", "target"]].to_csv(out_path / "labeled_event_index.csv", index=False)

    print("Rows after label join:", len(df))
    print("Positive rate:", float(df[target_col].mean()))
    print("Top 10 features by MI:")
    print(metrics[["feature", "mutual_info", "cramers_v", "chi2_p_value"]].head(10))
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Categorical EDA for Data Fusion train data")
    parser.add_argument(
        "--train-glob",
        default="src/data/train_part_*.parquet",
        help="Glob pattern for train parquet parts.",
    )
    parser.add_argument(
        "--labels-path",
        default="src/data/train_labels.csv",
        help="Path to labels file (csv/parquet) with event_id,target.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/categorical_analysis",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--categorical-columns-json",
        default="",
        help="Optional JSON list of categorical columns.",
    )
    parser.add_argument("--event-col", default=DEFAULT_EVENT_COL, help="Event id column name.")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL, help="Target column name.")
    parser.add_argument(
        "--batch-size",
        default=500_000,
        type=int,
        help="Batch size for parquet scanning.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.categorical_columns_json:
        categorical_columns = json.loads(args.categorical_columns_json)
    else:
        categorical_columns = None

    run_categorical_analysis(
        train_glob=args.train_glob,
        labels_path=args.labels_path,
        out_dir=args.out_dir,
        categorical_columns=categorical_columns,
        event_col=args.event_col,
        target_col=args.target_col,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
