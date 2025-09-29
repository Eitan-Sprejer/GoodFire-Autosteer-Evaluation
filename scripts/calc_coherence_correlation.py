#!/usr/bin/env python3
"""
Compute correlation between coherence and accuracy for all CSVs in `results/` that end with `_with_subject`.

Outputs:
 - prints per-file instance-level Pearson/Spearman correlations
 - prints per-file method-aggregated correlations (mean coherence vs accuracy)
 - writes `results/correlations_summary.csv` with summary rows per file

Usage:
    python scripts/calc_coherence_correlation.py

This script is self-contained and uses only pandas and scipy (if available).
If scipy is not installed, Spearman correlation is skipped.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr
    _has_scipy = True
except Exception:
    _has_scipy = False


RESULTS_DIR = Path("results")
OUT_FILE = RESULTS_DIR / "correlations_summary.csv"


def safe_read_csv(p: Path) -> pd.DataFrame:
    for enc in (None, "utf-8", "latin-1"):
        try:
            return pd.read_csv(p, encoding=None if enc is None else enc)
        except Exception:
            continue
    return pd.read_csv(p, engine="python")


def compute_instance_level_corr(df: pd.DataFrame):
    """Compute correlation between coherence_num and is_correct at instance level.
    Returns a dict with n, pearson_r, pearson_p, spearman_r, spearman_p (spearman may be None).
    """
    out = {"n": 0, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    if "coherence_num" not in df.columns or "is_correct" not in df.columns:
        return out
    sub = df[["coherence_num", "is_correct"]].dropna()
    if sub.empty:
        return out
    # ensure numeric
    x = pd.to_numeric(sub["coherence_num"], errors="coerce")
    y = sub["is_correct"].astype(float)
    mask = x.notna()
    x = x[mask]
    y = y[mask]
    out["n"] = len(x)
    if len(x) < 2:
        return out
    if _has_scipy:
        try:
            r, p = pearsonr(x, y)
            out["pearson_r"] = float(r)
            out["pearson_p"] = float(p)
        except Exception:
            pass
        try:
            sr, sp = spearmanr(x, y)
            out["spearman_r"] = float(sr)
            out["spearman_p"] = float(sp)
        except Exception:
            pass
    else:
        # fallback: compute Pearson via numpy (no p-value)
        try:
            r = np.corrcoef(x, y)[0, 1]
            out["pearson_r"] = float(r)
        except Exception:
            pass
    return out


def compute_method_agg_corr(df: pd.DataFrame, method_col: str = "steering_method"):
    """Aggregate by method: mean coherence and accuracy per method, then compute correlation across methods."""
    out = {"n_methods": 0, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    if method_col not in df.columns or "coherence_num" not in df.columns or "is_correct" not in df.columns:
        return out
    agg = df.groupby(method_col, dropna=False).agg({"coherence_num": "mean", "is_correct": "mean"}).dropna()
    if agg.empty:
        return out
    x = pd.to_numeric(agg["coherence_num"], errors="coerce")
    y = agg["is_correct"].astype(float)
    mask = x.notna()
    x = x[mask]
    y = y[mask]
    out["n_methods"] = len(x)
    if len(x) < 2:
        return out
    if _has_scipy:
        try:
            r, p = pearsonr(x, y)
            out["pearson_r"] = float(r)
            out["pearson_p"] = float(p)
        except Exception:
            pass
        try:
            sr, sp = spearmanr(x, y)
            out["spearman_r"] = float(sr)
            out["spearman_p"] = float(sp)
        except Exception:
            pass
    else:
        try:
            r = np.corrcoef(x, y)[0, 1]
            out["pearson_r"] = float(r)
        except Exception:
            pass
    return out


def process_file(p: Path):
    df = safe_read_csv(p)
    # Ensure flags exist: coherence_num and is_correct
    # If file already contains 'coherence_num' and 'is_correct' use them; otherwise try to infer
    if "coherence_num" not in df.columns and "coherence" in df.columns:
        df["coherence_num"] = pd.to_numeric(df["coherence"], errors="coerce")
    if "is_correct" not in df.columns:
        if "result" in df.columns:
            df["result_norm"] = df["result"].astype(str)
            df["is_correct"] = df["result_norm"].str.lower().str.startswith("hit")
        else:
            # fallback to predicted==gold
            if "predicted_answer" in df.columns and "gold_answer" in df.columns:
                df["is_correct"] = (df["predicted_answer"].astype(str) == df["gold_answer"].astype(str))
            else:
                df["is_correct"] = np.nan

    inst = compute_instance_level_corr(df)
    agg = compute_method_agg_corr(df)

    summary = {
        "file": p.name,
        "n_instances": inst.get("n", 0),
        "inst_pearson_r": inst.get("pearson_r"),
        "inst_pearson_p": inst.get("pearson_p"),
        "inst_spearman_r": inst.get("spearman_r"),
        "inst_spearman_p": inst.get("spearman_p"),
        "n_methods": agg.get("n_methods", 0),
        "method_pearson_r": agg.get("pearson_r"),
        "method_pearson_p": agg.get("pearson_p"),
        "method_spearman_r": agg.get("spearman_r"),
        "method_spearman_p": agg.get("spearman_p"),
    }
    return summary


def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)
    files = sorted(RESULTS_DIR.glob("*_with_subject.csv")) + sorted(RESULTS_DIR.glob("*_with_subject"))
    if not files:
        print("No files found matching '*_with_subject.csv' or '*_with_subject' in results/")
        sys.exit(0)

    rows = []
    for p in files:
        try:
            print(f"Processing {p.name}...")
            rows.append(process_file(p))
        except Exception as e:
            print(f"Error processing {p.name}: {e}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_FILE, index=False)
    print("Saved summary to:", OUT_FILE)
    print(df_out)


if __name__ == "__main__":
    main()
