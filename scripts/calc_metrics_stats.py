"""
Generates statistics by method for each result CSV file.

This script searches all files in `results/` that match `*_with_subject.csv` and,
for **each** file, calculates:
- mean and standard deviation by method for `coherence` and `behavior`
- the count (n) by method

For each input file, it writes a table to:
`results/<stem>_methods_stats.csv`
and also prints the table to stdout.

Usage: python scripts/calc_coherence_stats_per_file.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")


def find_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*_with_subject.csv"))


def read_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def collect_method_stats_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates by method: mean/std/count for coherence and behavior.
    Returns a DataFrame with the following columns:
      method, Coherence Mean, Coherence Std, Behaviour Mean, Behaviour Std, n
    """
    required_cols = {"coherence", "behavior", "steering_method"}
    if not required_cols.issubset(df.columns):
        # Return an empty DataFrame with the expected schema for easier handling
        return pd.DataFrame(
            columns=[
                "method",
                "Coherence Mean",
                "Coherence Std",
                "Behaviour Mean",
                "Behaviour Std",
                "n",
            ]
        )

    # Group and calculate aggregates
    stats_df = (
        df.groupby("steering_method")[["coherence", "behavior"]]
        .agg(["mean", "std", "count"])
        .swaplevel(axis=1)
    )

    # Flatten columns
    stats_df.columns = [f"{stat}_{col}" for stat, col in stats_df.columns]
    stats_df = stats_df.reset_index().rename(columns={"steering_method": "method"})

    # Select and rename to final format
    out = stats_df[
        [
            "method",
            "mean_coherence",
            "std_coherence",
            "mean_behavior",
            "std_behavior",
            "count_coherence",
        ]
    ].rename(
        columns={
            "mean_coherence": "Coherence Mean",
            "std_coherence": "Coherence Std",
            "mean_behavior": "Behaviour Mean",
            "std_behavior": "Behaviour Std",
            "count_coherence": "n",
        }
    )

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        "-r",
        default=str(RESULTS_DIR),
        help="Path to the directory containing *_with_subject.csv files",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results dir not found: {results_dir}")
        return 2

    files = find_files(results_dir)
    if not files:
        print("No '*_with_subject.csv' files found in results dir")
        return 0

    pd.set_option("display.max_columns", None)

    for p in files:
        try:
            df = read_df(p)
        except Exception as e:
            warnings.warn(f"Failed to read {p}: {e}")
            continue

        methods_df = collect_method_stats_for_df(df)

        # Output file name per input file
        stem = p.stem#.replace("_with_subject", "")
        out_path = results_dir / f"{stem}_methods_stats.csv"
        methods_df.to_csv(out_path, index=False)
        print(f"\nWrote per-method stats for '{p.name}' to {out_path}")

        # Print the table to stdout (one per file)
        print(f"Per-method stats for: {p.name}")
        if methods_df.empty:
            # Clear message if necessary columns are missing
            missing_cols = {"coherence", "behavior", "steering_method"} - set(df.columns)
            print(f"(no data: missing columns {sorted(missing_cols)})")
        else:
            print(methods_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
