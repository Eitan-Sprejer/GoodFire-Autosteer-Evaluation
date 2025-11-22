#!/usr/bin/env python3
"""
Run pairwise t-tests between all steering methods for accuracy, behavior, and coherence.
Applies Bonferroni correction for multiple comparisons.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from pathlib import Path

# Paths
RESULTS_DIR = Path("results")

# Load data
df_70b = pd.read_csv(RESULTS_DIR / "eval_gpt-4o-mini_var_Llama-3.3-70B-Instruct_dt_20250929_1404_answers_with_subject.csv")
df_8b = pd.read_csv(RESULTS_DIR / "eval_gpt-4o-mini_var_Meta-Llama-3.1-8B-Instruct_dt_20250929_1201_answers_with_subject.csv")

# Prepare data
df_70b['model'] = 'Llama-3.3-70B'
df_8b['model'] = 'Llama-3.1-8B'

# Create binary accuracy
df_70b['is_correct'] = df_70b['result'].astype(str).str.lower().str.startswith('hit')
df_8b['is_correct'] = df_8b['result'].astype(str).str.lower().str.startswith('hit')

# Convert to numeric
df_70b['coherence_num'] = pd.to_numeric(df_70b['coherence'], errors='coerce')
df_70b['behavior_num'] = pd.to_numeric(df_70b['behavior'], errors='coerce')
df_8b['coherence_num'] = pd.to_numeric(df_8b['coherence'], errors='coerce')
df_8b['behavior_num'] = pd.to_numeric(df_8b['behavior'], errors='coerce')

# Define methods
methods = ['Control', 'Simple Prompting', 'Auto Steer', 'Combined Approach']

# Number of comparisons per metric per model: C(4,2) = 6
n_comparisons_per_metric = len(list(combinations(methods, 2)))
# Total comparisons: 6 comparisons × 3 metrics × 2 models = 36
total_comparisons = n_comparisons_per_metric * 3 * 2
bonferroni_alpha = 0.05 / total_comparisons

print("="*100)
print(f"PAIRWISE T-TESTS WITH BONFERRONI CORRECTION")
print(f"Number of pairwise comparisons per metric per model: {n_comparisons_per_metric}")
print(f"Total comparisons: {total_comparisons}")
print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.6f}")
print("="*100)

def run_pairwise_ttests(df, model_name, metric_name, metric_col):
    """Run pairwise t-tests for a given metric."""
    print(f"\n{'='*100}")
    print(f"{model_name} - {metric_name.upper()}")
    print(f"{'='*100}")

    results = []

    for method1, method2 in combinations(methods, 2):
        # Get data for each method
        data1 = df[df['steering_method'] == method1][metric_col].dropna()
        data2 = df[df['steering_method'] == method2][metric_col].dropna()

        # Run t-test
        t_stat, p_value = stats.ttest_ind(data1, data2)

        # Calculate means and effect size (Cohen's d)
        mean1 = data1.mean()
        mean2 = data2.mean()
        pooled_std = np.sqrt(((len(data1)-1)*data1.std()**2 + (len(data2)-1)*data2.std()**2) / (len(data1)+len(data2)-2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

        # Determine significance
        is_significant = p_value < bonferroni_alpha
        sig_marker = "***" if is_significant else ""

        results.append({
            'Comparison': f'{method1} vs {method2}',
            'Mean 1': f'{mean1:.4f}',
            'Mean 2': f'{mean2:.4f}',
            'Diff': f'{mean1 - mean2:.4f}',
            't-statistic': f'{t_stat:.4f}',
            'p-value': f'{p_value:.6f}',
            'Significant': sig_marker,
            "Cohen's d": f'{cohens_d:.4f}',
            'n1': len(data1),
            'n2': len(data2)
        })

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    return df_results

# Store all results
all_results = []

# Run tests for Llama-3.3-70B
for metric_name, metric_col in [('Accuracy', 'is_correct'),
                                  ('Behavior', 'behavior_num'),
                                  ('Coherence', 'coherence_num')]:
    df_result = run_pairwise_ttests(df_70b, 'Llama-3.3-70B', metric_name, metric_col)
    df_result['Model'] = 'Llama-3.3-70B'
    df_result['Metric'] = metric_name
    all_results.append(df_result)

# Run tests for Llama-3.1-8B
for metric_name, metric_col in [('Accuracy', 'is_correct'),
                                  ('Behavior', 'behavior_num'),
                                  ('Coherence', 'coherence_num')]:
    df_result = run_pairwise_ttests(df_8b, 'Llama-3.1-8B', metric_name, metric_col)
    df_result['Model'] = 'Llama-3.1-8B'
    df_result['Metric'] = metric_name
    all_results.append(df_result)

# Combine all results
df_all = pd.concat(all_results, ignore_index=True)

# Save to CSV
output_file = RESULTS_DIR / "pairwise_ttests_bonferroni.csv"
df_all.to_csv(output_file, index=False)

print("\n" + "="*100)
print(f"SUMMARY")
print("="*100)
print(f"✓ All results saved to: {output_file}")
print(f"✓ Bonferroni-corrected significance threshold: p < {bonferroni_alpha:.6f}")
print(f"✓ Comparisons marked with '***' are statistically significant after correction")
print("="*100)
