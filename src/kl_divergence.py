# This script computes the KL divergence between the finite width and the infinite width distributions

import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, entropy

def compute_kl(emp_vals, theo_vals, xs):
    # Numerically compute D_KL(emp || theo) on grid xs
    kde_emp = gaussian_kde(emp_vals)
    kde_theo = gaussian_kde(theo_vals)
    p_emp = kde_emp(xs) + 1e-12
    p_theo = kde_theo(xs) + 1e-12
    p_emp /= np.trapezoid(p_emp, xs)
    p_theo /= np.trapezoid(p_theo, xs)
    return entropy(p_emp, p_theo)

def process_pattern(pattern):
    # Read all CSV files matching pattern, compute KL for each file (seed)
    # Return DataFrame with index=seed, columns=params

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No files match {pattern}')
    # Determine parameter values from first file
    df0 = pd.read_csv(files[0])
    params = sorted(df0['param'].unique())
    # Build evaluation grid once over union of all files
    all_vals = []
    for f in files:
        d = pd.read_csv(f)
        all_vals.append(d['y_emp'].values)
        all_vals.append(d['y_theo'].values)
    all_vals = np.concatenate(all_vals)
    xs = np.linspace(all_vals.min(), all_vals.max(), 500)

    # Compute KL per file
    records = {}
    for fpath in files:
        # Extract seed as the digits after last underscore
        fname = os.path.basename(fpath)
        m = re.search(r'_(\d+)\.csv$', fname)
        if not m:
            continue
        seed = int(m.group(1))
        df = pd.read_csv(fpath)
        kl_list = []
        for p in params:
            emp = df[df['param']==p]['y_emp'].values
            theo = df[df['param']==p]['y_theo'].values
            kl_list.append(compute_kl(emp, theo, xs))
        records[seed] = kl_list

    # Build DataFrame: index=seed, columns=params
    kl_df = pd.DataFrame.from_dict(
        records, orient='index', columns=[str(p) for p in params]
    ).sort_index()
    return kl_df

if __name__ == '__main__':
    # Process all seeds for vary_n
    kl_n = process_pattern('data/data_vary_n_*.csv')
    kl_n.to_csv('data/kl_vary_n.csv', index_label='seed')

    # Process all seeds for vary_H
    kl_H = process_pattern('data/data_vary_H_*.csv')
    kl_H.to_csv('data/kl_vary_H.csv', index_label='seed')
