# This script reads CSV outputs and generates plots
# Save via SVG->PDF, then remove SVG

import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, entropy
import plotly.graph_objects as go
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

# plot settings
width = 600
height = 400
margin = dict(
    l=20,
    r=20,
    t=20,
    b=20
)
font = dict(
    family='Times New Roman',
    size=24,
    color='black'
)
grid = dict(
    showgrid=True,
    gridcolor='lightgray',
    zeroline=True,
    zerolinecolor='lightgray'
)

def compute_kl_divergences(csv_file: str):
    df = pd.read_csv(csv_file)
    params = sorted(df['param'].unique())

    # evaluation grid
    all_vals = np.concatenate([df['y_emp'].values, df['y_theo'].values])
    x_min, x_max = all_vals.min(), all_vals.max()
    xs = np.linspace(x_min, x_max, 500)

    kl_vals = []
    for p in params:
        emp = df[df['param'] == p]['y_emp'].values
        theo = df[df['param'] == p]['y_theo'].values

        kde_emp = gaussian_kde(emp)
        kde_theo = gaussian_kde(theo)

        p_emp = kde_emp(xs) + 1e-12
        p_theo = kde_theo(xs) + 1e-12

        p_emp /= np.trapezoid(p_emp, xs)
        p_theo /= np.trapezoid(p_theo, xs)

        kl_vals.append(entropy(p_emp, p_theo))

    return params, kl_vals

def plot_kl(csv_file, param_name, output_basename):
    # Read KL table: index=seed, columns=param
    kl_df = pd.read_csv(csv_file, index_col='seed')
    params = [float(c) for c in kl_df.columns]

    # x = log4(param), y = mean log(KL), err = std(log(KL))
    x = np.log(params) / np.log(4)
    logkl = np.log(kl_df.values)
    y = logkl.mean(axis=0)
    err = logkl.std(axis=0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        line=dict(color='blue'),
        marker=dict(color='blue'),
        error_y=dict(
            type='data',
            array=err,
            visible=True,
            color='black'
        )
    ))

    fig.update_layout(
        font=font,
        width=width, height=height,
        margin=margin,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=grid,
        xaxis_title = f'log_4({param_name})',
        yaxis=grid,
        yaxis_title='log(KL)'
    )

    svg_file = f'{output_basename}.svg'
    pdf_file = f'{output_basename}.pdf'
    fig.write_image(svg_file)
    drawing = svg2rlg(svg_file)
    renderPDF.drawToFile(drawing, pdf_file)
    os.remove(svg_file)

if __name__ == '__main__':
    plot_kl('data/kl_vary_n.csv', 'n', 'figures/kl_vary_n')
    plot_kl('data/kl_vary_H.csv', 'H', 'figures/kl_vary_H')
