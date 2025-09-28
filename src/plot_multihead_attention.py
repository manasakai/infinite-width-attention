# This script reads CSV outputs and generates plots
# Save via SVG->PDF, then remove SVG

import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
import plotly.graph_objects as go
import plotly.express as px
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

# setting for experiments
s = 4
num_runs = 50000

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
legend_n = dict(
    x=0.75,
    y=0.98
)
legend_H = dict(
    x=0.65,
    y=0.98
)
xmin_n = -3
xmax_n = 3
xmin_H = -3
xmax_H = 3
xsize = 1000

# Softmax function
def softmax(x, axis=-1):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# Monte Carlo estimate of E[(Z^y^1)^2]=sE[Softmax_1(p)^2] where p ~ N(0,I_s)
def compute_Zy_squared_expectation(s, num_runs):
    p_samples = np.random.randn(num_runs, s)
    probs = softmax(p_samples, axis=1)
    probs_squared = probs[:, 0] ** 2
    mean_probs_squared = np.mean(probs_squared)
    variance = s * mean_probs_squared
    return variance

def plot_vary_n(csv_file: str, output_basename: str, variance: float):
    df = pd.read_csv(csv_file)
    n_vals = sorted(df['param'].unique()) # list of n values

    # gradient from blue -> purple -> red
    grad = px.colors.sample_colorscale([[0.0, '#0000FF'], [0.5, '#800080'], [1.0, '#FF0000']], len(n_vals))

    # range for PDF curve
    xs = np.linspace(xmin_n, xmax_n, xsize)

    # Create figure
    fig = go.Figure()

    # finite width KDEs
    for idx, n_val in enumerate(n_vals):
        arr = df[df['param'] == n_val]['y_emp'].values
        kde_emp = gaussian_kde(arr)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=kde_emp(xs),
                mode='lines',
                name=f'n={n_val}',
                line=dict(
                    color=grad[idx],
                    dash='dot',
                    width=1.5
                ),
                legendrank=1
            )
        )

    # infinite width KDE
    kde_theo = gaussian_kde(df['y_theo'].values)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=kde_theo(xs),
            mode='lines',
            name='∞-width',
            line=dict(
                color='black',
                dash='solid',
                width=1.5
            ),
            legendrank=2
        )
    )

    fig.update_layout(
        font=font,
        width=width,
        height=height,
        margin=margin,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=grid,
        yaxis=grid,
        legend=legend_n
    )

    svg_file = f'{output_basename}.svg'
    pdf_file = f'{output_basename}.pdf'
    fig.write_image(svg_file)
    drawing = svg2rlg(svg_file)
    renderPDF.drawToFile(drawing, pdf_file)
    os.remove(svg_file)

def plot_vary_H(csv_file: str, output_basename: str, variance: float):
    df = pd.read_csv(csv_file)
    H_vals = sorted(df['param'].unique()) # list of H values

    # colors
    grad = ['blue', 'red']

    # range for PDF curve
    xs = np.linspace(xmin_H, xmax_H, xsize)

    # Create figure
    fig = go.Figure()

    for idx, H_val in enumerate(H_vals):
        sub = df[df['param'] == H_val]

        # finite width KDEs
        arr = df[df['param'] == H_val]['y_emp'].values
        kde_emp = gaussian_kde(arr)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=kde_emp(xs),
                mode='lines',
                name=f'n=256, H={H_val}',
                line=dict(
                    color=grad[idx],
                    dash='dot',
                    width=1.5
                ),
                legendrank=1
            )
        )

        # infinite width KDE
        kde_theo = gaussian_kde(sub['y_theo'].values)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=kde_theo(xs),
                mode='lines',
                name=f'∞-width, H={H_val}',
                line=dict(
                    color=grad[idx],
                    dash='solid',
                    width=1.5
                ),
                legendrank=2
            )
        )

    # infinite width and head KDE
    pdf_inf_head= norm.pdf(xs, loc=0, scale=np.sqrt(variance))
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=pdf_inf_head,
            mode='lines',
            name='∞-width ,∞-head',
            line=dict(
                color='black',
                dash='solid',
                width=1.5
            ),
            legendrank=3
        )
    )

    fig.update_layout(
        font=font,
        width=width,
        height=height,
        margin=margin,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=grid,
        xaxis_range=[xmin_H, xmax_H],
        yaxis=grid,
        legend=legend_H
    )

    svg_file = f'{output_basename}.svg'
    pdf_file = f'{output_basename}.pdf'
    fig.write_image(svg_file)
    drawing = svg2rlg(svg_file)
    renderPDF.drawToFile(drawing, pdf_file)
    os.remove(svg_file)

if __name__ == '__main__':
    np.random.seed(0)
    variance = compute_Zy_squared_expectation(s, num_runs)
    plot_vary_n('data/data_vary_n_0.csv', 'figures/vary_n', variance)
    plot_vary_H('data/data_vary_H_0.csv', 'figures/vary_H', variance)
