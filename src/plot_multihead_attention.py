# This script reads CSV outputs and generates plots
# Save via SVG->PDF, then remove SVG

import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import plotly.express as px
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
legend_n = dict(
    x=0.75,
    y=0.98
)
legend_H = dict(
    x=0.6,
    y=0.98
)
xmin_n = -4
xmax_n = 4
xmin_H = -20
xmax_H = 20
xsize = 200

def plot_vary_n(csv_file: str, output_basename: str):
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
                    dash='dot'
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
                dash='solid'
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

def plot_vary_H(csv_file: str, output_basename: str):
    df = pd.read_csv(csv_file)
    H_vals = sorted(df['param'].unique()) # list of H values

    # colors
    grad = ['mediumpurple', 'indigo']

    # range for PDF curve
    xs = np.linspace(xmin_H, xmax_H, xsize)

    # Create figure
    fig = go.Figure()

    for idx, H_val in enumerate(H_vals):
        sub = df[df['param'] == H_val]

        # finite width histogram
        fig.add_trace(
            go.Histogram(
                x=sub['y_emp'].values,
                histnorm='probability density',
                name=f'n=256, H={H_val}',
                marker_color=grad[idx],
                opacity=0.5,
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
                    dash='solid'
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
    plot_vary_n('data/data_vary_n_0.csv', 'figures/vary_n')
    plot_vary_H('data/data_vary_H_0.csv', 'figures/vary_H')
