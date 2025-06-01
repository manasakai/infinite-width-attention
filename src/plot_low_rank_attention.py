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
width = 700
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
xmin = -10
xmax = 10
xsize = 200

def plot_lowrank(csv_file: str, output_basename: str):
    df = pd.read_csv(csv_file)
    H_vals = sorted(df['param'].unique()) # list of H values

    # gradient from blue -> purple -> red
    grad = px.colors.sample_colorscale([[0.0, '#0000FF'], [0.5, '#800080'], [1.0, '#FF0000']], len(H_vals))

    # range for PDF curve
    xs = np.linspace(xmin, xmax, xsize)

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
                name=f'n={H_val*64}, H={H_val}',
                line=dict(
                    color=grad[idx],
                    dash='dot'
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
                    dash='solid')
            )
        )

    fig.update_layout(
        font=font,
        width=width,
        height=height,
        margin=margin,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=grid,
        yaxis=grid
    )

    svg_file = f'{output_basename}.svg'
    pdf_file = f'{output_basename}.pdf'
    fig.write_image(svg_file)
    drawing = svg2rlg(svg_file)
    renderPDF.drawToFile(drawing, pdf_file)
    os.remove(svg_file)

if __name__ == '__main__':
    plot_lowrank('data/data_lowrank.csv', 'figures/lowrank')
