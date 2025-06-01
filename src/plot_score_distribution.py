# This script reads CSV outputs and generates plots
# Save via SVG->PDF, then remove SVG

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
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
legend = dict(
    x=0.53,
    y=0.98
)
xmin = -3
xmax = 3
xsize = 200
color_sqrtn = 'limegreen'
color_n = 'darkgreen'

def plot_score_dist(csv_file: str, output_basename: str):
    df = pd.read_csv(csv_file)
    vals_sqrtn = df.iloc[:, 0]
    vals_n = df.iloc[:, 1]

    # range for PDF curve
    xs = np.linspace(xmin, xmax, xsize)

    # Create figure
    fig = go.Figure()

    # finite width histograms
    fig.add_trace(
        go.Histogram(
            x=vals_sqrtn,
            histnorm='probability density',
            name='n=256, 1/√n-scaling',
            marker_color=color_sqrtn,
            opacity=0.5
        )
    )
    fig.add_trace(
        go.Histogram(
            x=vals_n,
            histnorm='probability density',
            name='n=256, 1/n-scaling',
            marker_color=color_n,
            opacity=0.5
        )
    )

    # infinite width PDF for 1/√n (standard normal)
    y_pdf = norm.pdf(xs)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=y_pdf,
            mode='lines',
            name='∞-width, 1/√n-scaling',
            line=dict(
                color=color_sqrtn,
                width=2
            )
    ))

    # infinite width line for 1/n (point mass at 0)
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[0, 7],
        mode='lines',
        name='∞-width, 1/n-scaling',
        line=dict(
            color=color_n,
            width=2
        )
    ))

    fig.update_layout(
        font=font,
        width=width,
        height=height,
        margin=margin,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=grid,
        xaxis_range=[xmin, xmax],
        yaxis=grid,
        legend=legend
    )

    # Save via SVG -> PDF
    svg_file = f'{output_basename}.svg'
    pdf_file = f'{output_basename}.pdf'
    fig.write_image(svg_file)
    drawing = svg2rlg(svg_file)
    renderPDF.drawToFile(drawing, pdf_file)
    os.remove(svg_file)

if __name__ == '__main__':
    plot_score_dist('data/data_score.csv', 'figures/score_dist')
