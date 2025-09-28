# This script reads CSV outputs and generates plots
# Save via SVG->PDF, then remove SVG

import os
import numpy as np
import pandas as pd
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

def plot_kl(csv_file, param_name, output_basename):
    # Read KL table: index=seed, columns=param
    kl_df = pd.read_csv(csv_file, index_col='seed')
    params = [float(c) for c in kl_df.columns]

    x = np.log(params) / np.log(4)
    kl_mean = kl_df.values.mean(axis=0)
    kl_std = kl_df.values.std(axis=0)
    y = np.log(kl_mean)
    err = kl_std / kl_mean # Delta method for error propagation

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
    plot_kl('data/kl_lowrank.csv', 'n', 'figures/kl_lowrank')
