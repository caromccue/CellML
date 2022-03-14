import os

from plotly import offline as py
from plotly import tools
import plotly.graph_objs as go
import plotly.io as pio

def plot_cell_data(df, directory):
    '''Plot data from cell detahcment experiment'''

    fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Cell Detachment', 'Process control'))

    fig.append_trace(go.Scatter(
        x = df["RelTime"],
        y = df["Num cells"],
        name = 'Total',
        hoverinfo = 'text',
        text = df["Image Name"]
    ), 2, 1)

    fig.append_trace(go.Scatter(
        x = df["RelTime"],
        y = df["Num attached"],
        name = 'Attached'
    ), 2, 1)

    fig.append_trace(go.Scatter(
        x = df["RelTime"],
        y = df["Num detached"],
        name = 'Detached'
    ), 2, 1)

    fig.append_trace(go.Scatter(
        x = df["RelTime"],
        y = df["Num detached"] / df["Num attached"],
        name = 'Attached/Total'
    ), 1, 1)

    fig['layout']['xaxis'].update(title='Time (s)')
    fig['layout']['yaxis2'].update(title='Number of cells')
    fig['layout']['yaxis1'].update(title='Attached/Total', range=[0,1.05])

    fig['layout'].update(
        title='CCell detachement data for {}'.format(directory)
    )

    py.plot(fig)
    pio.write_image(fig, os.path.join(directory, 'Cell_detachment_plot.pdf'))
