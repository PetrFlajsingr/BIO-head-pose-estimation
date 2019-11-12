import plotly.graph_objects as go
import numpy as np


def create_plot(correct_data, detected_data, path_to_save, name):
    assert (len(correct_data) == len(detected_data))
    x_axis = np.linspace(0, 1, len(correct_data))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=correct_data, mode='lines', name='Expected'))
    fig.add_trace(go.Scatter(x=x_axis, y=detected_data, mode='lines', name='Estimate'))

    fig.write_image("{}/{}.png".format(path_to_save, name))
