import numpy as np

import plotly.graph_objects as go
import plotly.express as px

def plot_time_series(ts : np.ndarray, **kwargs):
    """
        Plot a univariate time series using Plotly.

        Params :
    
        - ts : one-dimensional array containing the time series values to plot.
        - **kwargs : additional keyword arguments controlling the plot appearance:

            - name : name of the time series (used in the legend)
            - color : line color (any Plotly-compatible color format)
            - title : title of the plot
            - x_axis : label for the x-axis
            - y_axis : label for the y-axis

        Note : 

        The x-axis is automatically generated as a range from 0 to `len(ts) - 1

    """

    name = kwargs["name"]
    color = kwargs["color"]
    title = kwargs["title"]
    x_axis = kwargs["x_axis"]
    y_axis = kwargs["y_axis"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[i for i in range (len(ts))],
        y=ts,
        mode='lines',
        line=dict(color=color, width=1),
        name=name
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_axis,
        yaxis_title=y_axis
    )
    fig.show()

def plot_forecast(input_ts : np.ndarray, ground_truth : np.ndarray, prediction : np.ndarray, **kwargs):
    """
        Plot a univariate time series forecast using Plotly.

        Params :
    
        - input_ts : one-dimensional array containing the input time series values to plot.
        - ground_truth : one-dimensional array containing the ground truth time series values to plot.
        - prediction : one-dimensional array containing the predicted time series values to plot.
        - **kwargs : additional keyword arguments controlling the plot appearance:

            - name : name of the time series (used in the legend)
            - color : line color (any Plotly-compatible color format)
            - title : title of the plot
            - x_axis : label for the x-axis
            - y_axis : label for the y-axis

        Note : 

        The x-axis is automatically generated as a range from 0 to `len(ts) - 1
        
    """

    name = kwargs["name"]
    color = kwargs["color"]
    title = kwargs["title"]
    x_axis = kwargs["x_axis"]
    y_axis = kwargs["y_axis"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[i for i in range (len(input_ts))],
        y=input_ts,
        mode='lines',
        line=dict(color="blue", width=1),
        name=f"input {name}"
    ))
    fig.add_trace(go.Scatter(
        x=[i for i in range (len(input_ts), len(input_ts) + len(ground_truth))],
        y=ground_truth,
        mode='lines+markers',
        line=dict(color="blue", width=1),
        name=f"ground truth {name}"
    ))
    fig.add_trace(go.Scatter(
        x=[i for i in range (len(input_ts), len(input_ts) + len(prediction))],
        y=prediction,
        mode='lines+markers',
        line=dict(color=color, width=1),
        name=f"prediction {name}"
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_axis,
        yaxis_title=y_axis
    )
    fig.show()

def plot_multiple_forecast(input_ts : np.ndarray, ground_truth : np.ndarray, predictions : dict[np.ndarray], **kwargs):
    """
        Plot multiple univariate time series forecast using Plotly.

        Params :
    
        - input_ts : one-dimensional array containing the input time series values to plot.
        - ground_truth : one-dimensional array containing the ground truth time series values to plot.
        - predictions : dictionary containing multiple one-dimensional array containing the predicted time series values to plot.
        - **kwargs : additional keyword arguments controlling the plot appearance:

            - name : name of the time series (used in the legend)
            - color : line color (any Plotly-compatible color format)
            - title : title of the plot
            - x_axis : label for the x-axis
            - y_axis : label for the y-axis

        Note : 

        The x-axis is automatically generated as a range from 0 to `len(ts) - 1
        
    """

    name = kwargs["name"]
    color = kwargs["color"]
    title = kwargs["title"]
    x_axis = kwargs["x_axis"]
    y_axis = kwargs["y_axis"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[i for i in range (len(input_ts))],
        y=input_ts,
        mode='lines',
        line=dict(color="blue", width=1),
        name=f"input {name}"
    ))
    fig.add_trace(go.Scatter(
        x=[i for i in range (len(input_ts), len(input_ts) + len(ground_truth))],
        y=ground_truth,
        mode='lines+markers',
        line=dict(color="blue", width=2),
        name=f"ground truth {name}"
    ))
    colors = px.colors.sample_colorscale("RdBu", [i/(len(predictions)-1) for i in range(len(predictions))])
    for i, prediction in enumerate(predictions):
        fig.add_trace(go.Scatter(
            x=[i for i in range (len(input_ts), len(input_ts) + len(prediction))],
            y=predictions[prediction],
            mode='lines',
            line=dict(color=colors[i], width=1),
            name=f"{prediction}"
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_axis,
        yaxis_title=y_axis
    )
    fig.show()

def plot_forecast_and_input():
    pass


