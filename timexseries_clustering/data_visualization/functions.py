import logging

import pandas
from pandas import Grouper, DataFrame
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from typing import List

import dash_core_components as dcc
import dash_html_components as html
from plotly.subplots import make_subplots
import networkx as nx
import dash_bootstrap_components as dbc

from colorhash import ColorHash
from statsmodels.tsa.seasonal import seasonal_decompose

from timexseries_clustering.data_clustering import ValidationPerformance
from timexseries_clustering.data_clustering.models.predictor import SingleResult
import calendar

from timexseries_clustering.timeseries_container import TimeSeriesContainer

log = logging.getLogger(__name__)

# Default method to get a translated text.
_ = lambda x: x


def create_timeseries_dash_children(timeseries_container: TimeSeriesContainer, param_config: dict):
    """
    Creates the Dash children for a specific time-series. They include a line plot, histogram, box plot and
    autocorrelation plot. For each model on the time-series the prediction plot and performance plot are also added.

    Cross-correlation plots and graphs are shown, if the the `timeseries_container` have it.

    If the `timeseries_container` also have the `historical_prediction`, it is shown in a plot.
    
    Parameters
    ----------
    timeseries_container: TimeSeriesContainer
        Time-series for which the various plots and graphs will be returned.
    
    param_config : dict
        TIMEX CLUSTERING configuration parameters dictionary, used for `visualization_parameters` which contains settings to
        customize some plots and graphs.

    Returns
    -------
    list
        List of Dash children.

    Examples
    --------
    Given a `timexseries.timeseries_container.TimeSeriesContainer` object, obtained for example through
    `timexseries.data_prediction.pipeline.create_timeseries_containers`, create all the Dash object which could be shown in a
    Dash app:
    >>> param_config = {
    ...  "input_parameters": {},
    ...  "model_parameters": {
    ...      "models": "fbprophet",
    ...      "possible_transformations": "none,log_modified",
    ...      "main_accuracy_estimator": "mae",
    ...      "delta_training_percentage": 20,
    ...      "test_values": 5,
    ...      "prediction_lags": 7,
    ...  },
    ...  "historical_prediction_parameters": {
    ...      "initial_index": "2000-01-25",
    ...      "save_path": "example.pkl"
    ...  },
    ...  "visualization_parameters": {}
    ...}
    >>>  plots = create_timeseries_dash_children(timeseries_container, param_config)
    """
    children = []

    visualization_parameters = param_config["visualization_parameters"]
    timeseries_data = timeseries_container.timeseries_data
    clustering_approach = timeseries_container.approach
    
    #clustering_models = timeseries_container.models['k_means']
    
    # Data visualization with plots
    children.extend([
        html.H2(children = clustering_approach + (' approach analysis'), id=clustering_approach),
        html.H3("Data visualization"),
        timeseries_plot(timeseries_data),
        #histogram_plot(timeseries_data),
        #box_plot(timeseries_data, visualization_parameters),
        #components_plot(timeseries_data),
    ])

    
    # Plot the prediction results, if requested.
    if timeseries_container.models is not None:
        param_configuration = param_config["model_parameters"]
        main_accuracy_estimator = param_configuration["main_accuracy_estimator"]
        models = timeseries_container.models

        children.append(
            html.H3("Clustering results"),
        )

        all_performances = []
        
        for model_name in models:
            model = models[model_name]
            model_characteristic = {}
            for metric_key in model:
                metric = model[metric_key] #ModelResult object
                model_performances = metric.results #[SingleResult]
                model_characteristic = metric.characteristics.copy()
                all_performances.append(model_performances) #[[SingleResult]]
            best_performances = [x[0] for x in all_performances] #[SingleResult]
            model_characteristic['distance_metric'] = param_configuration['distance_metric']
            model_characteristic['n_clusters'] = param_configuration['n_clusters']
            
            if main_accuracy_estimator=="silhouette":
                best_performances.sort(key=lambda x: getattr(x.performances, main_accuracy_estimator), reverse=True)
            else:
                best_performances.sort(key=lambda x: getattr(x.performances, main_accuracy_estimator))

            best_model = best_performances[0].characteristics['model']
            best_metric = best_performances[0].characteristics['distance_metric']
            if best_model=='K Means': best_model='k_means'
            if best_metric=='Euclidean': best_metric='euclidean'
            elif best_metric=='DTW': best_metric='dtw'
            elif best_metric=='SoftDTW': best_metric='softdtw'

            children.extend([
                html.H4(f"{model_name}"),
                characteristics_list(model_characteristic, best_performances[0]),
                cluster_plot(timeseries_data, model),
                performance_plot(timeseries_data, param_config, all_performances),
                validation_performance_info(),
                cluster_distribution_plot(timeseries_container.models[best_model][best_metric].best_clustering),
            ])

            # EXTRA
            # Warning: this will plot every model result, with every distance metric used!
            # children.extend(plot_every_prediction(ingested_data, model_results, main_accuracy_estimator, test_values))
    
    # Plot cross-correlation plot and graphs, if requested.
    if timeseries_container.xcorr is not None:
        graph_corr_threshold = visualization_parameters[
            "xcorr_graph_threshold"] if "xcorr_graph_threshold" in visualization_parameters else None

        children.extend([
            html.H3("Cross-correlation"),
            html.Div("Negative lags (left part) show the correlation between this scenario and the future of the "
                       "others."),
            html.Div("Meanwhile, positive lags (right part) shows the correlation between this scenario "
                       "and the past of the others."),
            cross_correlation_plot(timeseries_container.xcorr),
            html.Div("The peaks found using each cross-correlation modality are shown in the graphs:"),
            cross_correlation_graph(clustering_approach, timeseries_container.xcorr, graph_corr_threshold)
        ])

    return children


def create_dash_children(timeseries_containers: List[TimeSeriesContainer], param_config: dict):
    """
    Create Dash children, in order, for a list of `timexseries.timeseries_container.TimeSeriesContainer`.

    Parameters
    ----------
    timeseries_containers : [TimeSeriesContainer]
        Time-series for which all the plots and graphs will be created.

    param_config : dict
        TIMEX configuration parameters dictionary.

    Returns
    -------
    list
        List of Dash children.

    """
    children = []
    for s in timeseries_containers:
        children.extend(create_timeseries_dash_children(s, param_config))

    return children


def line_plot(df: DataFrame) -> dcc.Graph:
    """
    Create and return the line plot for a dataframe.
    Parameters
    ----------
    df : DataFrame
        Dataframe to plot.
    Returns
    -------
    g : dcc.Graph
        Dash object containing the line plot.
    Examples
    --------
    Get the `figure` attribute if you want to display this in a Jupyter notebook.
    >>> line_plot = line_plot(timeseries_container.timeseries_data).figure
    >>> line_plot.show()
    """
    fig = go.Figure(data=go.Scatter(x=df.index, y=df.iloc[:, 0], mode='lines+markers'))
    fig.update_layout(title='Line plot', xaxis_title=df.index.name, yaxis_title=df.columns[0])

    g = dcc.Graph(
        figure=fig
    )
    return g


def line_plot_multiIndex(df: DataFrame) -> dcc.Graph:
    """
    Returns a line plot for a dataframe with a MultiIndex.
    It is assumed that the first-level index is the real index, and that data should be grouped using the
    second-level one.

    Parameters
    ----------
    df : DataFrame
        Dataframe to plot. It is a multiIndex dataframe.

    Returns
    -------
    g : dcc.Graph
    """
    fig = go.Figure()
    for region in df.index.get_level_values(1).unique():
        fig.add_trace(go.Scatter(x=df.index.get_level_values(0).unique(), y=df.loc[
            (df.index.get_level_values(1) == region), df.columns[0]], name=region))

    fig.update_layout(title=_('Line plot'), xaxis_title=df.index.get_level_values(0).name,
                      yaxis_title=df.columns[0])
    g = dcc.Graph(
        figure=fig
    )
    return g


def histogram_plot(df: DataFrame) -> dcc.Graph:
    """
    Create and return the histogram plot for a dataframe.

    Parameters
    ----------
    df : DataFrame
        Dataframe to plot.

    Returns
    -------
    g : dcc.Graph

    Examples
    --------
    Get the `figure` attribute if you want to display this in a Jupyter notebook.
    >>> hist_plot = hist_plot(timeseries_container.timeseries_data).figure
    >>> hist_plot.show()
    """

    max_value = abs(df.iloc[:, 0].max() - df.iloc[:, 0].min())
    step = max_value / 400

    fig = go.Figure(data=[go.Histogram(x=df.iloc[:, 0])])
    fig.layout.sliders = [dict(
        steps=[dict(method='restyle', args=['xbins.size', i]) for i in np.arange(step, max_value / 2, step)],
        font=dict(color="rgba(0,0,0,0)"),
        tickcolor="rgba(0,0,0,0)"
    )]

    fig.update_layout(title=_('Histogram'), xaxis_title_text=df.columns[0], yaxis_title_text=_('Count'))
    g = dcc.Graph(
        figure=fig
    )
    return g


def components_plot(ingested_data: DataFrame) -> html.Div:
    """
    Create and return the plots of all the components of the time series: level, trend, residual.
    It uses both an additive and multiplicative model, with a subplot.

    Parameters
    ----------
    ingested_data : DataFrame
        Original time series values.

    Returns
    -------
    g : dcc.Graph

    Examples
    --------
    Get the `figure` attribute if you want to display this in a Jupyter notebook.
    >>> comp_plot = components_plot(timeseries_container.timeseries_data)[0].figure
    >>> comp_plot.show()
    """
    modes = ["additive", "multiplicative"]

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[_("Trend"), _("Seasonality"), _("Residual")], shared_xaxes=True, vertical_spacing=0.05,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
    )

    interpolated = ingested_data.interpolate()
    interpolated = interpolated.fillna(0)

    for mode in modes:
        try:
            result = seasonal_decompose(interpolated, model=mode)
            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid

            secondary_y = False if mode == "additive" else True

            fig.add_trace(go.Scatter(x=trend.index, y=trend,
                                     mode='lines+markers',
                                     name=_(mode.capitalize()), legendgroup=_(mode.capitalize()),
                                     line=dict(color=ColorHash(mode).hex)),
                          row=1, col=1, secondary_y=secondary_y)
            fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal,
                                     mode='lines+markers', showlegend=False,
                                     name=_(mode.capitalize()), legendgroup=_(mode.capitalize()),
                                     line=dict(color=ColorHash(mode).hex)),
                          row=2, col=1, secondary_y=secondary_y)
            fig.add_trace(go.Scatter(x=residual.index, y=residual,
                                     mode='lines+markers', showlegend=False,
                                     name=_(mode.capitalize()), legendgroup=_(mode.capitalize()),
                                     line=dict(color=ColorHash(mode).hex)),
                          row=3, col=1, secondary_y=secondary_y)
        except ValueError:
            log.warning(f"Multiplicative decomposition not available for {ingested_data.columns[0]}")

    fig.update_layout(title=_("Components decomposition"), height=1000, legend_title_text=_('Decomposition model'))
    fig.update_yaxes(title_text="<b>" + _('Additive') + "</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>" + _('Multiplicative') + "</b>", secondary_y=True)

    g = dcc.Graph(
        figure=fig
    )

    warning = html.H5(_("Multiplicative model is not available for series which contain zero or negative values."))

    return html.Div([g, warning])


def autocorrelation_plot(df: DataFrame) -> dcc.Graph:
    """
    Create and return the autocorrelation plot for a dataframe.

    Parameters
    ----------
    df : DataFrame
        Dataframe to use in the autocorrelation plot.

    Returns
    -------
    g : dcc.Graph

    Examples
    --------
    Get the `figure` attribute if you want to display this in a Jupyter notebook.
    >>> auto_plot = autocorrelation_plot(timeseries_container.timeseries_data).figure
    >>> auto_plot.show()
    """
    # Code from https://github.com/pandas-dev/pandas/blob/v1.1.4/pandas/plotting/_matplotlib/misc.py
    n = len(df)
    data = np.asarray(df)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0

    x = np.arange(n) + 1
    y = [r(loc) for loc in x]

    z95 = 1.959963984540054
    z99 = 2.5758293035489004

    c1 = z99 / np.sqrt(n)
    c2 = z95 / np.sqrt(n)
    c3 = -z95 / np.sqrt(n)
    c4 = -z99 / np.sqrt(n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=_('autocorrelation')))
    fig.add_trace(go.Scatter(x=x, y=np.full(n, c1), line=dict(color='gray', width=1), name='z99'))
    fig.add_trace(go.Scatter(x=x, y=np.full(n, c2), line=dict(color='gray', width=1), name='z95'))
    fig.add_trace(go.Scatter(x=x, y=np.full(n, c3), line=dict(color='gray', width=1), name='-z95'))
    fig.add_trace(go.Scatter(x=x, y=np.full(n, c4), line=dict(color='gray', width=1), name='-z99'))
    fig.update_layout(title=_('Autocorrelation plot'), xaxis_title=_('Lags'), yaxis_title=_('Autocorrelation'))
    fig.update_yaxes(tick0=-1.0, dtick=0.25)
    fig.update_yaxes(range=[-1.2, 1.2])
    g = dcc.Graph(
        figure=fig
    )
    return g


def cross_correlation_plot(xcorr: dict):
    """
    Create and return the cross-correlation plot for all the columns in the dataframe.
    The time-series column is used as target; the correlation is shown in a subplot for every modality used to compute
    the x-correlation.

    Parameters
    ----------
    xcorr : dict
        Cross-correlation values.

    Returns
    -------
    g : dcc.Graph

    Examples
    --------
    Get the `figure` attribute if you want to display this in a Jupyter notebook.
    >>> xcorr_plot = cross_correlation_plot(timeseries_container.xcorr).figure
    >>> xcorr_plot.show()
    """
    subplots = len(xcorr)
    combs = [(1, 1), (1, 2), (2, 1), (2, 2)]

    rows = 1 if subplots < 3 else 2
    cols = 1 if subplots < 2 else 2

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=([*xcorr.keys()]))

    i = 0
    for mode in xcorr:
        for col in xcorr[mode].columns:
            fig.add_trace(go.Scatter(x=xcorr[mode].index, y=xcorr[mode][col],
                                     mode='lines',
                                     name=col, legendgroup=col, line=dict(color=ColorHash(col).hex),
                                     showlegend=True if i == 0 else False),
                          row=combs[i][0], col=combs[i][1])
        i += 1

    # Formula from https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/time-series/how-to/cross-correlation/interpret-the-results/all-statistics-and-graphs/
    # significance_level = DataFrame(columns=['Value'], dtype=np.float64)
    # for i in range(-lags, lags):
    #     significance_level.loc[i] = 2 / np.sqrt(lags - abs(i))

    # fig.add_trace(
    #     go.Scatter(x=significance_level.index, y=significance_level['Value'], line=dict(color='gray', width=1), name='z95'))
    # fig.add_trace(
    #     go.Scatter(x=significance_level.index, y=-significance_level['Value'], line=dict(color='gray', width=1), name='-z95'))

    fig.update_layout(title=_("Cross-correlation using different algorithms"))
    fig.update_xaxes(title_text=_("Lags"))
    fig.update_yaxes(tick0=-1.0, dtick=0.25, range=[-1.2, 1.2], title_text=_("Correlation"))

    g = dcc.Graph(
        figure=fig
    )
    return g


def cross_correlation_graph(name: str, xcorr: dict, threshold: float = 0) -> dcc.Graph:
    """
    Create and return the cross-correlation graphs for all the columns in the dataframe.
    A graph is created for each mode used to compute the x-correlation.

    The nodes are all the time-series which can be found in `xcorr`; an arc is drawn from `target` node to another node
    if the cross-correlation with that time-series, at any lag, is above the `threshold`. The arc contains also the
    information on the lag.

    Parameters
    ----------
    name : str
        Name of the target.

    xcorr : dict
        Cross-correlation dataframe.

    threshold : int
        Minimum value of correlation for which a edge should be drawn. Default 0.

    Returns
    -------
    g : dcc.Graph

    Examples
    --------
    This is thought to be shown in a Dash app, so it could be difficult to show in Jupyter.
    >>> xcorr_graph = cross_correlation_graph("a", timeseries_container.xcorr, 0.7)
    """
    figures = []

    i = 0
    for mode in xcorr:
        G = nx.DiGraph()
        G.add_nodes_from(xcorr[mode].columns)
        G.add_node(name)

        for col in xcorr[mode].columns:
            index_of_max = xcorr[mode][col].abs().idxmax()
            corr = xcorr[mode].loc[index_of_max, col]
            if abs(corr) > threshold:
                G.add_edge(name, col, corr=corr, lag=index_of_max)

        pos = nx.layout.spring_layout(G)

        # Create Edges
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(color='black'),
            mode='lines',
            hoverinfo='skip',
        )

        for edge in G.edges():
            start = edge[0]
            end = edge[1]
            x0, y0 = pos.get(start)
            x1, y1 = pos.get(end)
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        # Create Nodes
        node_trace = go.Scatter(
            x=[],
            y=[],
            mode='markers+text',
            text=[node for node in G.nodes],
            textposition="bottom center",
            hoverinfo='skip',
            marker=dict(
                color='green',
                size=15)
        )

        for node in G.nodes():
            x, y = pos.get(node)
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])

        # Annotations to support arrows
        edges_positions = [e for e in G.edges]
        annotateArrows = [dict(showarrow=True, arrowsize=1.0, arrowwidth=2, arrowhead=2, standoff=2, startstandoff=2,
                               ax=pos[arrow[0]][0], ay=pos[arrow[0]][1], axref='x', ayref='y',
                               x=pos[arrow[1]][0], y=pos[arrow[1]][1], xref='x', yref='y',
                               text="bla") for arrow in edges_positions]

        graph = go.Figure(data=[node_trace, edge_trace],
                          layout=go.Layout(title=str(mode),
                                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                           showlegend=False,
                                           annotations=annotateArrows,
                                           height=400, margin=dict(l=10, r=10, t=50, b=30)))

        # Add annotations on edges
        for e in G.edges:
            lag = str(G.edges[e]['lag'])
            corr = str(round(G.edges[e]['corr'], 3))

            end = e[1]
            x, y = pos.get(end)

            graph.add_annotation(x=x, y=y, text=_("Lag: ") + lag + ", corr: " + corr, yshift=20, showarrow=False,
                                 bgcolor='white')

        figures.append(graph)
        i += 1

    n_graphs = len(figures)
    if n_graphs == 1:
        g = dcc.Graph(figure=figures[0])
    elif n_graphs == 2:
        g = html.Div(dbc.Row([
            dbc.Col(dcc.Graph(figure=figures[0])),
            dbc.Col(dcc.Graph(figure=figures[1]))
        ]))
    elif n_graphs == 3:
        g = html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=figures[0])),
                dbc.Col(dcc.Graph(figure=figures[1]))
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=figures[2]))
            ])
        ])
    elif n_graphs == 4:
        g = html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=figures[0])),
                dbc.Col(dcc.Graph(figure=figures[1])),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=figures[2])),
                dbc.Col(dcc.Graph(figure=figures[3]))
            ])
        ])
    else:
        g = html.Div()

    return g


def box_plot(df: DataFrame, visualization_parameters: dict) -> dcc.Graph:
    """
    Create and return the box-and-whisker plot for a dataframe.

    Parameters
    ----------
    df : DataFrame
        Dataframe to use in the box plot.

    visualization_parameters : dict
        Options set by the user. In particular, `box_plot_frequency` is used to determine the number of boxes. The
        default value is `1W`.

    Returns
    -------
    g : dcc.Graph

    Examples
    --------
    Get the `figure` attribute if you want to display this in a Jupyter notebook.
    >>> box_plot = box_plot(timeseries_container.timeseries_data, param_config["visualization_parameters"]).figure
    >>> box_plot.show()
    """
    temp = df.iloc[:, 0]

    aggregations = ["1W", "2W", "1M", "3M", "4M", "6M", "1Y"]

    try:
        initial_freq = visualization_parameters['box_plot_frequency']
    except KeyError:
        initial_freq = '1W'

    traces = []

    for freq in aggregations:
        is_visible = True if initial_freq == freq else 'legendonly'

        # Small trick needed to show `name` in legend.
        traces.append(go.Scatter(x=[None], y=[None], legendgroup=freq, name=freq,
                                 visible=is_visible))

    for freq in aggregations:
        groups = temp.groupby(Grouper(freq=freq))
        is_visible = True if initial_freq == freq else 'legendonly'

        for group in groups:
            traces.append(go.Box(
                name=str(group[0].normalize()),
                y=group[1], legendgroup=freq, showlegend=False, visible=is_visible, xperiodalignment='end'
            ))

    fig = go.Figure(traces)
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            direction='right',
            yanchor='bottom',
            xanchor='left',
            y=1,
            active=aggregations.index(_(initial_freq)),
            buttons=list(
                [dict(label=f, method='update', args=[{'visible': [tr.legendgroup == f for tr in traces]}])
                 for f in aggregations]))])

    fig.update_layout(title=_('Box plot'), xaxis_title=df.index.name, yaxis_title=_('Count'), showlegend=False)
    fig.update_xaxes(type='category')
    fig.add_annotation(text=_("The index refer to the end of the period."),
                       xref="paper", yref="paper",
                       x=1.0, y=1.0, yanchor='bottom', showarrow=False)

    g = dcc.Graph(
        figure=fig
    )
    return g


def box_plot_aggregate(df: DataFrame, visualization_parameters: dict) -> dcc.Graph:
    """
    Create and return the aggregate box plot for a dataframe, i.e. a box plot which shows, for each day of the week/for
    each month of the year the distribution of the values. Now also with aggregation over minute, hour, week, month,
    year.

    Parameters
    ----------
    df : DataFrame
        Dataframe to use in the box plot.

    visualization_parameters : dict
        Options set by the user. In particular, `aggregate_box_plot_frequency` is used. Default `weekday`. It controls
        which frequency is activated at launch.

    Returns
    -------
    g : dcc.Graph

    Examples
    --------
    Get the `figure` attribute if you want to display this in a Jupyter notebook.
    >>> abox_plot = box_plot_aggregate(timeseries_container.timeseries_data,
    ...                                param_config["visualization_parameters"]).figure
    >>> abox_plot.show()
    """
    aggregations = ["minute", "hour", "weekday", "day", "month", "year"]
    translated_aggregations = [_("minute"), _("hour"), _("weekday"), _("day"), _("month"), _("year")]

    temp = df.iloc[:, 0]

    try:
        initial_freq = visualization_parameters['aggregate_box_plot_frequency']
    except KeyError:
        initial_freq = 'weekday'

    traces = []

    for freq, translated_freq in zip(aggregations, translated_aggregations):
        is_visible = True if initial_freq == freq else False

        # Small trick needed to show `name` in legend.
        traces.append(go.Scatter(x=[None], y=[None], legendgroup=translated_freq, name=translated_freq,
                                 visible=is_visible))

    for freq, translated_freq in zip(aggregations, translated_aggregations):
        groups = temp.groupby(getattr(temp.index, freq))
        is_visible = True if initial_freq == freq else False

        if freq == "weekday":
            for group in groups:
                traces.append(go.Box(
                    name=calendar.day_name[group[0]],
                    y=group[1], legendgroup=translated_freq, showlegend=False, visible=is_visible
                ))
        elif freq == "month":
            for group in groups:
                traces.append(go.Box(
                    name=calendar.month_name[group[0]],
                    y=group[1], legendgroup=translated_freq, showlegend=False, visible=is_visible
                ))
        else:
            for group in groups:
                traces.append(go.Box(
                    name=group[0],
                    y=group[1], legendgroup=translated_freq, showlegend=False, visible=is_visible
                ))

    fig = go.Figure(data=traces)

    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            direction='right',
            yanchor='bottom',
            xanchor='left',
            y=1,
            active=translated_aggregations.index(_(initial_freq)),
            buttons=list(
                [dict(label=f, method='update', args=[{'visible': [tr.legendgroup == f for tr in traces]}])
                 for f in translated_aggregations]))])

    fig.update_layout(title=_('Aggregate box plot'), yaxis_title=_('Count'), showlegend=False)
    fig.update_xaxes(type='category')

    g = dcc.Graph(
        figure=fig
    )
    return g


def timeseries_plot(df: DataFrame) -> dcc.Graph:
    """
    Create and return a plot which contains the time series of a dataframe.
    The plot is built using a dataframe: `ingested_data`.

    `ingested_data` includes the raw data ingested by the app, while `cluster_data` contains the cluster indexes, cluster characteristics
    and cluster centers made by a model.

    Parameters
    ----------
    df : DataFrame
        Raw values ingested by the app.

    Returns
    -------
    g : dcc.Graph

    See Also
    --------
    Check `create_timeseries_dash_children` to check the use.
    """

    fig = go.Figure()

    for i in df.columns[0:]:
        fig.add_trace(go.Scatter(x=df.index, y=df[i],
                            mode='lines',
                            name=i))
        
    fig.update_layout(title=("Time-Series ingested"), xaxis_title=df.index.name)

    g = dcc.Graph(
        figure=fig )
    return g


def cluster_plot(df: DataFrame, cluster_data: dict) -> dcc.Graph:
    """
    Create and return a plot which contains the clustering for a dataframe.
    The plot is built using a dataframe: `ingested_data` and dictionary: `cluster_data`.

    `ingested_data` includes the raw data ingested by the app, while `cluster_data` contains the cluster indexes, cluster characteristics
    and cluster centers made by a model.

    Note that `cluster_data` its is a dictionary with distance metric as keys and ModelResult objects as values.

    The time-series are plotted in black and the cluster centers are plotted in red.

    Parameters
    ----------
    df : DataFrame
        Raw values ingested by the app.

    cluster_data : DataFrame
        Clustering created by a model.

    Returns
    -------
    g : dcc.Graph

    See Also
    --------
    Check `create_timeseries_dash_children` to check the use.
    """

    dframe = df.copy()
    df_array = dframe.to_numpy()
    df_array = df_array.transpose()
    column_names = df.columns.values

    num_model = len(cluster_data)
    num_dist_metrics = len(cluster_data)
    subplotmult = 0
    first_key = list(cluster_data)[0]
    num_clusters = cluster_data[first_key].characteristics['n_clusters']
    model = cluster_data[first_key].characteristics['model']

    titles = []
    for key, value in cluster_data.items() :
        for i in range(1,len(cluster_data)+1):
            titles.append('Metric: '+str(key)+', Cluster '+str(i))
    
    fig = make_subplots(rows = num_dist_metrics, cols = num_clusters, subplot_titles=(titles))

    subplotmult = 1
    for key, value in cluster_data.items() :
        for yi in range(num_clusters):
            i = 0
            cluster_names = column_names[value.best_clustering == yi]
            for xx in df_array[value.best_clustering == yi]:
                fig.add_trace(go.Scatter(x=df.index, y=xx,
                                    line=dict(color='grey',width= 0.6),
                                    mode='lines',name=cluster_names[i]),
                                    row=subplotmult, col=yi+1)
                i = i+1
            fig.add_trace(go.Scatter(x=df.index, y=value.cluster_centers[yi],
                                line=dict(color='red'),
                                mode='lines',
                                name= (str(key)+', cluster center '+ str(yi+1))),
                                row=subplotmult, col=yi+1)
        subplotmult = subplotmult + 1

    fig.update_layout(title="Best clustering for the dataset", height=750)
    fig.update_yaxes(matches='y')
    
    g = dcc.Graph(
        figure=fig)
    return g


def cluster_plot_matplotlib(df: DataFrame, cluster_data: dict):
    """
    Create and return a plot using cluster_plot_matplotlib which contains the clustering for a dataframe.
    The plot is built using a dataframe: `ingested_data` and dictionary: `cluster_data`.

    `ingested_data` includes the raw data ingested by the app, while `cluster_data` contains the cluster indexes, cluster characteristics
    and cluster centers made by a model.

    Note that `cluster_data` its is a dictionary with distance metric as keys and ModelResult objects as values.

    The time-series are plotted in black and the cluster centers are plotted in red.

    Parameters
    ----------
    df : DataFrame
        Raw values ingested by the app.

    cluster_data : DataFrame
        Clustering created by a model.

    """
    plt.figure()
    plt.figure(figsize=(13, 8))

    X_train = df.to_numpy()
    X_train = X_train.transpose()
    cluster_data
    num_dist_metrics = len(cluster_data)
    subplotmult = 0
    first_key = list(cluster_data)[0]
    num_clusters = cluster_data[first_key].characteristics['n_clusters']
    subplotmult = 0
    sz = len(df)

    for key, value in cluster_data.items() :
        for yi in range(num_clusters):
            plt.subplot(num_dist_metrics, num_clusters, yi + 1 + num_clusters*subplotmult)
            for xx in X_train[value.best_clustering == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(value.cluster_centers[yi], "r-")
            plt.xlim(0, sz)
            plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                    transform=plt.gca().transAxes)
            if yi == 1:
                plt.title('Model: '+str(value.characteristics['model'])+', Distance metric: '+str(value.characteristics['distance_metric']))
        subplotmult = subplotmult + 1

    plt.tight_layout()
    plt.show()


def performance_plot(df: DataFrame, param_config : dict, all_performances: List) -> dcc.Graph:
    """
    Create and return the performance plot of the model; for every error kind (i.e. Silhouette, Davies Bouldin, etc) plot the values it
    assumes using differentclustering model parameters.
    Plot the best clustering data in the end. **

    Parameters
    ----------
    df : DataFrame
        Raw values ingested by the app.

    param_config : dict
        TIMEX configuration parameters dictionary.

    all_performances : List
        List of [SingleResults] objects. Every object is related to a different model parameter configuration,
        hence it shows the performance using that configuration.

    Returns
    -------
    g : dcc.Graph

    See Also
    --------
    Check `create_timeseries_dash_children` to check the use.
    """
    
    distance_metrics = [*param_config["model_parameters"]["distance_metric"].split(",")]
    n_cluster_test_values = param_config['model_parameters']['n_clusters']
    transformations = [*param_config["model_parameters"]["possible_transformations"].split(",")]
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    import numpy
    n_cls = len(n_cluster_test_values)
    
    if all_performances[0][0].characteristics['transformation']=='none': #Observation based approach
        nparray_performances = numpy.zeros((n_cls,9))
        for metric in all_performances:
            nc=0
            for n_cluster in metric:
                if n_cluster.characteristics['distance_metric']=='Euclidean':
                    nparray_performances[nc][0] = n_cluster.performances.silhouette
                    nparray_performances[nc][1] = n_cluster.performances.davies_bouldin
                    nparray_performances[nc][2] = n_cluster.performances.calinski_harabasz
                elif n_cluster.characteristics['distance_metric']=='DTW':
                    nparray_performances[nc][3] = n_cluster.performances.silhouette
                    nparray_performances[nc][4] = n_cluster.performances.davies_bouldin
                    nparray_performances[nc][5] = n_cluster.performances.calinski_harabasz        
                elif n_cluster.characteristics['distance_metric']=='SoftDTW':
                    nparray_performances[nc][6] = n_cluster.performances.silhouette
                    nparray_performances[nc][7] = n_cluster.performances.davies_bouldin
                    nparray_performances[nc][8] = n_cluster.performances.calinski_harabasz
                nc=nc+1
        df_performances = pandas.DataFrame(nparray_performances, columns=['silhouette_ED', 'davies_bouldin_ED', 'calinski_harabasz_ED',
                                                                    'silhouette_DTW', 'davies_bouldin_DTW', 'calinski_harabasz_DTW',
                                                                    'silhouette_softDTW', 'davies_bouldin_softDTW', 'calinski_harabasz_softDTW'])
        # Euclidian metric plots
        if 'euclidean' in distance_metrics:
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances['silhouette_ED'],
                                        line=dict(color='magenta'),
                                        mode="lines+markers",
                                        name='Silhouette ED'), row=1, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances['davies_bouldin_ED'],
                                        line=dict(color='yellow'),
                                        mode="lines+markers",
                                        name='Davies Bouldin ED'), row=2, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances['calinski_harabasz_ED'],
                                        line=dict(color='DeepSkyBlue'),
                                        mode="lines+markers",
                                        name='Calinski Harabasz ED'), row=3, col=1)
        # DTW metric plots
        if 'dtw' in distance_metrics:
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances['silhouette_DTW'],
                                        line=dict(color='goldenrod'),
                                        mode="lines+markers",
                                        name='Silhouette DTW'), row=1, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances['davies_bouldin_DTW'],
                                        line=dict(color='limegreen'),
                                        mode="lines+markers",
                                        name='Davies Bouldin DTW'), row=2, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances['calinski_harabasz_DTW'],
                                        line=dict(color='purple'),
                                        mode="lines+markers",
                                        name='Calinski Harabasz DTW'), row=3, col=1)
        # SoftDTW metric plots
        if 'softdtw' in distance_metrics:
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances['silhouette_softDTW'],
                                        line=dict(color='red'),
                                        mode="lines+markers",
                                        name='Silhouette Soft DTW'), row=1, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances['davies_bouldin_softDTW'],
                                        line=dict(color='green'),
                                        mode="lines+markers",
                                        name='Davies Bouldin Soft DTW'), row=2, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances['calinski_harabasz_softDTW'],
                                        line=dict(color='blue'),
                                        mode="lines+markers",
                                        name='Calinski Harabasz Soft DTW'), row=3, col=1)
                
    if all_performances[0][0].characteristics['transformation']!='none': #Feature based approach
        num_trans = len(transformations)
        nparray_performances = numpy.zeros((n_cls*num_trans,9))
        for metric in all_performances:
            nc=0
            for n_cluster in metric:
                if n_cluster.characteristics['distance_metric']=='Euclidean' and n_cluster.characteristics['transformation']=='DWT':
                    nparray_performances[nc][0] = n_cluster.performances.silhouette
                    nparray_performances[nc][1] = n_cluster.performances.davies_bouldin
                    nparray_performances[nc][2] = n_cluster.performances.calinski_harabasz
                elif n_cluster.characteristics['distance_metric']=='DTW' and n_cluster.characteristics['transformation']=='DWT':
                    nparray_performances[nc][3] = n_cluster.performances.silhouette
                    nparray_performances[nc][4] = n_cluster.performances.davies_bouldin
                    nparray_performances[nc][5] = n_cluster.performances.calinski_harabasz        
                elif n_cluster.characteristics['distance_metric']=='SoftDTW' and n_cluster.characteristics['transformation']=='DWT':
                    nparray_performances[nc][6] = n_cluster.performances.silhouette
                    nparray_performances[nc][7] = n_cluster.performances.davies_bouldin
                    nparray_performances[nc][8] = n_cluster.performances.calinski_harabasz
                if n_cluster.characteristics['distance_metric']=='Euclidean' and n_cluster.characteristics['transformation']=='DFT':
                    nparray_performances[nc][0] = n_cluster.performances.silhouette
                    nparray_performances[nc][1] = n_cluster.performances.davies_bouldin
                    nparray_performances[nc][2] = n_cluster.performances.calinski_harabasz
                elif n_cluster.characteristics['distance_metric']=='DTW' and n_cluster.characteristics['transformation']=='DFT':
                    nparray_performances[nc][3] = n_cluster.performances.silhouette
                    nparray_performances[nc][4] = n_cluster.performances.davies_bouldin
                    nparray_performances[nc][5] = n_cluster.performances.calinski_harabasz        
                elif n_cluster.characteristics['distance_metric']=='SoftDTW' and n_cluster.characteristics['transformation']=='DFT':
                    nparray_performances[nc][6] = n_cluster.performances.silhouette
                    nparray_performances[nc][7] = n_cluster.performances.davies_bouldin
                    nparray_performances[nc][8] = n_cluster.performances.calinski_harabasz
                nc=nc+1
        df_performances = pandas.DataFrame(nparray_performances, columns=['silhouette_ED_DWT', 'davies_bouldin_ED_DWT', 'calinski_harabasz_ED_DWT',
                                                                    'silhouette_DTW_DWT', 'davies_bouldin_DTW_DWT', 'calinski_harabasz_DTW_DWT',
                                                                    'silhouette_softDTW_DWT', 'davies_bouldin_softDTW_DWT', 'calinski_harabasz_softDTW_DWT'])
        # Euclidian metric plots with DWT transformation
        if 'euclidean' in distance_metrics:
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[0:n_cls,0],
                                        line=dict(color='magenta'),
                                        mode="lines+markers",
                                        name='Silhouette ED-DWT'), row=1, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[0:n_cls,1],
                                        line=dict(color='yellow'),
                                        mode="lines+markers",
                                        name='Davies Bouldin ED-DWT'), row=2, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[0:n_cls,2],
                                        line=dict(color='DeepSkyBlue'),
                                        mode="lines+markers",
                                        name='Calinski Harabasz ED-DWT'), row=3, col=1)
        # DTW metric plots with DWT transformation
        if 'dtw' in distance_metrics:
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[0:n_cls,3],
                                        line=dict(color='goldenrod'),
                                        mode="lines+markers",
                                        name='Silhouette DTW-DWT'), row=1, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[0:n_cls,4],
                                        line=dict(color='limegreen'),
                                        mode="lines+markers",
                                        name='Davies Bouldin DTW-DWT'), row=2, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[0:n_cls,5],
                                        line=dict(color='purple'),
                                        mode="lines+markers",
                                        name='Calinski Harabasz DTW-DWT'), row=3, col=1)
        # SoftDTW metric plots with DWT transformation
        if 'softdtw' in distance_metrics:
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[0:n_cls,6],
                                        line=dict(color='red'),
                                        mode="lines+markers",
                                        name='Silhouette Soft DTW-DWT'), row=1, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[0:n_cls,7],
                                        line=dict(color='green'),
                                        mode="lines+markers",
                                        name='Davies Bouldin Soft DTW-DWT'), row=2, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[0:n_cls,8],
                                        line=dict(color='blue'),
                                        mode="lines+markers",
                                        name='Calinski Harabasz Soft DTW-DWT'), row=3, col=1)
        # Euclidian metric plots with DFT transformation
        if 'euclidean' in distance_metrics:
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[n_cls:,0],
                                        line=dict(color='magenta'),
                                        mode="lines+markers",
                                        name='Silhouette ED-DFT'), row=1, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[n_cls:,1],
                                        line=dict(color='yellow'),
                                        mode="lines+markers",
                                        name='Davies Bouldin ED-DFT'), row=2, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[n_cls:,2],
                                        line=dict(color='DeepSkyBlue'),
                                        mode="lines+markers",
                                        name='Calinski Harabasz ED-DFT'), row=3, col=1)
        # DTW metric plots with DFT transformation
        if 'dtw' in distance_metrics:
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[n_cls:,3],
                                        line=dict(color='goldenrod'),
                                        mode="lines+markers",
                                        name='Silhouette DTW-DFT'), row=1, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[n_cls:,4],
                                        line=dict(color='limegreen'),
                                        mode="lines+markers",
                                        name='Davies Bouldin DTW-DFT'), row=2, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[n_cls:,5],
                                        line=dict(color='purple'),
                                        mode="lines+markers",
                                        name='Calinski Harabasz DTW-DFT'), row=3, col=1)
        # SoftDTW metric plots with DFT transformation
        if 'softdtw' in distance_metrics:
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[n_cls:,6],
                                        line=dict(color='red'),
                                        mode="lines+markers",
                                        name='Silhouette Soft DTW-DFT'), row=1, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[n_cls:,7],
                                        line=dict(color='green'),
                                        mode="lines+markers",
                                        name='Davies Bouldin Soft DTW-DFT'), row=2, col=1)
            fig.append_trace(go.Scatter(x=n_cluster_test_values, y=df_performances.iloc[n_cls:,8],
                                        line=dict(color='blue'),
                                        mode="lines+markers",
                                        name='Calinski Harabasz Soft DTW-DFT'), row=3, col=1)

    fig.update_yaxes(title_text="Silhouette", row=1, col=1)
    fig.update_yaxes(title_text="Davies Bouldin", row=2, col=1)
    fig.update_yaxes(title_text="Calinski Harabasz", row=3, col=1)
    fig.update_xaxes(title_text="Number of clusters", row=3, col=1)

    fig.update_layout(title='Performances with different number of clusters', height=750)
    g = dcc.Graph(
        figure=fig
    )
    return g


def plot_every_prediction(df: DataFrame, model_results: List[SingleResult],
                          main_accuracy_estimator: str, test_values: int):
    new_childrens = [html.Div("EXTRA: plot _EVERY_ prediction\n")]

    model_results.sort(key=lambda x: len(x.prediction))

    for r in model_results:
        predicted_data = r.prediction
        testing_performance = r.testing_performances
        plot = cluster_plot(df, predicted_data, test_values)
        plot.figure.update_layout(title="")
        new_childrens.extend([
            html.Div(main_accuracy_estimator.upper()
                     + ": " + str(getattr(testing_performance, main_accuracy_estimator.upper()))),
            plot
        ])

    return new_childrens


def characteristics_list(model_characteristics: dict, best_performances: SingleResult)-> html.Div: #, testing_performances: List[ValidationPerformance]) -> html.Div:
    """
    Create and return an HTML Div which contains a list of natural language characteristic
    relative to a prediction model.

    Parameters
    ----------
    model_characteristics : dict
        key-value for each characteristic to write in natural language.

    best_performances : SingleResult
        Useful to write also information about the best clustering performance.
    
    Returns
    -------
    html.Div()
    """

    def get_text_char(key: str, value: any) -> str:
        value = str(value)
        switcher = {
            "clustering_approach": "Clustering approach: " + value,
            "model": "Model type: " + value,
            "distance_metric": 'Distance metrics used: ' + value,
            "n_clusters":'Number of clusters tested: ' + value,
            "transformation": ('The model has used a ') + value + (
                ' transformation on the input data.') if value != "none"
            else ('The model has not used any pre/post transformation on input data.')
        }
        return switcher.get(key, "Invalid choice!")

    elems = [html.Div('Model characteristics:'),
             html.Ul([html.Li(get_text_char(key, model_characteristics[key])) for key in model_characteristics]),
             html.Div("This model using the best clustering parameters, reaches the next performances:"),
             show_errors_html(best_performances)
            ]

    return html.Div(elems)


def show_errors_html(best_performances: SingleResult) -> html.Ul:
    """
    Create an HTML list with the best performance evaluation criteria result.

    Parameters
    ----------
    best_performances :  SingleResult
        Error metrics to show.

    Returns
    -------
    html.Ul
        HTML list with all the error-metrics.
    """
    import math

    def round_n(n: float):
        dec_part, int_part = math.modf(n)

        if abs(int_part) > 1:
            return str(round(n, 3))
        else:
            return format(n, '.3g')

    def get_text_char(key: str, value: any) -> str:
        switcher = {
            "silhouette": "Silhouette score: " + value,
            "davies_bouldin": "Davies Bouldin score: " + value,
            "calinski_harabasz": "Calinski Harabasz score: " + value,
            "distance_metric": "Best distance metric: " + value,
            "n_clusters": "Best number of clusters: " + value,
            "transformation": ('The best model has used a ') + value + (
                ' transformation on the input data.') if value != "none"
            else ('The model has not used any pre/post transformation on input data.')
        }
        return switcher.get(key, "Invalid choice!")

    best_performances_dict = best_performances.performances.get_dict()
    for key in best_performances_dict:
        best_performances_dict[key] = round_n(best_performances_dict[key])
    best_performances_dict['distance_metric'] = best_performances.characteristics['distance_metric']
    best_performances_dict['n_clusters'] = str(best_performances.characteristics['n_clusters'])
    best_performances_dict['transformation'] = best_performances.characteristics['transformation']

    return html.Ul([html.Li(get_text_char(key, best_performances_dict[key])) for key in best_performances_dict])


def validation_performance_info()-> html.Div:
    """
    Create and return an HTML Div which contains a information of the performance scores..

    Parameters
    ----------
    None
    
    Returns
    -------
    html.Div()
    """
    """
    info = [html.Div('Silhouette Coefficient:'
                     'The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.'
                     'Scores around zero indicate overlapping clusters. The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.'
                     'The Silhouette Coefficient is generally higher for convex clusters than other concepts of clusters'),
            html.Div('Calinski-Harabasz Index:'
                     'Also known as the Variance Ratio Criterion.The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.'
                     'The index is the ratio of the sum of between-clusters dispersion and of within-cluster dispersion for all clusters (where dispersion is defined as the sum of distances squared)'),
            html.Div('Davies-Bouldin Index:'
                     'It can be used to evaluate the model, where a lower Davies-Bouldin index relates to a model with better separation between the clusters.'
                     'Zero is the lowest possible score. Values closer to zero indicate a better partition.'),
            ]
    """
    markdown_text = '''
        **Silhouette Coefficient:**
        The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.
        Scores around zero indicate overlapping clusters. The score is higher when clusters are dense and well separated.
        The Silhouette Coefficient is generally higher for convex clusters than other concepts of clusters.
        
        **Calinski-Harabasz Index:**
        Also known as the Variance Ratio Criterion, the score is higher when clusters are dense and well separated.
        The index is the ratio of the sum of between-clusters dispersion and of within-cluster dispersion for all clusters 
        (where dispersion is defined as the sum of distances squared)
        
        **Davies-Bouldin Index:**
        It can be used to evaluate the model, where a lower Davies-Bouldin index relates to a model with better separation between the clusters.
        Zero is the lowest possible score. Values closer to zero indicate a better partition.
        '''
    
    return html.Div(dcc.Markdown(children=markdown_text))


def cluster_distribution_plot(cluster_indexes: DataFrame) -> dcc.Graph:
    """
    Create and return a plot which contains the cluster distribution.
    The plot is built using a dataframe: `ingested_data`.

    `ingested_data` includes the raw data ingested by the app, while `cluster_data` contains the cluster indexes, cluster characteristics
    and cluster centers made by a model.

    Parameters
    ----------
    cluster_indexes : DataFrame
    Clustering indexes, and index for each timeseries corresponding the cluster which each timeseries belongs.

    Returns
    -------
    g : dcc.Graph

    """

    fig = go.Figure()

    clusters = np.unique(cluster_indexes)
    count_arr = np.bincount(cluster_indexes)
    counts = []
    for num_cluster in clusters:
      counts.append(count_arr[num_cluster])

    fig.add_trace(go.Bar(x=clusters, y=counts))
        
    fig.update_layout(title=("Cluster Distribution"), xaxis_title='Clusters', yaxis_title='Count')

    g = dcc.Graph(
        figure=fig )
    return g
