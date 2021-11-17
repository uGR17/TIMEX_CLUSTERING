from pandas import DataFrame


class TimeSeriesContainer:
    """
    A TimeSeriesContainer collect all the relevant information useful to characterizethe time-series coming from
    the ingested dataset.

    Parameters
    ----------
    timeseries_data : DataFrame
        Historical time-series data, in the form of a DataFrame with a index and more than one data column.
    models : dict
        Dictionary of ModelResult objects, all trained on these time-series.
    xcorr : dict
        Cross-correlation between the data of this time-series and all the other ones.
    centroids : dict
        The historical prediction, i.e. the predictions computed on a rolling window on the historical data.
        This is useful to verify the performances of each model not only on the very last data, but throughout the
        history of the time-series, in a cross-validation fashion. This dictionary contains one entry for each model
        tested.
    """
    def __init__(self, timeseries_data: DataFrame, models: dict, xcorr: dict, centroids: dict = None):
        self.timeseries_data = timeseries_data
        self.models = models
        self.xcorr = xcorr
        self.centroids = centroids

    def set_centroids(self, centroids):
        self.centroids = centroids
