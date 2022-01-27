from pandas import DataFrame


class TimeSeriesContainer:
    """
    A TimeSeriesContainer collect all the relevant information useful to characterizethe time-series coming from
    the ingested dataset.

    Parameters
    ----------
    timeseries_data : DataFrame
        Historical time-series data, in the form of a DataFrame with an index and more than one data column.
    approach : str
        Approach used to cluster time-series data.
    models : dict
        Dictionary of ModelResult objects, all trained on these time-series.
    xcorr : dict
        Cross-correlation between the data of this time-series and all the other ones.
    cluster_centers : dict **
        The cluster centers of the best clusters obtained. This is useful to verify the performances of each model. 
        This dictionary contains one entry for each modeltested.
    """
    def __init__(self, timeseries_data: DataFrame, approach: str, models: dict, xcorr: dict):
        self.timeseries_data = timeseries_data
        self.approach = approach
        self.models = models
        self.xcorr = xcorr
