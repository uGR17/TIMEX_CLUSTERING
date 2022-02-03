from math import sqrt

from pandas import DataFrame
from tslearn.clustering import silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score


class ValidationPerformance:
    """
    Class for the summary of various statistical indexes relative to the performance of a clustering model.

    Parameters
    ----------
    None

    Attributes
    ----------
    silhouette: float
        Silhouette score. Default 0
    davies_bouldin: float
        Davies Bouldin score. Default 0
    calinski_harabasz: float
        Calinski Harabasz score. Default 0
    """
    def __init__(self):
        self.silhouette = 0
        self.davies_bouldin = 0
        self.calinski_harabasz = 0

    def set_performance_stats(self, dataset: DataFrame, labels: DataFrame, metric: str = None):
        """
        Set all the statistical indexes according to input data.

        Parameters
        ----------
        dataset: DataFrame
            Time series dataset stored in a Pandas DataFrame.
        labels: DataFrame
            Predicted labels for each time series.
        metric : string, optional, default None
            The metric to use when calculating distance between time series, optional because is used only for silhouette score. 
            Should be one of {‘dtw’, ‘softdtw’, ‘euclidean’} or a callable distance function or None. If ‘softdtw’ is passed, a 
            normalized version of Soft-DTW is used that is defined as sdtw_(x,y) := sdtw(x,y) - 1/2(sdtw(x,x)+sdtw(y,y)). If X 
            is the distance array itself, use metric="precomputed". If None, dtw is used.

        Examples
        --------
        >>> import numpy
        >>> from tslearn.generators import random_walks
        >>> from timexseries_clustering.data_clustering import ValidationPerformance
        >>> numpy.random.seed(0)
        >>> X = random_walks(n_ts=20, sz=16, d=1)
        >>> X = numpy.resize(X,(20,16))
        >>> labels = numpy.random.randint(2, size=20)

        Calculate the performances.
        >>> perf = ValidationPerformance()
        >>> perf.set_performance_stats(X, labels, 'euclidean')

        >>> print(perf.silhouette)
        0.09

        >>> print(perf.davies_bouldin)
        2.28
        """
        self.silhouette = silhouette_score(dataset, labels, metric)
        self.davies_bouldin = davies_bouldin_score(dataset, labels)
        self.calinski_harabasz = calinski_harabasz_score(dataset, labels)

    def get_dict(self) -> dict:
        """
        Return all the parameters, in a dict.

        Returns
        -------
        d : dict
            All the statistics, in a dict.

        Examples
        --------
        >>> perf = ValidationPerformance()
        >>> perf.set_performance_stats(actual_dataframe['a'], predicted_dataframe['yhat'])
        >>> perf.get_dict()
        {'first_used_index': None, 'MSE': 4.0, 'RMSE': 2.0, 'MAE': 2.0, 'AM': -2.0, 'SD': 0.0}
        """
        d = {}
        for attribute, value in self.__dict__.items():
            d[attribute] = value

        return d