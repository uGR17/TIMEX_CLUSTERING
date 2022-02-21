import itertools
import json
import pkgutil
import logging
import os
import numpy as np
import pandas as pd
import tslearn

from pandas import DataFrame
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from timexseries_clustering.data_clustering.models.predictor import ModelResult, SingleResult
from timexseries_clustering.data_clustering.validation_performances import ValidationPerformance
from timexseries_clustering.data_clustering import ClustersModel
from sklearn import mixture
from timexseries_clustering.data_clustering.transformation import transformation_factory


logging.getLogger('GaussianMixtureModel').setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def GaussianMixtureModel(ingested_data: DataFrame, clustering_approach: str, distance_metric: str, 
                param_config: dict, transformation: str = None, n_clusters: int = 3)->ModelResult:
    """
    Gaussian Mixture Clustering Model
    
    Parameters
    ----------
    clustering_approach : str
        Clustering approach, e.g. "observation_based"
    param_config : dict
        TIMEX configuration dictionary, to pass to the just created model.
    distance_metric : str, e.g. **
        Distance/similarity measure type, e.g. "euclidean, dtw, softdtw" **
    transformation : str, optional, default None
        Optional `transformation` parameter to pass to the just created model.
    n_clusters : int, optional, default 3
        Optional `number of clusters` parameter to pass to the just created model.
    
    Returns
    -------
    ModelResult
        Model Result of the class specified in `model_class`, it contains the 
        results of the best clustering with the index of the cluster that each 
        time series belongs to. Contains also the model characteristics and the 
        centers of each cluster.
    
    """
    
    seed=0
    model_centers = []
    model_characteristics = {}
    
    try:
        gamma = param_config["model_parameters"]["gamma"]
    except KeyError:
        gamma = 0.01
    try:
        pre_transformation = param_config["model_parameters"]["pre_transformation"]
    except KeyError:
        pre_transformation = "none"

    X = ingested_data.copy().transpose()
    gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full', verbose=False, random_state=seed)
    best_clusters = gmm.fit_predict(X.values)

    model_centers = gmm.means_
    model_centers_dataframe = pd.DataFrame(model_centers).T
    model_centers_dataframe = model_centers_dataframe.set_index(ingested_data.index.date)
    inverse_pre_transf = transformation_factory(pre_transformation)
    model_centers_dataframe = inverse_pre_transf.inverse(model_centers_dataframe.copy())

    model_characteristics["clustering_approach"] = clustering_approach
    model_characteristics["model"] = "Gaussian Mixture Model"
    model_characteristics["distance_metric"] = "Log-likelihood"
    model_characteristics["n_clusters"] = n_clusters
    model_characteristics["feature_transformation"] = transformation
    model_characteristics["pre_transformation"] = pre_transformation
    performance = ValidationPerformance()
    #performance.set_performance_stats(ingested_data.transpose(), best_clusters, None)
    performance.set_performance_stats(X.values, best_clusters)
    single_result = SingleResult(model_characteristics, performance)
    return ModelResult(best_clustering=best_clusters, results=[single_result],characteristics=model_characteristics,
                cluster_centers=model_centers_dataframe)