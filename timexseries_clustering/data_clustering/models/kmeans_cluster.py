import itertools
import json
import pkgutil
import logging
import os
import numpy as np
import pandas as pd
import tslearn

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from pandas import DataFrame
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from timexseries_clustering.data_clustering.models.predictor import ModelResult, SingleResult
from timexseries_clustering.data_clustering.validation_performances import ValidationPerformance
from timexseries_clustering.data_clustering import ClustersModel

logging.getLogger('kMeansModel').setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def KMeansModel(ingested_data: DataFrame, clustering_approach: str, distance_metric: str, 
                param_config: dict, transformation: str = None, n_clusters: int = 3)->ModelResult:
    """
    K Means clustering model
    
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
    
    if distance_metric == "euclidean":
        km = TimeSeriesKMeans(n_clusters=n_clusters, metric=distance_metric, verbose=False, random_state=seed)
        best_clusters = km.fit_predict(ingested_data.copy().transpose())
        for yi in range(n_clusters):
            centrd = km.cluster_centers_[yi].ravel()
            model_centers.append(centrd)
        model_centers_dataframe = pd.DataFrame(model_centers).T
        model_centers_dataframe = model_centers_dataframe.set_index(ingested_data.index.date)
        model_characteristics["clustering_approach"] = clustering_approach
        model_characteristics["model"] = "K Means"
        model_characteristics["distance_metric"] = "Euclidean"
        model_characteristics["n_clusters"] = n_clusters
        model_characteristics["transformation"] = transformation
        performance = ValidationPerformance()
        performance.set_performance_stats(ingested_data.transpose(), best_clusters, distance_metric)
        single_result = SingleResult(model_characteristics, performance)
        return ModelResult(best_clustering=best_clusters, results=[single_result],characteristics=model_characteristics,
                    cluster_centers=model_centers_dataframe)
    
    if distance_metric == "dtw":
        km = TimeSeriesKMeans(n_clusters=n_clusters, metric=distance_metric, verbose=False, max_iter_barycenter=10, random_state=seed)
        best_clusters = km.fit_predict(ingested_data.copy().transpose())
        performance = float(silhouette_score(ingested_data.transpose(), best_clusters, metric=distance_metric))
        for yi in range(n_clusters):
            centrd = km.cluster_centers_[yi].ravel()
            model_centers.append(centrd)
        model_centers_dataframe = pd.DataFrame(model_centers).T
        model_centers_dataframe = model_centers_dataframe.set_index(ingested_data.index.date)
        model_characteristics["clustering_approach"] = clustering_approach
        model_characteristics["model"] = "K Means"
        model_characteristics["distance_metric"] = "DTW"
        model_characteristics["n_clusters"] = n_clusters
        model_characteristics["transformation"] = transformation
        performance = ValidationPerformance()
        performance.set_performance_stats(ingested_data.transpose(), best_clusters, distance_metric)
        single_result = SingleResult(model_characteristics,performance)
        return ModelResult(best_clustering=best_clusters, results=[single_result],characteristics=model_characteristics,
                    cluster_centers=model_centers_dataframe)
        
    if distance_metric == "softdtw":
        km = TimeSeriesKMeans(n_clusters=n_clusters, metric=distance_metric, verbose=False, metric_params={"gamma": gamma}, random_state=seed)
        best_clusters = km.fit_predict(ingested_data.copy().transpose())
        performance = float(silhouette_score(ingested_data.transpose(), best_clusters, metric=distance_metric))
        for yi in range(n_clusters):
            centrd = km.cluster_centers_[yi].ravel()
            model_centers.append(centrd)
        model_centers_dataframe = pd.DataFrame(model_centers).T
        model_centers_dataframe = model_centers_dataframe.set_index(ingested_data.index.date)
        model_characteristics["clustering_approach"] = clustering_approach
        model_characteristics["model"] = "K Means"
        model_characteristics["distance_metric"] = "SoftDTW"
        model_characteristics["n_clusters"] = n_clusters
        model_characteristics["transformation"] = transformation
        performance = ValidationPerformance()
        performance.set_performance_stats(ingested_data.transpose(), best_clusters, distance_metric)
        single_result = SingleResult(model_characteristics,performance)
        return ModelResult(best_clustering=best_clusters, results=[single_result],characteristics=model_characteristics,
                    cluster_centers=model_centers_dataframe)