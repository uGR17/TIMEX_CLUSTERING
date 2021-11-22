import logging
import os
import pickle
from functools import reduce
from typing import Tuple, List

import dateparser
import numpy
import tslearn
from pandas import DataFrame

from timexseries_clustering.data_clustering import ClustersModel
from timexseries_clustering.data_clustering.models.predictor import ModelResult
from timexseries_clustering.data_clustering.models.arima_predictor import ARIMAModel
from timexseries_clustering.data_clustering.models.lstm_predictor import LSTMModel
from timexseries_clustering.data_clustering.models.mockup_predictor import MockUpModel
from timexseries_clustering.data_clustering.models.kmeans_cluster import KMeansModel
from timexseries_clustering.data_clustering.xcorr import calc_all_xcorr
from timexseries_clustering.timeseries_container import TimeSeriesContainer
from tslearn.clustering import TimeSeriesKMeans

log = logging.getLogger(__name__)


def get_best_univariate_clusters(ingested_data: DataFrame, param_config: dict, total_xcorr: dict = None) -> \
        Tuple[dict, list]:
    """
    Compute, for all the columns in `ingested_data` (every time-series) the best univariate clustering possible.
    This is done using the clustering approach specified in `param_config` and testing the effect of the different 
    clustering algorithms, similarity measurements and transformations specified in `param_config`. 
    Moreover, the best feature transformation found, across the possible ones, will be returned.

    Parameters
    ----------
    ingested_data : DataFrame
        Initial data of the time-series.
    param_config : dict
        TIMEX configuration dictionary. In particular, the `model_parameters` sub-dictionary will be used. In
        `model_parameters` the following options has to be specified:

        - `clustering_approach`: clustering approach which will be use (options: "observation_based", "feature_based" or "model_based").
        - `possible_transformations`: comma-separated list of transformations keywords (e.g. "none,DWT,DFT,SVD").
        - `distance_metric`: distance/similarity measure which will be use (e.g. "ED,DTW,arma").
        - `models`: comma-separated list of the models to use (e.g. "agglomerative, k_means").
        - `main_accuracy_estimator`: error metric which will be minimized as target by the procedure. E.g. "rand_index", "silhouette_index","sse".
    total_xcorr : dict, optional, default None
        Cross-correlation dictionary computed by `calc_all_xcorr`. The cross-correlation is actually not used in this
        function, however it is used to build the returned `timexseries.timeseries_container.TimeSeriesContainer`, if given.

    Returns
    ----------
    dict **
        Dictionary which assigns the best transformation for every used prediction model, for every time-series.
    list **
        A list of `timexseries.timeseries_container.TimeSeriesContainer` objects, one for each time-series.

    Examples
    --------
    Create some fake data:
    >>> dates = pd.date_range('2000-01-01', periods=30)  # Last index is 2000-01-30
    >>> ds = pd.DatetimeIndex(dates, freq="D")
    >>> a = np.arange(30, 60)
    >>> b = np.arange(60, 90)
    >>> timeseries_dataframe = DataFrame(data={"a": a, "b": b}, index=ds)

    And create the model configuration part of the TIMEX configuration dictionary:
    >>> param_config = {
    ...   "model_parameters": {
    ...     "clustering_approach": "observation_based",  # Clustering approach which will be tested.
    ...     "possible_transformations": "none,log_modified,DWT",  # Possible feature transformation to test.**
    ...     "distance_metric": "DTW,ED",  # Distance/similarity measure which will be tested.
    ...     "models": "k_means",  # Model(s) which will be tested.
    ...     "main_accuracy_estimator": "mae",
    ...     "delta_training_percentage": 20,  # Training windows will be incremented by the 20% each step...
    ...     "test_values": 5,  # Use the last 5 values as validation set.
    ...     "prediction_lags": 7,  # Predict the next 7 points after 2000-01-30.
    ...     }
    ... }

    Now, get the univariate predictions:
    >>> best_transformations, timeseries_outputs = get_best_univariate_clusters(timeseries_dataframe, param_config)

    Let's inspect the results. `best_transformations` contains the suggested feature transformations to use:
    >>> best_transformations
    {'fbprophet': {'a': 'none', 'b': 'none'}}

    It is reasonable with this simple data that no transformation is the best transformation.**
    We have the `timexseries.timeseries_container.TimeSeriesContainer` list as well:
    >>> timeseries_outputs
    [<timexseries.timeseries_container.TimeSeriesContainer at 0x7f62f45d1fa0>,
     <timexseries.timeseries_container.TimeSeriesContainer at 0x7f62d4e97cd0>]

    These are the `timexseries.timeseries_container.TimeSeriesContainer` objects, one for time-series `a` and one for `b`.
    Each one has various fields, in this case the most interesting one is `models`:**
    >>> timeseries_outputs[0].models
    {'fbprophet': <timexseries.data_prediction.models.predictor.ModelResult at 0x7f62f45d1d90>}

    This is the `timexseries.data_prediction.models.predictor.ModelResult` object for FBProphet that we have just computed.
    """

    case_name = [param_config["activity_title"]]
    clustering_approach = param_config["model_parameters"]["clustering_approach"]
    transformations = [*param_config["model_parameters"]["possible_transformations"].split(",")]
    dist_measures_to_test = [*param_config["model_parameters"]["distance_metric"].split(",")]
    models = [*param_config["model_parameters"]["models"].split(",")]
    main_accuracy_estimator = param_config["model_parameters"]["main_accuracy_estimator"]

    best_transformations = dict.fromkeys(models, {})
    timeseries_containers = []

    # Get the set of CPUs on which the calling process is eligible to run.
    try:
        max_threads = param_config['max_threads']
    except KeyError:
        try:
            max_threads = len(os.sched_getaffinity(0))
        except:
            max_threads = 1

    columns = ingested_data.columns

    for col in columns:
        model_results = {}
        model_centers = {}
        timeseries_data = ingested_data[[col]]
        xcorr = total_xcorr[col] if total_xcorr is not None else None

    for model in models:
        this_model_performances = []
        model_results[model] = {}
        model_centers[model] = {}
        log.info(f"Using model {model}...")

        for metric in dist_measures_to_test:
            log.info(f"Computing univariate clustering using approach: {clustering_approach} and distance metric: {metric}...")
            _result = model_factory(ingested_data, clustering_approach, model, distance_metric=metric, param_config=param_config, transformation=transformations)
            #_result = predictor.fit_predict(ingested_data.copy())
            #_result = predictor.launch_model(timeseries_data.copy(), max_threads=max_threads)

            #performances = _result.results
            #performances.sort(key=lambda x: getattr(x.testing_performances, main_accuracy_estimator.upper()))
            #performances = getattr(performances[0].testing_performances, main_accuracy_estimator.upper())

            #this_model_performances.append((_result, performances, transf))
            cluster_centers = _result.cluster_centers #**
            characteristics = _result.characteristics
            clusters_vector = _result.best_clustering

            this_model_performances.append((clusters_vector, metric))
            model_results[model][metric] = _result #clusters_vector
            model_centers[model][metric] = cluster_centers

        #this_model_performances.sort(key=lambda x: x[1])
        #best_tr = this_model_performances[0][2]
        #[log.debug(f"Error with {t}: {e}") for t, e in zip(map(lambda x: x[2], this_model_performances),
        #                                                    map(lambda x: x[1], this_model_performances))]
        #log.info(f"Best transformation for {col} using {model}: {best_tr}")
        #best_transformations[model][col] = best_tr
        #model_results[model] = this_model_performances[0][0]

    log.info(f"Process of {clustering_approach} clustering finished")
    timeseries_containers.append(
        TimeSeriesContainer(ingested_data, model_results, xcorr, model_centers)
    )
    
    #return best_transformations, timeseries_containers 
    return timeseries_containers 


def get_best_clusters(ingested_data: DataFrame, param_config: dict):
    """
    Starting from `ingested_data`, using the models/cross correlation settings set in `param_config`, return the best
    possible clustering in a `timexseries_clustering.timeseries_container.TimeSeriesContainer` for all the time-series in `ingested_data`.
    Parameters
    ----------
    ingested_data : DataFrame
        Initial data of the time-series.
    param_config : dict
        TIMEX CLUSTERING configuration dictionary. `get_best_univariate_clusters` and `get_best_multivariate_clusters` (multivariate_clustering will be realased in timexseries_clustering 2.0.0) will
        use the various settings in `param_config`.
    Returns
    -------
    list
        A list of `timexseries_clustering.timeseries_container.TimeSeriesContainer` objects, one for each time-series.
    Examples
    --------
    This is basically the function on top of `get_best_univariate_clusters` and `get_best_multivariate_predictions`:
    it will call first the univariate and then the multivariate if the cross-correlation section is present in `param_config`.
    Create some data:
    >>> dates = pd.date_range('2000-01-01', periods=30)  # Last index is 2000-01-30
    >>> ds = pd.DatetimeIndex(dates, freq="D")
    >>> a = np.arange(30, 60)
    >>> b = np.arange(60, 90)
    >>>
    >>> timeseries_dataframe = DataFrame(data={"a": a, "b": b}, index=ds)
    Simply compute the clustering and get the returned `timexseries_clustering.timeseries_container.TimeSeriesContainer` objects:
    >>> timeseries_outputs = get_best_clusters(timeseries_dataframe, param_config)
    """

    if "xcorr_parameters" in param_config and len(ingested_data.columns) > 1:
        log.info(f"Computing the cross-correlation...")
        total_xcorr = calc_all_xcorr(ingested_data=ingested_data, param_config=param_config)
    else:
        total_xcorr = None

    timeseries_containers = get_best_univariate_clusters(ingested_data, param_config, total_xcorr)
    #best_transformations, timeseries_containers = get_best_univariate_clusters(ingested_data, param_config, total_xcorr)
    """ **
    if total_xcorr is not None or "additional_regressors" in param_config:
        timeseries_containers = get_best_multivariate_predictions(timeseries_containers=timeseries_containers, ingested_data=ingested_data,
                                                      best_transformations=best_transformations,
                                                      total_xcorr=total_xcorr,
                                                      param_config=param_config)
    """
    return timeseries_containers


def create_timeseries_containers(ingested_data: DataFrame, param_config: dict):
    """
    Entry points of the pipeline; it will compute univariate (multivariate in future realeases 2.0.0) clustering, or only
    create the containers with the time-series data, according to the content of `param_config`, with this logic:

    - if `model_parameters` is in `param_config`, then `get_best_clusters` will be called;
    - else, create a list of `timexseries.timeseries_container.TimeSeriesContainer` with only the time-series data and, if
      `xcorr_parameters` is in `param_config`, with also the cross-correlation.

    Parameters
    ----------
    ingested_data : DataFrame
        Initial data of the time-series.

    param_config : dict
        TIMEX configuration dictionary.

    Returns
    -------
    list
        A list of `timexseries_clustering.timeseries_container.TimeSeriesContainer` objects, one for each time-series.

    Examples
    --------
    The first example of `get_best_clusters` applies also here; calling `create_timeseries_containers` will
    produce the same identical result.

    However, if no clustering should be made but we just want the time-series containers:
    >>> dates = pd.date_range('2000-01-01', periods=30)  # Last index is 2000-01-30
    >>> ds = pd.DatetimeIndex(dates, freq="D")
    >>> a = np.arange(30, 60)
    >>> b = np.arange(60, 90)
    >>> timeseries_dataframe = DataFrame(data={"a": a, "b": b}, index=ds)

    Create the containers:
    >>> param_config = {}
    >>> timeseries_outputs = create_timeseries_containers(timeseries_dataframe, param_config)

    Check that no models, no historical predictions and no cross-correlation are present in the containers:
    >>> print(timeseries_outputs[0].models)
    None
    >>> print(timeseries_outputs[0].historical_prediction)
    None
    >>> print(timeseries_outputs[0].xcorr)
    None

    If `xcorr_parameters` was specified, then the last command would not return None.
    Check that the time-series data is there:

    >>> print(timeseries_outputs[0].timeseries_data)
                 a
    2000-01-01  30
    2000-01-02  31
    ...
    2000-01-29  58
    2000-01-30  59
    """
    if "model_parameters" in param_config:
        log.debug(f"Computing best clustering.")
        timeseries_containers = get_best_clusters(ingested_data, param_config)
    else:
        log.debug(f"Creating containers only for data visualization.")
        timeseries_containers = []
        if "xcorr_parameters" in param_config and len(ingested_data.columns) > 1:
            total_xcorr = calc_all_xcorr(ingested_data=ingested_data, param_config=param_config)
        else:
            total_xcorr = None

        for col in ingested_data.columns:
            timeseries_data = ingested_data[[col]]
            timeseries_xcorr = total_xcorr[col] if total_xcorr is not None else None
            timeseries_containers.append(
                TimeSeriesContainer(timeseries_data, None, timeseries_xcorr)
            )

    return timeseries_containers


def model_factory(ingested_data: DataFrame, clustering_approach: str, model_class: str, distance_metric: str, param_config: dict, transformation: str = None): #-> ClustersModel:
    """
    Given the clustering_approach and name of the model, return the corresponding ClustersModel.

    Parameters
    ----------
    clustering_approach : str
        Clustering approach, e.g. "observation_based"
    model_class : str
        Model type, e.g. "k_means"
    param_config : dict
        TIMEX configuration dictionary, to pass to the just created model.
    distance_metric : str, e.g. **
        Distance/similarity measure type, e.g. "DTW, ED" **
    transformation : str, optional, default None
        Optional `transformation` parameter to pass to the just created model.

    Returns
    -------
    ModelResult
        Model Result of the class specified in `model_class`, it contains the 
        results of the best clustering with the index of the cluster that each 
        time series belongs to. Contains also the model characteristics and the 
        centers of each cluster.

    Examples
    --------
    >>> param_config = {
    ...    "model_parameters": {
    ...        "possible_transformations": "none,log_modified",
    ...        "main_accuracy_estimator": "mae",
    ...        "delta_training_percentage": 20,
    ...        "test_values": 5,
    ...        "prediction_lags": 7,
    ...    },
    ...}

    >>> model = model_factory("fbprophet", param_config, "none")me
    >>> print(type(model))
    <class 'timexseries.data_prediction.models.prophet_predictor.FBProphetModel'>
    """

    if clustering_approach == "observation_based":
        if model_class == "k_means": #fbprophet
            #return KMeansModel(params=param_config, distance_metric=distance_metric, transformation=transformation)
            try:
                n_clusters = int(param_config["model_parameters"]["n_clusters"])
            except KeyError:
                n_clusters = 3
            try:
                gamma = float(param_config["model_parameters"]["gamma"])
            except KeyError:
                gamma = 0.01
            
            seed=0
            model_centers = []
            model_characteristics = {}
            if distance_metric == "ED": #fbprophet
                log.info(f"Computing k means with ED metric...")
                km = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", verbose=False, random_state=seed)
                best_clusters = km.fit_predict(ingested_data.transpose())
                for yi in range(n_clusters):
                    centrd = km.cluster_centers_[yi].ravel()
                    model_centers.append(centrd)
                model_characteristics["clustering_approach"] = clustering_approach
                model_characteristics["model"] = model_class
                model_characteristics["n_clusters"] = n_clusters
                model_characteristics["distance_metric"] = distance_metric
                model_characteristics["transformation"] = transformation
                return ModelResult(cluster_centers=model_centers, characteristics=model_characteristics,
                            best_clustering=best_clusters)
                #return best_clusters, model_centers
                
            if distance_metric == "DTW":
                log.info(f"Computing k means with DTW metric...")
                km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=False, max_iter_barycenter=10, random_state=seed)
                best_clusters = km.fit_predict(ingested_data.transpose())
                for yi in range(n_clusters):
                    centrd = km.cluster_centers_[yi].ravel()
                    model_centers.append(centrd)
                model_characteristics["clustering_approach"] = clustering_approach
                model_characteristics["model"] = model_class
                model_characteristics["n_clusters"] = n_clusters
                model_characteristics["distance_metric"] = distance_metric
                model_characteristics["transformation"] = transformation
                return ModelResult(cluster_centers=model_centers, characteristics=model_characteristics,
                            best_clustering=best_clusters)
            if distance_metric == "soft_DTW":
                log.info(f"Computing k means with soft_DTW metric...")
                km = TimeSeriesKMeans(n_clusters=n_clusters, metric="softdtw", verbose=False, metric_params={"gamma": gamma}, random_state=seed)
                best_clusters = km.fit_predict(ingested_data.transpose())
                for yi in range(n_clusters):
                    centrd = km.cluster_centers_[yi].ravel()
                    model_centers.append(centrd)
                model_characteristics["clustering_approach"] = clustering_approach
                model_characteristics["model"] = model_class
                model_characteristics["n_clusters"] = n_clusters
                model_characteristics["distance_metric"] = distance_metric
                model_characteristics["transformation"] = transformation
                return ModelResult(cluster_centers=model_centers, characteristics=model_characteristics,
                            best_clustering=best_clusters)
        #if model_class == "mockup":
        #    return MockUpModel(param_config, distance_metric)
        #else:
        #    return ARIMAModel(param_config, distance_metric)
    
    if clustering_approach == "feature_based":
        if model_class == "k_means":
            print("feature_based in progress")
    
    if clustering_approach == "model_based":
        if model_class == "k_means": #fbprophet
            print("model_based in progress")
