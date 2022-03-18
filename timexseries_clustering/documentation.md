`TIMEX-CLUSTERING` (referred as `timexseries_clustering` in code) is a Python package which implements a pipeline for time-series clustering.

The pipeline consists in:

- time-series data ingestion;
- time-series pre-processing; 
- time-series analysis and clustering;
- results visualization.

While `TIMEX-CLUSTERING` is intended to be use in a fully automatic way (which means that each
step is automatically computed, without user intervention), users can also take
advantage of single parts of the framework.

## Reason
Time-series are one of the most important data shapes. An important part of data that we produce and consume everyday
can be encoded in the time-series form. For our purposes, time-series are an ordered list of values, each one associated with a timestamp. The values can be the number of units sold that day of a specific product, the heartbeat of a person, the amount of new Covid-19 cases in a specific country or city. 

Clustering time-series data has become a popular research topic over the past decades and there is a rich literature on this topic. Clustering is a category of unsupervised learning techniques that allows us to discover hidden patterns in data; these patterns can commonly take place in the dataset. Finding the clusters of time-series can help to real world problems such as anomaly detection (i.e. discover anomalies in sensor databases), recognizing dynamic changes in time-series (i.e. in financial data bases to find companies with similar stock price move), forecasting (i.e. infer likely future behavior based on cluster membership) and for pattern discovery (i.e. in marketing databases different daily patterns of sales can be discovered).

The clustering of time-series is organized into three approaches:
The observation-based approach is a direct approach for clustering time-series based on a comparison of the observed time-series or a suitable transformation of the observed time-series. The observation-based approach is recommended when the aim is to identify similar geometric profiles and when the time-series are not very long.

The feature-based metrics are more suitable when the aim is to discriminate between generating processes, that is, between underlying dependence structures. In this approach the clustering target is achieved by using the features extracted, these features highlight higher level dynamic aspects of the time-series; through this approach a dimensional reduction is attained and consequently computational time saving can be achieved. 

Finally, in the model-based methods, the time-series under consideration are assumed to have been generated from specific underlying models or by a combination of probability distributions, and the similarity between fitted models is evaluated .

The proposed TIMEX-CLUSTERING framework, developed in Python, is characterized by the pipeline depicted in. Such pipeline comprises the following six steps: data ingestion, data pre-processing, data description, data clustering and service delivery.

TIMEX-CLUSTERING's goal is to provide an automatic clustering framework, which could be used by any user on their datasets with a minimal amount of knowledge about clustering procedures. It should provide efficient and high quality time-series clustering. TIMEX-CLUSTERING will use the three different clustering approaches (observation-based, feature-based and model-based) in order to cover datasets with different nature. It is emphasized that a clustering-as-a-service is not yet available in the literature: this framework will be also unique because will include all the clustering approaches.

![Pipeline of TIMEX-CLUSTERING](https://github.com/uGR17/TIMEX_CLUSTERING/blob/5573c65f90fb99922524d4f32da0625c8885112d/examples/figures/pipeline.PNG)

## Use cases
There are at least two use cases worth mentioning:

1. A certain group of time-series is relevant for a group of users. The time-series are updated every
day and can be downloaded in a suitable format (e.g. CSV or XLSX) every day.   
Those users would like to have a way to obtain the clusters of such time-series.
   by installing TIMEX-CLUSTERING on a server, configure the bare minimum requested by TIMEX-CLUSTERING, and start it. A website will be made available, and it will provide a link to the users.
2. A company is interested in creating a clustering-as-a-service platform where users can upload a CSV file containing 
some time-series, set some settings or use defaults, and press a button to receive clusters for the uploaded 
   time-series. TIMEX-CLUSTERING can be adapted to work this way.
   
## Get started
Refer to the Examples section of the GitHub repository, or check the 
[source code](https://github.com/uGR17/TIMEX_CLUSTERING) 
[covid-19 timex-clustering example](https://github.com/uGR17/TIMEX_CLUSTERING/blob/ff89d8b55826436ea74ba142e4129f57b3b7d786/examples/COVID_Clustering.ipynb)
[UAE&UCR dataset timex-clustering example](https://github.com/uGR17/TIMEX_CLUSTERING/blob/ff89d8b55826436ea74ba142e4129f57b3b7d786/examples/UEA_UCR_Dataset_Clustering_Example.ipynb)

## Architecture
The general architecture of TIMEX-CLUSTERING is here presented:

![Pipeline of TIMEX-CLUSTERING](https://github.com/uGR17/TIMEX_CLUSTERING/blob/5573c65f90fb99922524d4f32da0625c8885112d/examples/figures/pipeline.PNG)

The most important aspect of TIMEX-CLUSTERING architecture is the __configuration options__ dictionary. It is a key-value list of settings which influences all the key aspects of the pipeline. It can be stored in a JSON file, loaded before the 
pipeline starts.

## Configuration parameters dictionary
An example of working dictionary, used for the [UAE&UCR dataset timex-clustering example](https://github.com/uGR17/TIMEX_CLUSTERING/blob/ff89d8b55826436ea74ba142e4129f57b3b7d786/examples/UEA_UCR_Dataset_Clustering_Example.ipynb) website is:

```json
{
  "activity_title": "UEA&UCR Dataset - Clustering Example",
  "verbose": "INFO",
  "input_parameters": {
    "source_data_url": "https://raw.githubusercontent.com/uGR17/TIMEX_CLUSTERING/main/examples/datasets/k_means_example_5ts.csv",
    "index_column_name": "date",
    "frequency": "D",
    "timeseries_names":{
      "date": "Date",
      "ts1": "timeseries1",
      "ts2": "timeseries2",
      "ts3": "timeseries3",
    }
  },
  "model_parameters": {
     "clustering_approach": "observation_based,feature_based,model_based",
     "models": "k_means,gaussian_mixture",
     "pre_transformation": "none",
     "distance_metric": "euclidean,dtw,softdtw",
     "feature_transformations": "DWT",
     "n_clusters": [3, 4, 5, 6],
     "gamma": 0.01,
     "main_accuracy_estimator": "silhouette"
  },
  "xcorr_parameters": {
    "xcorr_max_lags": 120,
    "xcorr_mode": "pearson",
    "xcorr_mode_target": "pearson"
  },
  "visualization_parameters": {
    "xcorr_graph_threshold": 0.8,
  }
}
```

All the options are here described.

### General

- `activity_title`: used in the website creation as title for the page. Optional.
- `verbose`: it is not strictly used by TIMEX, but it can be included here in order to set the verbosity level in the
  main Python script. In fact, TIMEX has important logging capabilities. Optional, can be one of "DEBUG", "INFO", 
  "WARNING", "ERROR", "CRITICAL".
- `max_threads`: set the max number of threads (processes) used by TIMEX during the forecasting models computation. 
  Optional, if not set TIMEX will try to understand the number of available cores on the system.
  
### Input parameters
Remember that TIMEX expects the input data to be in the following form. Some changes can be forced through settings.

| time_index | time-series A | time-series B | ... |
|------------|---------------|---------------|-----|
| 2000-01-01 | x             | x             | ... |
| 2000-01-02 | x             | x             | ... |
| ...        | ...           | ...           | ... |


The following options has to be specified:

- `source_data_url`: local or remote URL pointing to a CSV file. 

Additionally, some other parameters can be specified:

- `index_column_name`: the name of the column to use as index for the DataFrame. If not specified the first one will
  be used. This column's values will be parsed with [dateparser](https://pypi.org/project/dateparser/) to obtain a 
  DateTimeIndex;
- `frequency`: if specified, the corresponding frequency will be forced. Refer to
  https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases for a list of possible
  values. If not specified the frequency will be infered.
- `columns_to_load_from_url`: comma-separated string of columns's names which will be read from the CSV file. If not
  specified, all columns will be read;
- `add_diff_column`: comma-separated string of columns's names for which a corresponding column containing the diff
  values should be created. They will be created with the name `col_name_diff`. Note that the first row of the
  dataset will be discarded;
- `timeseries_names`: dictionary of key-values (old_name: new_name) used to rename some columns in the CSV;
- `dateparser_options`: dictionary of key-values which will be given to `dateparser.parse()`. This is useful to set the
  date format if it is not common. Refer to `dateparser` documentation for the available options.
  
### Selection parameters
These parameters can be used to filter some points in the dataset. This is optional.

- `column_name_selection`: if specified, only the rows in which the value of the column named `column_name_selection`
  is equal to `value_selection` are kept. If this is specified, also `value_selection` has to be specified.
- `init_datetime`: if specified, only the rows where the Datetimeindex value is greater than `init_datetime` are kept.
- `end_datetime`: if specified, only the rows where the Datetimeindex value is less than `end_datetime` are kept.

Moreover, if `dateparser_options` is specified in `input_parameters', then the options will be passed to
dateparser to parse the dates.

### Model parameters
These parameters control the classes of models used in the forecasting, the feature transformation tested, etc.

The following options has to be specified:

- `possible_transformations`: comma-separated list of transformations keywords. Check the 
  [available transformations](#available-transformations). Example: `"none,log_modified"`.
- `main_accuracy_estimator`: error metric which will be minimized as target by the procedure. It can be one of `"mae"`,
  `"mse"`.
- `models`: comma-separated list of the models to use. Check the [available models](#available-forecasting-models). 
  Example: `"fbprophet,exponential_smoothing`".
  
Moreover, some other options can be set. Otherwise default values will be used:

- `test_values`: number of points used as validation in order to choose the transformation and training window. These
  points will be used, however, to compute the actual forecast. Has precedence over `test_percentage`.
- `test_percentage`: Percentage of the time-series length to used for the validation set. 
- `prediction_lags`: Number of future lags for for the forecast.
- `delta_training_percentage`: Length, in percentage of the time-series length, of the training windows.
- `min_values`: Key-values dictioanry where key is the name of a column and value is the minimum expected value in that 
  column. If "_all" is in the dict the corresponding value will be used for all values in each column.
- `max_values`: Key-values where key is the name of a column and value is the maximum expected value in that column.
  If "_all" is in the dict the corresponding value will be used for all values in each column.
- `round_to_integer`: List of columns name which should be rounded to integer. If "_all" is in the list, all values in 
  every column will be rounded to integer.

### Cross-correlation parameters
These parameters controls the cross-correlation computation, which is used to determine if there are time-series useful
as extra-regressors for multivariate multiple-input-single-output forecasting models.
This part is optional.

- `xcorr_modes`: indicate the different algorithms which should be used to compute the cross-correlation.
  The computed cross-correlation will be shown in the data visualization part.
  The available choices are:

  - `matlab_normalized`: same as using the MatLab function xcorr(x, y, 'normalized')
  - `pearson` : use Pearson formula (NaN values are fillled to 0)
  - `kendall`: use Kendall formula (NaN values are filled to 0)
  - `spearman`: use Spearman formula (NaN values are filled to 0)
  
- `xcorr_max_lags` is the number of lags, both in positive and negative direction, to which the cross-correlation
  calculations should be limited to.
- `xcorr_mode_target`: which cross-correlation algorithm should be used as target in evaluating useful
  additional regressors. E.g. `"pearson"`.
- `xcorr_extra_regressor_threshold`: the minimum absolute value of cross-correlation which indicates a useful
  extra-regressor. E.g. 0.8.
  
### Additional regressors parameters
These are additional regressors, which can be used in multivariate multiple-input-single-output forecasting models for
which the future values are already known at the forecasting time. For example, if we have weather forecast for the next
two weeks and we think that the temperature influences the target time-series, we can feed it in TIMEX. This is 
optional.

The additional regressors should be put in a CSV file, in the form:

| time_index | additional regressor A | additional regressor B | ... |
|------------|------------------------|------------------------|-----|
| 2000-01-01 | x                      | x                      | ... |
| 2000-01-02 | x                      | x                      | ... |
| ...        | ...                    | ...                    | ... |

And the following options has to be specified:

- `additional_regressors`: Key-value dict where key is a column name, and value is the path of the CSV containing the
  additional regressors useful for that series.
  If some additional regressors may be useful for all the time-series in the dataset, we can specify them using the 
  "`_all`" key.
  Example:
  
```json
"additional_regressors": {
  "_all": "../datasets/AdditionalRegressors/Temperature.csv",
  "time-series A": "../datasets/AdditionalRegressors/RegressorUsefulForTimeSeriesA.csv"
}
```

### Visualization parameters
Parameters used to tune the visualization part of TIMEX.

- `language`: Can be one between `"en"` and `"it"`. TIMEX for now supports English and Italian.
- `xcorr_graph_threshold`: This controls the minumum value of cross-correlation for which an arc in the 
  cross-correlation graph is drawn. It should be between 0.0 and 1.0.

## Available forecasting models
The following models are implemented in TIMEX and can be used in the forecasting procedure:

- Facebook Prophet: This is the [Facebook Prophet](https://github.com/facebook/prophet) forecasting model. Keyword: 
  `fbprophet`.
- LSTM: A simple LSTM implementation built with [PyTorch](https://pytorch.org/). Keyword: `lstm`.
- Exponential Smoothing: An exponential smoothing model, with automatic choice of trend and seasonality period, built on
  top of [sklearn](https://scikit-learn.org/stable/). Keyword: `exponentialsmoothing`.
- ARIMA: Simple ARIMA model, with automatic tuning. Keyword: `arima`.  

## Available transformations
The following transformations are available on TIMEX_CLUSTERING, which will use them in order to pre-process the time-series and
check if this increases the forecasting performance on the validation set.

- Identity: don't modify the values. Keyword: `none`.
- Logarithmic: Simple logarithmic function, with support for negative values. It uses the formula: 
  $$f(x) = sign(x) * log(|x|)$$ 
  if `x` > 1, 0 otherwise. However, the modified logarithmic one should be preferred. Keyword: `log`.
- Modified logarithmic: Simple logarithmic function, adapted to work on all the real numbers. It uses the formula:
  $$f(x) = sign(x) * log(|x| + 1)$$
  Keyword: `log_modified`.
- Yeo-Johnson: transformation presented in this [paper](https://doi.org/10.1093/biomet/87.4.954). 
  Keyword: `yeo-johnson`
