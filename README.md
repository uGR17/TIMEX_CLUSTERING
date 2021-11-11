# TIMEX_CLUSTERING
Library for time-series-clustering-as-a-service.

[![Tests with PyTest](https://github.com/AlexMV12/TIMEX/actions/workflows/run_tests.yml/badge.svg)](https://github.com/AlexMV12/TIMEX/actions/workflows/run_tests.yml)
![Coverage](badges/coverage.svg)
![PyPI](https://img.shields.io/pypi/v/timexseries)
![PyPI - Downloads](https://img.shields.io/pypi/dm/timexseries)

TIMEX_CLUSTERING (referred in code as `timexseries_clustering`) is a framework for time-series-clustering-as-a-service.

Its main goal is to provide a simple and generic tool to build websites and, more in general,
platforms, able to provide the clustering of time-series in the "as-a-service" manner.

This means that users should interact with the service as less as possible.


## Installation
The main two dependencies of TIMEX CLUSTERING are [Tslearn] and [PyTorch](https://pytorch.org/). 
If you prefer, you can install them beforehand, maybe because you want to choose the CUDA/CPU
version of Torch.

However, installation is as simple as running:

`pip install timexseries_clustering`

## Get started
Please, refer to the Examples folder. You will find some Jupyter Notebook which illustrate
the main characteristics of TIMEX CLUSTERING. A Notebook explaining the covid-timex.it website is present,
along with the source code of the site, [here](https://github.com/uGR17/TIMEX_CLUSTERING).

## Documentation
The full documentation is available at [here](https://ugr17.github.io/TIMEX_CLUSTERING/timexseries/).

## Contacts
If you have questions, suggestions or problems, feel free to open an Issue.
You can contact us at:

- uriel.guadarrama@polimi.it
- alessandro.falcetta@polimi.it
- manuel.roveri@polimi.it

