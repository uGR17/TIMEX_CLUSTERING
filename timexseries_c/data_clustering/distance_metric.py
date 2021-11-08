from pandas import Series
import numpy as np
from pandas import DataFrame
from scipy.stats import yeojohnson

import tslearn
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler


class Distance_metric:
    """
    Super-class used to represent various types of Distance metrics.
    """

    def apply(self, data: Series) -> Series:
        """
        Apply the transformation on each value in a Pandas Series. Returns the transformed Series, i.e. a Series with
        transformed values.

        Note that it is not guaranteed that the dtype of the returned Series is the same of `data`.

        Parameters
        ----------
        data : Series
            Data to transform.

        Returns
        -------
        Series
            Transformed data.
        """
        pass

    def inverse(self, data: Series) -> Series:
        """
        Apply the inverse of the transformation on the values of a Pandas Series of transformed values.
        Returns the data re-transformed back to the real world.

        Any class implementing Transformation should make the `inverse` method always return a Series with the same
        shape as the one of `data`. If the function is not invertible (e.g. Log), the returning values should be
        approximated. It is assumed in the rest of TIMEX that `inverse` does not fail.

        Parameters
        ----------
        data : Series
            Data to transform.

        Returns
        -------
        Series
            Transformed data.
        """
        pass


class ED(Distance_metric):
    """Class corresponding to a Euclidian Distance ED distance measure.

    Notes
    -----
    The actual function computed by this transformation is:

    .. math::
        f(x) = sign(x) * log(|x|)

    if `x` > 1, 0 otherwise.

    Note that this way time-series which contain 0 values will have its values modified, because `inverse` will return
    1 instead of 0 when returning the transformed time-series to the real world.

    The inverse function, indeed, is:

    .. math::
        f^{-1}(x) = sign(x) * e^{abs(x)}

    LogModified should be preferred.
    """
    def apply(self, data: Series) -> Series:
        seed=0
        km = TimeSeriesKMeans(n_clusters=3, verbose=True, random_state=seed)
        y_pred = km.fit_predict(data)
        return data.apply(lambda x: np.sign(x) * np.log(abs(x)) if abs(x) > 1 else 0)

    def inverse(self, data: Series) -> Series:
        return data.apply(lambda x: np.sign(x) * np.exp(abs(x)))

    def __str__(self):
        return "Log"


class sof_DWT(Distance_metric):
    """Class corresponding to the a custom variant of logarithmic feature transformation.
    In particular, this transformation tries to overcome the traditional issues of a logarithmic transformation, i.e.
    the impossibility to work on negative data and the different behaviour on 0 < x < 1.

    Notes
    -----
    The actual function computed by this transformation is:

    .. math::
        f(x) = sign(x) * log(|x| + 1)

    The inverse, instead, is:

    .. math::
        f^{-1}(x) = sign(x) * e^{(abs(x) - sign(x))}
    """
    def apply(self, data: Series) -> Series:
        return data.apply(lambda x: np.sign(x) * np.log(abs(x) + 1))

    def inverse(self, data: Series) -> Series:
        return data.apply(lambda x: np.sign(x) * np.exp(abs(x)) - np.sign(x))

    def __str__(self):
        return "modified Log"


class DWT(Distance_metric):
    """Class corresponding to the identity transformation.
    This is useful because the absence of a data pre-processing transformation would be a particular case for functions
    which compute predictions; instead, using this, that case is not special anymore.

    Notes
    -----
    The actual function computed by this transformation is:

    .. math::
        f(x) = x

    The inverse, instead, is:

    .. math::
        f^{-1}(x) = x
    """
    def apply(self, data: Series) -> Series:
        return data

    def inverse(self, data: Series) -> Series:
        return data

    def __str__(self):
        return "none"


class YeoJohnson(Distance_metric):
    """Class corresponding to the Yeo-Johnson transformation.

    Notes
    -----
    Introduced in [^1], this transformation tries to make the input data more stable.

    Warnings
    --------
    .. warning:: Yeo-Johnson is basically broken for some series with high values.
                 Follow this issue: https://github.com/scikit-learn/scikit-learn/issues/14959
                 Until this is solved, Yeo-Johnson may not work as expected and create random crashes.

    References
    ----------
    [^1]: Yeo, I. K., & Johnson, R. A. (2000). A new family of power transformations to improve normality or symmetry.
          Biometrika, 87(4), 954-959. https://doi.org/10.1093/biomet/87.4.954
    """

    def __init__(self):
        self.lmbda = 0

    def apply(self, data: Series) -> Series:
        res, lmbda = yeojohnson(data)
        self.lmbda = lmbda
        return res

    def inverse(self, data: Series) -> Series:
        lmbda = self.lmbda
        x_inv = np.zeros_like(data)
        pos = data >= 0

        # when x >= 0
        if abs(lmbda) < np.spacing(1.):
            x_inv[pos] = np.exp(data[pos]) - 1
        else:  # lmbda != 0
            x_inv[pos] = np.power(data[pos] * lmbda + 1, 1 / lmbda) - 1

        # when x < 0
        if abs(lmbda - 2) > np.spacing(1.):
            x_inv[~pos] = 1 - np.power(-(2 - lmbda) * data[~pos] + 1,
                                       1 / (2 - lmbda))
        else:  # lmbda == 2
            x_inv[~pos] = 1 - np.exp(-data[~pos])

        return Series(x_inv)

    def __str__(self):
        return f"Yeo-Johnson (lambda: {round(self.lmbda, 3)})"


def distance_metric_factory(dst_class: str) -> Distance_metric:
    """
    Given the type of the transformation, encoded as string, return the Transformation object.

    Parameters
    ----------
    tr_class : str
        Transformation type.

    Returns
    -------
    Transformation
        Transformation object.

    Examples
    --------
    Create a Pandas Series and apply the logarithmic transformation:

    >>> x = Series([2, 3, 4, 5])
    >>> tr = transformation_factory("log")
    >>> tr_x = tr.apply(x)
    >>> tr_x
    0    0.693147
    1    1.098612
    2    1.386294
    3    1.609438
    dtype: float64

    Now, let's compute the inverse transformation which should return the data to the real world:

    >>> inv_tr_x = tr.inverse(tr_x)
    >>> inv_tr_x
    0    2.0
    1    3.0
    2    4.0
    3    5.0
    dtype: float64
    """
    if dst_class == "ED":
        return ED()
    elif dst_class == "DWT":
        return DWT()
    elif dst_class == "sof_DWT":
        return sof_DWT()