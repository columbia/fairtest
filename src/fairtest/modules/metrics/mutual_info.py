"""
Mutual Information Metric.
"""

from .metric import Metric
import fairtest.modules.statistics.hypothesis_test as tests
import fairtest.modules.statistics.confidence_interval as intervals
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys


class NMI(Metric):
    """
    Normalized Mutual Information metric.
    """

    dataType = Metric.DATATYPE_CT

    @staticmethod
    def approx_stats(data, level):
        return mutual_info(data, norm=True, ci_level=level)

    @staticmethod
    def exact_test(data):
        return tests.permutation_test_ct(data)

    @staticmethod
    def exact_ci(data, level):
        return intervals.bootstrap_ci_ct(data,
                                         lambda s: mutual_info(s, norm=True),
                                         ci_level=level)

    @staticmethod
    def validate(sens, output, expl):
        if output.num_labels != 1:
            raise ValueError('NMI metric only usable for a single target')
        if expl is not None:
            raise ValueError('NMI metric not usable with explanatory features')
        if not sens.arity or not output.arity:
            raise ValueError('NMI metric only usable with categorical features')

    approx_LIMIT_P = 1000
    approx_LIMIT_CI = sys.maxint

    def abs_effect(self):
        return self.stats[0]

    def __str__(self):
        return 'NMI'


def mutual_info(data, norm=True, ci_level=None):
    """
    mutual information with or without normalization and confidence intervals.

    Parameters
    ----------
    data :
        a contingency table

    norm :
        whether the MI should be normalized

    ci_level :
        level for confidence intervals (or None)

    Returns
    -------
    ci_low :
        lower bound of confidence interval

    ci_high :
        upper bound of confidence interval

    pval :
        the corresponding p-value

    References
    ----------
    https://en.wikipedia.org/wiki/Mutual_information
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    # data smoothing
    data_smoothed = data.copy()
    data_smoothed += 1

    if data.shape[0] < 2 or data.shape[1] < 2:
        if ci_level is not None:
            return 0, 1, 0
        else:
            return 0

    # row/column sums
    sum_x = np.sum(data_smoothed, axis=1)
    sum_y = np.sum(data_smoothed, axis=0)

    # joint probabilities
    data_size = np.array(data_smoothed).sum()

    # entropies
    h_x = stats.entropy(sum_x)
    h_y = stats.entropy(sum_y)
    h_xy = stats.entropy(data_smoothed.flatten())

    mi = -h_xy + h_x + h_y

    # normalized mutual info
    if norm:
        if (h_x == 0) or (h_y == 0) or (mi == 0):
            mi = 0
        else:
            mi = mi/min(h_x, h_y)

    # no confidence levels, return single measure
    if ci_level is None:
        return mi

    gstat, pval, dof, _ = tests.g_test(data)

    ci_low, ci_high = intervals.ci_mi(gstat, dof, data_size, ci_level)

    if pval > 1-ci_level:
        ci_low = 0

    if norm:
        ci_low /= min(h_x, h_y)
        ci_high /= min(h_x, h_y)

    ci_low = max(ci_low, 0)
    ci_high = min(ci_high, 1)

    return ci_low, ci_high, pval
