"""
Correlation Metric.
"""

from .metric import Metric
import fairtest.modules.statistics.hypothesis_test as tests
import fairtest.modules.statistics.confidence_interval as intervals
import pandas as pd
import numpy as np
import scipy.stats as stats
from math import sqrt, atanh, tanh


class CORR(Metric):
    """
    Pearson Correlation Metric.
    """
    dataType = Metric.DATATYPE_CORR

    @staticmethod
    def approx_stats(data, conf):
        return correlation(data, conf=conf)

    @staticmethod
    def exact_test(data):
        return tests.permutation_test_corr(data[data.columns[0]],
                                           data[data.columns[1]])

    @staticmethod
    def exact_ci(data, conf):
        return intervals.bootstrap_ci_corr(
            data[data.columns[1]], data[data.columns[0]],
            lambda x, y: min(1, max(-1, stats.pearsonr(x, y)[0])),
            conf=conf)

    @staticmethod
    def validate(sens, output, expl):
        if output.num_labels != 1:
            raise ValueError('CORR metric only usable for a single target')
        if expl is not None:
            raise ValueError('CORR metric not usable with explanatory features')
        if sens.arity > 2 or output.arity > 2:
            raise ValueError('CORR metric not usable with multivalued features')

    approx_LIMIT_P = 1000
    approx_LIMIT_CI = 1000

    def abs_effect(self):
        return intervals.z_effect(self.stats[0], self.stats[1])

    def __str__(self):
        return 'CORR'


class CondCORR(Metric):
    """
    Conditional Correlation metric.
    """
    dataType = Metric.DATATYPE_CORR

    def compute(self, data, conf, exact=True):
        [ci_low, ci_high, pval] = cond_correlation(data, conf=conf)

        self.stats = pd.DataFrame(columns=['ci_low', 'ci_high', 'pval'])
        self.stats.loc[0] = [ci_low, ci_high, pval]

        # compute mutual information for each sub-group
        for (idx, sub_ct) in enumerate(data):
            self.stats.loc[idx+1] = CORR().compute(sub_ct, conf,
                                                   exact=exact).stats

        return self

    def abs_effect(self):
        (ci_low, ci_high, _) = self.stats.loc[0]
        return intervals.z_effect(ci_low, ci_high)

    @staticmethod
    def approx_stats(data, conf):
        raise NotImplementedError()

    @staticmethod
    def exact_test(data):
        raise NotImplementedError()

    @staticmethod
    def exact_ci(data, conf):
        raise NotImplementedError()

    @staticmethod
    def validate(sens, output, expl):
        if output.num_labels != 1:
            raise ValueError('CondCORR metric only usable for a single target')
        if expl is None:
            raise ValueError('CondCORR metric expects an explanatory feature')
        if sens.arity > 2 or output.arity > 2:
            raise ValueError('CORR metric not usable with multivalued features')

    def __str__(self):
        return 'CondCORR'


def cond_correlation(data, conf=None):
    """
    Compute the conditional correlation of two variables given a third
    Parameters
    ----------
    data :
        data for the `correlation' method, grouped by the conditioning variable
    conf :
        confidence level
    Returns
    -------
    cond_corr :
        the conditional correlation
    """
    weights = [d[5] if np.array(d).shape == (6,) else len(d) for d in data]
    corrs = [correlation(d, conf) for d in data]
    cond_corr = np.average(corrs, axis=0, weights=weights)
    return cond_corr


def correlation(data, conf=None):
    """
    Pearson correlation with confidence intervals.

    Parameters
    ----------
    data :
        data for correlation computation. Either aggregate statistics
        (sum_x, sum_x2, sum_y, sum_y2, sum_xy, n) or complete data

    conf :
        level for confidence intervals (or None)

    Returns
    -------
    ci_low :
        the lower bound of confidence interval

    ci_high :
        the upper bound of confidence interval

    pval :
        the corresponding p value

    References
    ---------
    https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
    """
    # check if we got aggregate statistics or full data
    if np.array(data).shape == (6,):
        sum_x = data[0]
        sum_x2 = data[1]
        sum_y = data[2]
        sum_y2 = data[3]
        sum_xy = data[4]
        n = data[5]
    else:
        if isinstance(data, pd.DataFrame):
            data = data.values
        x = data[:, 0]
        y = data[:, 1]
        sum_x = x.sum()
        sum_x2 = np.dot(x, x)
        sum_y = y.sum()
        sum_y2 = np.dot(y, y)
        sum_xy = np.dot(x, y)
        n = len(x)

    try:
        # correlation coefficient
        corr = (n*sum_xy - sum_x*sum_y) / \
               (sqrt(n*sum_x2 - sum_x**2) * sqrt(n*sum_y2 - sum_y**2))

        if np.isnan(corr):
            raise ValueError()

    except (ZeroDivisionError, ValueError):
        if conf:
            return -1, 1, 1.0
        else:
            return 0

    if conf:
        try:
            # Fisher transform
            fisher = atanh(max(min(corr, 1-1e-6), -1+1e-6))

            std = 1.0/sqrt(n-3)

            pval = tests.z_test(fisher, std)

            ci_fisher = intervals.ci_norm(conf, fisher, std)

            # inverse transform
            ci_low, ci_high = [tanh(ci_fisher[0]), tanh(ci_fisher[1])]

            if np.isnan(ci_low) or np.isnan(ci_high) or np.isnan(pval):
                return -1, 1, 1.0

            return ci_low, ci_high, pval

        except (ZeroDivisionError, ValueError):
            return -1, 1, 1.0
    else:
        return corr
