"""
Binary Ratio and Difference metrics.
"""

import fairtest.modules.statistics.hypothesis_test as tests
import fairtest.modules.statistics.confidence_interval as intervals
from .metric import Metric
import pandas as pd
import numpy as np
from math import sqrt, log, exp


class DIFF(Metric):
    """
    Difference metric.
    """
    dataType = Metric.DATATYPE_CT

    @staticmethod
    def approx_stats(data, level):
        return difference(data, ci_level=level)

    @staticmethod
    def exact_test(data):
        return tests.permutation_test_ct(data)

    @staticmethod
    def exact_ci(data, level):
        return intervals.bootstrap_ci_ct(data,
                                         lambda s: difference(s, ci_level=None),
                                         ci_level=level)

    @staticmethod
    def validate(sens, output, expl):
        if output.num_labels != 1:
            raise ValueError('DIFF metric only usable for a single target')
        if expl is not None:
            raise ValueError('DIFF metric not usable with explanatory features')
        if not sens.arity == 2 or not output.arity == 2:
            raise ValueError('DIFF metric only usable with binary features')

    approx_LIMIT_P = 1000
    approx_LIMIT_CI = 1000

    def abs_effect(self):
        return intervals.z_effect(self.stats[0], self.stats[1])

    def __str__(self):
        return 'DIFF'


class CondDIFF(Metric):
    """
    Conditional Difference metric.
    """
    dataType = Metric.DATATYPE_CT

    def compute(self, data, level, exact=True):

        pval = \
            tests.permutation_test_ct_cond(data,
                                           lambda ct: abs(cond_difference(ct)))

        ci_low, ci_high = intervals.bootstrap_ci_ct_cond(data, cond_difference,
                                                         ci_level=level)

        self.stats = pd.DataFrame(columns=['ci_low', 'ci_high', 'pval'])
        self.stats.loc[0] = [ci_low, ci_high, pval]

        # compute difference for each sub-group
        for (idx, sub_ct) in enumerate(data):
            self.stats.loc[idx+1] = DIFF().compute(sub_ct, level,
                                                   exact=exact).stats

        return self

    def abs_effect(self):
        (ci_low, ci_high, _) = self.stats.loc[0]
        return intervals.z_effect(ci_low, ci_high)

    @staticmethod
    def approx_stats(data, level):
        raise NotImplementedError()

    @staticmethod
    def exact_test(data):
        raise NotImplementedError()

    @staticmethod
    def exact_ci(data, level):
        raise NotImplementedError()

    @staticmethod
    def validate(sens, output, expl):
        if output.num_labels != 1:
            raise ValueError('CondDIFF metric only usable for a single target')
        if expl is None:
            raise ValueError('CondDIFF metric expects an explanatory feature')
        if not expl.arity:
            raise ValueError('CondDIFF metric expects a categorical explanatory'
                             ' feature')
        if not sens.arity == 2 or not output.arity == 2:
            raise ValueError('CondDIFF metric only usable with binary features')

    def __str__(self):
        return 'COND_DIFF'


class RATIO(Metric):
    """
    Ratio metric.
    """
    dataType = Metric.DATATYPE_CT

    @staticmethod
    def approx_stats(data, level):
        return ratio(data, ci_level=level)

    @staticmethod
    def exact_test(data):
        return tests.permutation_test_ct(data)

    @staticmethod
    def exact_ci(data, level):
        return intervals.bootstrap_ci_ct(data,
                                         lambda s: ratio(s, ci_level=None),
                                         ci_level=level)

    @staticmethod
    def validate(sens, output, expl):
        if output.num_labels != 1:
            raise ValueError('RATIO metric only usable for a single target')
        if expl is not None:
            raise ValueError('RATIO metric not usable with explanatory '
                             'features')
        if not sens.arity == 2 or not output.arity == 2:
            raise ValueError('RATIO metric only usable with binary features')

    approx_LIMIT_P = 1000
    approx_LIMIT_CI = 1000

    def abs_effect(self):
        return intervals.z_effect(log(self.stats[0]), log(self.stats[1]))

    def __str__(self):
        return 'RATIO'


def difference(data, ci_level=None):
    """
    Difference metric, possibly with confidence intervals.

    Parameters
    ----------
    data :
        2x2 contingency table

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
    """

    # check if data is degenerate
    if data.shape == (1, 1) or data.shape == (1, 2) or data.shape == (2, 1):
        if ci_level:
            return 0, 0, 1.0
        else:
            return 0, 1.0

    assert data.shape == (2, 2)
    if isinstance(data, pd.DataFrame):
        data = data.values

    # transform contingency table into probability table
    tot = np.sum(data, axis=0)
    probas = np.array(data, dtype=float)/tot

    # Difference metric
    diff = probas[1, 0]-probas[1, 1]

    #
    # confidence levels as in Ruggieri et al. '10
    #
    if ci_level:
        # contingency table values
        n1 = tot[0]
        n2 = tot[1]
        r1 = data[1][0]
        r2 = data[1][1]

        # proba of hired
        p1 = (1.0*r1)/n1
        p2 = (1.0*r2)/n2

        # standard deviations
        sigma_diff = sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
        pval = tests.z_test(diff, sigma_diff)
        # confidence intervals
        ci_low, ci_high = intervals.ci_norm(ci_level, diff, sigma_diff)

        ci_low = max(ci_low, -1)
        ci_high = min(ci_high, 1)

        return ci_low, ci_high, pval
    else:
        return diff


def cond_difference(data):
    """
    Conditional difference.

    Parameters
    ----------
    data :
        array of 2x2 contingency tables

    Returns
    -------
    cond_diff :
        conditional difference
    """
    weights = [d.sum() for d in data]
    diffs = [difference(d, ci_level=None) for d in data]
    cond_diff = np.average(diffs, axis=None, weights=weights)
    return cond_diff


def ratio(data, ci_level=None):
    """
    Ratio metric, possibly with confidence intervals

    Parameters
    ----------
    data :
        2x2 contingency table

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
    """
    # check if data is degenerate
    if data.shape == (1, 1) or data.shape == (1, 2) or data.shape == (2, 1):
        if ci_level:
            return 1, 1, 1.0
        else:
            return 1, 1.0

    assert data.shape == (2, 2)
    if isinstance(data, pd.DataFrame):
        data = data.values

    # data smoothing
    data = data.copy()
    data += 1

    # transform contingency table into probability table
    tot = np.sum(data, axis=0)
    probas = np.array(data, dtype=float)/tot

    # ratio metric
    ratio_stat = probas[1, 0]/probas[1, 1]

    # confidence levels as in Ruggieri et al. '10
    if ci_level:
        # contingency table values
        n1 = tot[0]
        n2 = tot[1]
        r1 = data[1][0]
        r2 = data[1][1]

        # standard deviations
        sigma_log_ratio = sqrt(1.0/r1+1.0/r2-1.0/n1-1.0/n2)
        pval = tests.z_test(log(ratio_stat), sigma_log_ratio)

        # confidence intervals
        ci_log_ratio = intervals.ci_norm(ci_level, log(ratio_stat),
                                         sigma_log_ratio)
        ci_low, ci_high = exp(ci_log_ratio[0]), exp(ci_log_ratio[1])

        return ci_low, ci_high, pval
    else:
        return ratio_stat
