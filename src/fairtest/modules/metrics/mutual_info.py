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

def algorithm2(data, m):
    """
    Calculates p-values for adaptive investifations according to:
        http://arxiv.org/pdf/1511.03376v2.pdf
    """
    N_SAMPLES = 500
    data = data


    # add noise to each individual cell
    n = data.size
    mean  = 0
    epsilon = 1.0 / m
    sigma = 2.0 / epsilon
    data = data + np.random.laplace(mean, sigma, data.shape)

    # calculate approximate statistic from noisy table
    gstat, _, _, _ = tests.g_test(data)

    #
    # Note the following:
    #   * T[i, .] = data.sum(SUM_j)[i]
    #   * T[., j] = data.sum(SUM_i)[j]
    #
    SUM_i = 0
    SUM_j = 1
    theta = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            theta[i, j] = data.sum(SUM_j)[i] * data.sum(SUM_i)[j] / pow(data.sum() ,2)

    # in case p-val equals zero because the approximation didn' t generate any
    # acceptable samle, set p-value = 1 / (m + 1), making an extremely modest
    # assumption -- instead of setting it to zero, which would be more realistic
    TINY_PVAL = 1.0 / (N_SAMPLES + 1)
    tau = []

    # generate sample to approximate p-val
    for _ in range(N_SAMPLES):
        a  = np.reshape(theta, (theta.shape[0]*theta.shape[1], 1)).diagonal()
        b1 = np.reshape(theta, (theta.shape[0]*theta.shape[1], 1))
        b2 = np.reshape(theta, (theta.shape[0]*theta.shape[1], 1)).transpose()
        multivariate_sigma = a - b1 * b2
        multivariate_mean = np.zeros(multivariate_sigma.shape[1])
        VecA = np.random.multivariate_normal(multivariate_mean, multivariate_sigma)
        A = np.reshape(VecA, data.shape)
        V =  np.random.laplace(mean, sigma, data.shape)
        X = A + V / pow(n, 1.0/2)

        sum1 = 0.0
        sum2 = 0.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                sum1 += X[i, j]**2 / theta[i, j]
            sum2 += X.sum(SUM_j)[i]**2 / theta.sum(SUM_j)[i]

        sum3 = 0.0
        for j in range(data.shape[1]):
            sum3 += X.sum(SUM_i)[j]**2 / theta.sum(SUM_i)[j]

        t = sum1 - sum2 - sum3 + X.sum()**2 / theta.sum()
        tau.append(t)

    # sample values to calculate approximate p-val
    tau = [t for t in tau if t >= gstat]

    if len(tau):
        pval = float(len(tau)) / float(N_SAMPLES)
    else:
        pval = TINY_PVAL

    return original_pval


class NMI(Metric):
    """
    Normalized Mutual Information metric.
    """

    dataType = Metric.DATATYPE_CT

    @staticmethod
    def approx_stats(data, conf, k, m):
        return mutual_info(data, k, m, norm=True, conf=conf)

    @staticmethod
    def exact_test(data):
        return tests.permutation_test_ct(data)

    @staticmethod
    def exact_ci(data, conf, k, m):
        return intervals.bootstrap_ci_ct(data,
                                         lambda s: mutual_info(s, k, m, norm=True),
                                         conf=conf)

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


class CondNMI(Metric):
    """
    Conditional Mutual Information metric.
    """
    dataType = Metric.DATATYPE_CT

    def compute(self, data, conf, k, m, exact=True):
        if exact:
            pval = tests.permutation_test_ct_cond(
                data, lambda ct: abs(cond_mutual_info(ct, k, m)))

            ci_low, ci_high = intervals.bootstrap_ci_ct_cond(
                data, lambda ct: abs(cond_mutual_info(ct, k, m)), conf=conf)
        else:
            [ci_low, ci_high, pval] = cond_mutual_info(data, k, m, conf=conf)

        self.stats = pd.DataFrame(columns=['ci_low', 'ci_high', 'pval'])
        self.stats.loc[0] = [ci_low, ci_high, pval]

        # compute mutual information for each sub-group
        for (idx, sub_ct) in enumerate(data):
            self.stats.loc[idx+1] = NMI().compute(sub_ct, conf,
                                                  exact=exact).stats

        return self

    def abs_effect(self):
        (ci_low, _, _) = self.stats.loc[0]
        return ci_low

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
            raise ValueError('CondNMI metric only usable for a single target')
        if expl is None:
            raise ValueError('CondNMI metric expects an explanatory feature')
        if not expl.arity:
            raise ValueError('CondNMI metric expects a categorical explanatory'
                             ' feature')

    def __str__(self):
        return 'CondNMI'


def mutual_info(data, k, m, norm=True, conf=None):
    """
    mutual information with or without normalization and confidence intervals.

    Parameters
    ----------
    data :
        a contingency table

    norm :
        whether the MI should be normalized

    conf :
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
        if conf is not None:
            return 0, 1, 1.0
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
    if conf is None:
        return mi

    gstat, pval, dof, _ = tests.g_test(data)

    ci_low, ci_high = intervals.ci_mi(gstat, dof, data_size, conf, k, m)


    if pval > 1-conf:
        ci_low = 0

    if norm:
        ci_low /= min(h_x, h_y)
        ci_high /= min(h_x, h_y)

    ci_low = max(ci_low, 0)
    ci_high = min(ci_high, 1)

    # override everything and use paper's algorithm
    if k and m > 1:
        #pval = algorithm2(data, 1)
        pval = algorithm2(data, m)
        #pval = algorithm2(data, m+5)
        #pval = algorithm2(data, m+10)
        #print "--"


    return ci_low, ci_high, pval


def cond_mutual_info(data, k, m, norm=True, conf=None):
    """
    Compute the conditional mutual information of two variables given a third
    Parameters
    ----------
    data :
        A 3-dimensional table. This method computes the mutual
        information between the first and second dimensions, given the third.
    norm :
        whether the MI should be normalized
    Returns
    -------
    cond_mi :
        the conditional mutual information
    References
    ----------
    https://en.wikipedia.org/wiki/Conditional_mutual_information
    """
    weights = [d.sum() for d in data]
    mis = [mutual_info(d, k, m, norm, conf) for d in data]
    cond_mi = np.average(mis, axis=0, weights=weights)
    return cond_mi
