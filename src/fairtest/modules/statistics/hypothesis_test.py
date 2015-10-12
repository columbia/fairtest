"""
Statistical Hypothesis Tests
"""

import pandas as pd
import scipy.stats as stats
import numpy as np
import sklearn.metrics as metrics
import rpy2.robjects as ro
from collections import Counter


def g_test(data, correction=False):
    """
    G-test (likelihood ratio test).

    Parameters
    ----------
    data :
        the contingency table

    correction :
        whether to apply continuity corrections

    Returns
    -------
    g :
        the test statistic
    p :
        the p-value
    df:
        the number of degrees of freedom
    expected:
        the expected frequencies

    References
    ----------
    https://en.wikipedia.org/wiki/G-test
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    # remove zero rows/columns
    data = data[~np.all(data == 0, axis=1)]
    data = data[:, ~np.all(data == 0, axis=0)]

    if data.sum() == 0:
        return 0, 1.0, 1, None

    return stats.chi2_contingency(data, correction=correction,
                                  lambda_="log-likelihood")


def z_test(stat, sigma):
    """
    Z-test.

    Parameters
    ----------
    stat :
        the asymptotically normal statistic

    sigma :
        the statistic's standard deviation

    Returns
    -------
    pval :
        the p-value

    References
    ----------
    https://en.wikipedia.org/wiki/Z-test
    """
    z = stat/sigma
    pval = 2*stats.norm.sf(abs(z))
    return pval


def permutation_test_ct2(data, num_samples=10000):
    """
    Monte-Carlo permutation test for a 2-way contingency table

    Parameters
    ----------
    data :
        the contingency table

    num_samples :
        the number of random permutations to perform

    Returns
    -------
    pval :
        the p-value

    References
    ----------
    https://en.wikipedia.org/wiki/Resampling_(statistics)
    """
    if isinstance(data, pd.DataFrame):
        data = np.array(data)

    dim = data.shape
    data_x = []
    data_y = []

    for x in range(0, dim[0]):
        for y in range(0, dim[1]):
            data_x += [x]*data[x, y]
            data_y += [y]*data[x, y]

    stat_0 = metrics.mutual_info_score(data_x, data_y)

    k = 0
    for _ in range(num_samples):
        np.random.shuffle(data_x)
        mi = metrics.mutual_info_score(data_x, data_y)
        k += stat_0 < mi

    pval = (1.0*k) / num_samples
    return max(pval, 1.0/num_samples)


def permutation_test_ct(data, num_samples=100000):
    """
    Monte-Carlo permutation test for a contingency table.
    Uses the chi square statistic.

    Parameters
    ----------
    data :
        the contingency table

    num_samples :
        the number of random permutations to perform

    Returns
    -------
    pval :
        the p-value

    Notes
    -----
    Uses the R 'coin' package that can directly handle contingency
    tables instead of having to convert into arrays x,y

    References
    ----------
    https://en.wikipedia.org/wiki/Resampling_(statistics)
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    data = data[data.columns[(data != 0).any()]]
    data = data[(data.T != 0).any()]

    # print 'permutation test of size {}'.format(data.sum())

    data = np.array(data, dtype='int')
    if len(data.shape) != 2:
        return 1
    if data.shape[0] < 2 or data.shape[1] < 2:
        return 1

    ro.globalenv['ct'] = data
    pval = ro.r('chisq.test(ct, simulate.p.value = TRUE, B = {})$p.value'.
                format(num_samples))[0]
    return max(pval, 1.0/num_samples)


def permutation_test_corr(x, y, num_samples=10000):
    """
    Monte-Carlo permutation test for correlation

    Parameters
    ----------
    x :
        Values for the first dimension

    y :
        Values for the second dimension

    num_samples :
        the number of random permutations to perform

    Returns
    -------
    pval :
        the p-value

    References
    ----------
    https://en.wikipedia.org/wiki/Resampling_(statistics)
    """
    x = np.array(x, dtype='float')
    y = np.array(y, dtype='float')

    obs_0, _ = stats.pearsonr(x, y)
    k = 0
    z = np.concatenate([x, y])
    for _ in range(num_samples):
        np.random.shuffle(z)
        k += abs(obs_0) < abs(stats.pearsonr(z[:len(x)], z[len(x):])[0])
    pval = (1.0*k) / num_samples
    return max(pval, 1.0/num_samples)


def permutation_test_ct_cond(data, stat, num_samples=10000):
    """
    Monte-Carlo permutation test for a 3-way contingency table

    Parameters
    ----------
    data :
        the contingency table

    stat :
        the statistic to apply to each permuted table

    num_samples :
        the number of random permutations to perform

    Returns
    -------
    pval :
        the p-value

    References
    ----------
    https://en.wikipedia.org/wiki/Resampling_(statistics)
    """
    data = np.array([ct.values if isinstance(ct, pd.DataFrame)
                     else ct for ct in data])

    dim = data.shape
    data_x = [[] for _ in data]
    data_y = [[] for _ in data]
    stat_0 = stat(data)

    # convert the table into an array of data points
    for e in range(0, dim[0]):
        for x in range(0, dim[1]):
            for y in range(0, dim[2]):
                data_x[e] += [x]*data[e, x, y]
                data_y[e] += [y]*data[e, x, y]

    k = 0
    for _ in range(num_samples):
        temp = np.zeros(dim)
        # permute each sub-group individually
        for e in range(0, dim[0]):
            np.random.shuffle(data_x[e])

            # re-form a contingency table
            counter = Counter(zip(data_x[e], data_y[e]))
            for x in range(0, dim[1]):
                for y in range(0, dim[2]):
                    temp[e, x, y] = counter.get((x, y), 0)

        temp_stat = stat(temp)
        k += stat_0 < temp_stat

    pval = (1.0*k) / num_samples
    return max(pval, 1.0/num_samples)
