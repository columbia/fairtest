"""
Methods for computing confidence intervals.
"""

import scipy.special as special
import numpy as np
import pandas as pd
import scipy.stats as stats


def z_effect(ci_low, ci_high):
    """
    Compute an effect score for a z-score.

    Parameters
    ----------
    ci_low :
        Lower bound of the confidence interval

    ci_high :
        Upper bound of the confidence interval

    Returns
    -------
    score :
        An effect score for a Z-score

    Notes
    ----
    This is the absolute value of the lower bound of the confidence interval,
    or zero if the interval contains zero.
    """
    if np.isnan(ci_low) or np.isnan(ci_high):
        return 0
    return 0 if (ci_low * ci_high < 0) else min(abs(ci_low), abs(ci_high))


def ci_mi(g, dof, n, conf):
    """
    Compute confidence interval for mutual information from the chi-squared
    distribution.

    Parameters
    ----------
    g :
        the G-test score

    dof :
        the number of degrees of freedom

    n :
        the size of the data sample

    conf :
        the confidence level

    Returns
    -------
    ci_low :
        The lower level of the confidence interval for MI
    ci_high :
        The upper level of the confidence interval for MI

    References
    ----------
    Smithson, M. (Ed.). (2003). Confidence Intervals. (07/140). Thousand Oaks,
    CA: SAGE Publications, Inc. doi: http://dx.doi.org/10.4135/9781412983761

    https://en.wikipedia.org/wiki/G-test
    """
    p_low = 1-(1-conf)/2
    p_high = (1-conf)/2

    g_low = special.chndtrinc(g, dof, p_low)
    g_high = special.chndtrinc(g, dof, p_high)
    ci_low, ci_high = ((g_low+dof)/(2.0*n), (g_high+dof)/(2.0*n))
    return ci_low, ci_high


def ci_norm(conf, stat, sigma):
    """
    Confidence interval for a normal approximation.

    Parameters
    ----------
    conf :
        the confidence level

    stat :
        the asymptotically normal statistic

    sigma :
        the standard deviation of the statistic

    Returns
    -------
    ci_low :
        The lower level of the confidence interval
    ci_high :
        The upper level of the confidence interval
    """

    ci_low, ci_high = stats.norm.interval(conf, loc=stat, scale=sigma)
    return ci_low, ci_high


def bootstrap_ci_ct(data, stat, num_samples=10000, conf=0.95):
    """
    Bootstrap confidence interval computation on a contingency table

    Parameters
    ----------
    data :
        Contingency table collected from independent samples

    stat :
        Statistic to bootstrap. Takes a contingency table as argument

    num_samples :
        Number of bootstrap samples to generate

    conf :
        Confidence level for the interval

    Returns
    -------
    ci_low :
        The lower level of the confidence interval
    ci_high :
        The upper level of the confidence interval
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    dim = data.shape
    data = data.flatten()
    data += 1
    n = data.sum()

    # print 'Bootstrap on data of size {}'.format(n)
    probas = (1.0*data)/n

    # Obtain `num_samples' random samples of `n' multinomial values, sampled
    # with replacement from {0, 1, ..., n-1}. For each sample, rebuild a
    # contingency table and compute the stat.
    temp = np.random.multinomial(n, probas, size=num_samples)
    bs_stats = [row.reshape(dim) for row in temp]
    bs_stats = [stat(ct) for ct in bs_stats]

    alpha = 1-conf
    ci_low = np.percentile(bs_stats, 100*alpha/2)
    ci_high = np.percentile(bs_stats, 100*(1-alpha/2))

    return ci_low, ci_high


def bootstrap_ci_corr(x, y, stat, num_samples=10000, conf=0.95):
    """
    Bootstrap confidence interval computation for correlation

    Parameters
    ----------
    x :
        First dimension of the data

    y :
        Second dimension of the data

    stat :
        Statistic to bootstrap. Takes a two-dimensional array as input

    num_samples :
        Number of bootstrap samples to generate

    conf :
        Confidence level for the interval

    Returns
    -------
    ci_low :
        The lower level of the confidence interval
    ci_high :
        The upper level of the confidence interval
    """
    data = np.array(zip(x, y))
    n = len(data)
    idxs = np.random.randint(0, n, (num_samples, n))
    samples = [data[idx] for idx in idxs]
    bs_stats = [stat(sample[:, 0], sample[:, 1]) for sample in samples]
    alpha = 1-conf
    ci_low = np.percentile(bs_stats, 100*alpha/2)
    ci_high = np.percentile(bs_stats, 100*(1-alpha/2))

    return ci_low, ci_high


def bootstrap_ci_ct_cond(data, stat, num_samples=10000, conf=0.95):
    """
    Bootstrap confidence interval computation on a 3-way contingency table

    Parameters
    ----------
    data :
        Contingency table collected from independent samples

    stat :
        Statistic to bootstrap. Takes a 3-way contingency table as argument

    num_samples :
        Number of bootstrap samples to generate

    conf :
        Confidence level for the interval

    Returns
    -------
    ci_low :
        The lower level of the confidence interval
    ci_high :
        The upper level of the confidence interval
    """
    data = np.array([ct.values if isinstance(ct, pd.DataFrame)
                     else ct for ct in data])

    dim = data.shape
    data = [ct.flatten()+1 for ct in data]

    probas = [(1.0*ct)/ct.sum() for ct in data]

    # Resample for each explanatory group
    temp = np.dstack([np.random.multinomial(data[i].sum(),
                                            probas[i],
                                            size=num_samples)
                      for i in range(dim[0])])

    bs_stats = [row.T.reshape(dim) for row in temp]
    bs_stats = [stat(ct) for ct in bs_stats]

    alpha = 1-conf
    ci_low = np.percentile(bs_stats, 100*alpha/2)
    ci_high = np.percentile(bs_stats, 100*(1-alpha/2))

    return ci_low, ci_high
