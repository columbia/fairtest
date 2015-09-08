# -*- coding: utf-8 -*-
"""
Module that contains various statistical measures
"""
import scipy.stats as stats
import scipy.special as special
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from math import sqrt, log, exp, atanh, tanh
import scikits.bootstrap as bs
import rpy2.robjects as ro


def z_effect(ci_low, ci_high):
    """
    Compute an effect score for a Z-score

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
    if the interval does not contain zero.
    """
    return 0 if (ci_low * ci_high < 0) else min(abs(ci_low), abs(ci_high))


class Measure(object):
    """
    Class representing a fairness measure
    """
    # Types of measures
    DATATYPE_CT = 'ct'      # Measures on a contingency table
    DATATYPE_CORR = 'corr'  # Correlation measures
    DATATYPE_REG = 'reg'    # Regression measures

    def __init__(self, ci_level=0.95):
        self.ci_level = ci_level
        self.stats = None
        self.exact_ci = None


class NMI(Measure):
    """
    Normalized Mutual Information measure
    """
    dataType = Measure.DATATYPE_CT

    def __init__(self, ci_level=None):
        Measure.__init__(self, ci_level)
        self.data = None

    def compute(self, data, approx=True, adj_ci_level=None):
        self.data = data
        ci_level = self.ci_level if adj_ci_level is None else adj_ci_level

        if approx:
            self.stats = mutual_info(data, norm=True, ci_level=ci_level)
        else:
            N = np.array(data).sum()
            if N < 1000:
                pval = permutation_test_ct(data)
            else:
                _, pval, _, _ = G_test(data)

            if ci_level:
                if N < 1000:
                    ci_low, ci_high = \
                        bootstrap_ci_ct(data,
                                        lambda x: mutual_info(x, norm=True,
                                                              ci_level=None,
                                                              p=False)[0],
                                        ci_level=ci_level)
                else:
                    ci_low, ci_high, _ = mutual_info(data,
                                                     norm=True,
                                                     ci_level=ci_level)

                if pval > 1-ci_level:
                    ci_low = 0

                self.stats = (ci_low, ci_high, pval)
            else:
                mi, _ = mutual_info(data, norm=True, ci_level=None)
                self.stats = mi, pval

        return self

    def abs_effect(self):
        return self.stats[0]

    def __str__(self):
        return 'NMI(confidence={})'.format(self.ci_level)

    def __copy__(self):
        return NMI(self.ci_level)


class CondNMI(Measure):
    """
    Conditional Mutual Information Measure
    """
    dataType = Measure.DATATYPE_CT

    def __init__(self, ci_level=None):
        Measure.__init__(self, ci_level)
        self.data = None

    def compute(self, data, approx=True, adj_ci_level=None):
        self.data = data
        ci_level = self.ci_level if adj_ci_level is None else adj_ci_level

        if approx:
            self.stats = cond_mutual_info(data, norm=True, ci_level=ci_level)
        else:
            N = np.max([np.array(ct).sum() for ct in np.array(data)])
            if N < 1000:
                pval = permutation_test_ct_cond(data)
            else:
                _, pval, _ = G_test_cond(data)

            if ci_level:
                if N < 1000:
                    ci_low, ci_high = \
                        bootstrap_ci_ct_cond(data,
                                             lambda x:
                                             cond_mutual_info(x, norm=True,
                                                              ci_level=None,
                                                              p=False)[0],
                                             ci_level=ci_level)
                else:
                    ci_low, ci_high, _ = cond_mutual_info(data, norm=True,
                                                          ci_level=ci_level)

                if pval > 1-ci_level:
                    ci_low = 0

                self.stats = (ci_low, ci_high, pval)
            else:
                mi, _ = cond_mutual_info(data, norm=True, ci_level=None)
                self.stats = mi, pval

        return self

    def abs_effect(self):
        return self.stats[0]

    def __copy__(self):
        return CondNMI(self.ci_level)

    def __str__(self):
        return 'Conditional NMI(confidence={})'.format(self.ci_level)


class CORR(Measure):
    """
    Pearson Correlation measure
    """
    dataType = Measure.DATATYPE_CORR

    def compute(self, corr_stats, data=None, approx=True, adj_ci_level=None):
        ci_level = self.ci_level if adj_ci_level is None else adj_ci_level

        if approx:
            self.stats = correlation(corr_stats, ci_level=ci_level)
        else:
            x = data[data.columns[1]]
            y = data[data.columns[0]]
            if len(x) < 1000:
                pval = permutation_test_corr(x, y)
            else:
                _, _, pval = correlation(corr_stats, ci_level)
            if ci_level:
                if len(x) < 1000:
                    ci_low, ci_high = \
                        bootstrap_ci_corr(x, y,
                                          lambda x,y: stats.pearsonr(x, y)[0],
                                          ci_level=ci_level)
                else:
                    ci_low, ci_high, _ = correlation(corr_stats, ci_level)
                if pval > 1-ci_level:
                    if ci_low < 0 and ci_high < 0:
                        ci_high = 0
                    elif ci_low > 0 and ci_high > 0:
                        ci_low = 0
                self.exact_ci = True
                self.stats = (ci_low, ci_high, pval)
            else:
                corr = stats.pearsonr(x, y)[0]
                self.stats = (corr, pval)

        return self

    def abs_effect(self):
        if self.ci_level:
            (ci_low, ci_high, p) = self.stats
            return z_effect(ci_low, ci_high)
        else:
            return abs(self.stats[0])

    def __copy__(self):
        return CORR(self.ci_level)

    def __str__(self):
        return 'Correlation(confidence={})'.format(self.ci_level)


class DIFF(Measure):
    """
    Difference measure
    """
    dataType = Measure.DATATYPE_CT

    def compute(self, data, approx=True, adj_ci_level=None):
        ci_level = self.ci_level if adj_ci_level is None else adj_ci_level

        if approx:
            self.stats = difference(data, ci_level=ci_level)
        else:
            N = np.array(data).sum()
            if N < 1000:
                pval = permutation_test_ct(data)
            else:
                _, pval, _, _ = G_test(data)

            if ci_level:
                ci_low, ci_high = \
                        bootstrap_ci_ct(data,
                                        lambda x: difference(x, ci_level=None),
                                        ci_level=ci_level)

                if pval > 1-ci_level:
                    if ci_low < 0 and ci_high < 0:
                        ci_high = 0
                    elif ci_low > 0 and ci_high > 0:
                        ci_low = 0

                self.exact_ci = True
                self.stats = (ci_low, ci_high, pval)
            else:
                self.stats = difference(data, ci_level=ci_level)

        return self

    def abs_effect(self):
        if self.ci_level:
            (ci_low, ci_high, p) = self.stats
            return z_effect(ci_low, ci_high)
        else:
            return abs(self.stats[0])

    def __copy__(self):
        return DIFF(self.ci_level)

    def __str__(self):
        return 'Difference(confidence={})'.format(self.ci_level)


class RATIO(Measure):
    """
    Ratio measure
    """
    dataType = Measure.DATATYPE_CT

    def compute(self, data, approx=True, adj_ci_level=None):
        ci_level = self.ci_level if adj_ci_level is None else adj_ci_level

        if approx:
            self.stats = ratio(data, ci_level=ci_level)
        else:
            N = np.array(data).sum()
            if N < 1000:
                pval = permutation_test_ct(data)
            else:
                _, pval, _, _ = G_test(data)

            if ci_level:
                ci_low, ci_high = \
                        bootstrap_ci_ct(data,
                                        lambda x: ratio(x, ci_level=None),
                                        ci_level=ci_level)
                if pval > 1-ci_level:
                    if ci_low < 1 and ci_high < 1:
                        ci_high = 1
                    elif ci_low > 1 and ci_high > 1:
                        ci_low = 1

                self.exact_ci = True
                self.stats = (ci_low, ci_high, pval)
            else:
                self.stats = ratio(data, ci_level=ci_level)

        return self

    def abs_effect(self):
        if self.ci_level:
            (ci_low, ci_high, p) = self.stats
            return z_effect(log(ci_low), log(ci_high))
        else:
            return abs(log(self.stats[0]))

    def __copy__(self):
        return RATIO(self.ci_level)

    def __str__(self):
        return 'Regression(confidence={})'.format(self.ci_level)


class REGRESSION(Measure):
    """
    Regression measure
    """
    dataType = Measure.DATATYPE_REG

    def __init__(self, ci_level=None, topK=10):
        Measure.__init__(self, ci_level)
        self.topK = topK
        self.type = None

    def __copy__(self):
        return REGRESSION(self.ci_level, self.topK)

    def compute(self, data, approx=False, adj_ci_level=None):
        ci_level = self.ci_level if adj_ci_level is None else adj_ci_level

        # regression not yet trained
        if not self.type:
            y = data[data.columns[-1]]
            X = data[data.columns[0:-1]]

            # print 'Regressing from {}...{} to {}'.\
            #        format(data.columns[0], data.columns[-2], data.columns[-1])

            reg = LogisticRegression()
            reg.fit(X, y)
            y_pred = reg.predict(X)

            # approximate the standard errors for all regression coefficients
            mse = np.mean((y - y_pred.T)**2)
            var_est = mse * np.diag(np.linalg.pinv(np.dot(X.T, X)))
            SE_est = np.sqrt(var_est)
            coeffs = reg.coef_[0].tolist()

            # compute confidence intervals and p-values for all coefficients
            results = pd.DataFrame(coeffs, columns=['coeff'])
            results['std err'] = SE_est
            results['z'] = abs(results['coeff']/results['std err'])
            results['p-value'] = 2*stats.norm.sf(results['z'])

            if not ci_level:
                results['effect'] = map(lambda c: abs(c), results['coeff'])
                sorted_results = results.sort(columns=['effect'],
                                              ascending=False)
                self.stats = sorted_results[['coeff',
                                             'p-value']].head(self.topK)
                self.type = "Regression"
                return self

            ci_s = stats.norm.interval(ci_level,
                                       loc=results['coeff'],
                                       scale=results['std err'])
            results['conf low'] = ci_s[0]
            results['conf high'] = ci_s[1]

            # compute a standardized effect size
            # and return the topK coefficients
            results['effect'] = \
                    map(lambda (ci_low, ci_high): z_effect(ci_low, ci_high),
                        zip(results['conf low'], results['conf high']))
            sorted_results = results.sort(columns=['effect'], ascending=False)
            self.stats = sorted_results[['conf low', 'conf high', 'p-value']].\
                    head(self.topK)
            self.type = "Regression"
            return self
        else:
            # model was already trained, get the top labels
            top_labels = self.stats.index

            for idx in top_labels:
                self.stats.loc[idx] = \
                    mutual_info(pd.crosstab(data[data.columns[idx]],
                                            data[data.columns[-1]]),
                                norm=True, ci_level=ci_level)
            self.type = "MI"
            return self

    def abs_effect(self):
        if self.type == "Regression":
            if self.ci_level:
                effects = np.array(map(
                    lambda (ci_low, ci_high, pval): z_effect(ci_low, ci_high),
                    self.stats.values))
            else:
                effects = np.array(map(lambda (coeff, pval): abs(coeff),
                                       self.stats.values))

            non_nan = effects[~np.isnan(effects)]
            if len(non_nan) == 0:
                return -1
            else:
                return np.sum(non_nan)/len(effects)
        else:
            return np.mean(map(lambda stat: stat[0], self.stats.values))

    def __str__(self):
        return 'Regression(confidence={}, topK={})'.\
                format(self.ci_level, self.topK)


def mutual_info(data, norm=False, ci_level=None, p=True):
    """
    mutual information with or without normalization and confidence intervals

    Parameters
    ----------
    data :
        the contingency table to compute the MI of

    norm :
        whether the MI should be normalized

    ci_level :
        level for confidence intervals (or None)

    p :
        whether a p-value should be computed (for efficiency reasons when
        bootstrapping)

    Returns
    -------
    max :
        upper bound of confidence interval

    min :
        lower bound of confidence interval

    pval :
        the corresponding p-value

    References
    ----------
    https://en.wikipedia.org/wiki/Mutual_information
    """
    assert not p or ci_level

    if isinstance(data, pd.DataFrame):
        data = data.values

    if p:
        G, pval, df, _ = G_test(data)
    else:
        pval = 0

    # data smoothing
    data = data.copy()
    data += 1

    shape = data.shape
    if shape[0] < 2 or shape[1] < 2:
        if ci_level:
            return 0, 1, pval
        else:
            return 0, pval

    # row/column sums
    sum_x = np.sum(data, axis=1)
    sum_y = np.sum(data, axis=0)

    # joint probabilities
    N = np.array(data).sum()

    # entropies
    hx = stats.entropy(sum_x)
    hy = stats.entropy(sum_y)
    hxy = stats.entropy(data.flatten())

    mi = -hxy + hx + hy

    # normalized mutual info
    if norm:
        if (hx == 0) or (hy == 0) or (mi == 0):
            mi = 0
        else:
            mi = mi/min(hx, hy)

    # no confidence levels, return single measure
    if not ci_level:
        return mi, pval

    # Compute confidence interval from chi-squared distribution
    p_low = 1-(1-ci_level)/2
    p_high = (1-ci_level)/2
    G_low = special.chndtrinc(G, df, p_low)
    G_high = special.chndtrinc(G, df, p_high)
    ci = ((G_low+df)/(2.0*N), (G_high+df)/(2.0*N))

    if pval > 1-ci_level:
        ci = (0, ci[1])

    if norm:
        ci = map(lambda x: x/min(hx, hy), ci)

    return max(ci[0], 0), min(ci[1], 1), pval


def statistical_parity(data):
    """
    statistical parity measure

    Parameters
    ----------
    data :
        the contingency table of shape (?, 2)

    Returns
    -------
    sp :
        the statistical parity measure
    """
    assert data.shape[1] == 2

    # transform contingency table into probability table
    tot = np.sum(data, axis=0)
    data = np.array(np.array(data, dtype=float)/tot, dtype='float')

    sp = 0.5*sum(abs(data[:, 0]-data[:, 1]))
    return sp


def difference(data, ci_level=0.95):
    """
    Difference measure, possibly with confidence intervals

    Parameters
    ----------
    data :
        contingency table

    ci_level :
        level for confidence intervals (or None)

    Returns
    -------
    max :
        upper bound of confidence interval

    min :
        lower bound of confidence interval

    pval :
        the corresponding p-value
    """

    # check if data is degenerate
    if data.shape == (1,1) or data.shape == (1,2) or data.shape == (2,1):
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

    # Difference measure
    diff = probas[1, 0]-probas[1, 1]

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
    z = diff/sigma_diff
    pval = 2*stats.norm.sf(abs(z))

    #
    # confidence levels as in Ruggieri et al. '10
    #
    if ci_level:
        # confidence intervals
        ci_diff = stats.norm.interval(ci_level, loc=diff, scale=sigma_diff)

        return max(ci_diff[0], -1), min(ci_diff[1], 1), pval
    else:
        return diff, pval


def ratio(data, ci_level=0.95):
    """
    Ratio measure, possibly with confidence intervals

    Parameters
    ----------
    data :
        contingency table

    ci_level :
        level for confidence intervals (or None)

    Returns
    -------
    max :
        upper bound of confidence interval

    min :
        lower bound of confidence interval

    pval :
        the corresponding p-value
    """
    # check if data is degenerate
    if data.shape == (1,1) or data.shape == (1,2) or data.shape == (2,1):
        if ci_level:
            return 1, 1, 1.0
        else:
            return 1, 1.0

    assert data.shape == (2, 2)
    if isinstance(data, pd.DataFrame):
        data = data.values

    # data smoothing
    data = data.copy()
    data += 5

    # transform contingency table into probability table
    tot = np.sum(data, axis=0)
    probas = np.array(data, dtype=float)/tot

    # Slift measures
    ratio = probas[1, 0]/probas[1, 1]

    # contingency table values
    n1 = tot[0]
    n2 = tot[1]
    r1 = data[1][0]
    r2 = data[1][1]

    # standard deviations
    sigma_log_ratio = sqrt(1.0/r1+1.0/r2-1.0/n1-1.0/n2)
    z = log(ratio)/sigma_log_ratio
    pval = 2*stats.norm.sf(abs(z))

    # confidence levels as in Ruggieri et al. '10
    if ci_level:
        # confidence intervals
        ci_log_ratio = stats.norm.interval(ci_level,
                                           loc=log(ratio),
                                           scale=sigma_log_ratio)

        return exp(ci_log_ratio[0]), exp(ci_log_ratio[1]), pval
    else:
        return ratio, pval


def G_test(data, correction=False):
    """
    Computes the G_test

    Parameters
    ----------
    data :
        contingency table to test

    correction :
        whether to apply continuity corrections

    Returns
    -------
    chi2_contingency :
        log-likelihood goodness of fit gtest

    References
    ----------
    https://en.wikipedia.org/wiki/G-test
    """
    # remove all-zero columns/rows
    if isinstance(data, pd.DataFrame):
        data = data.values

    data = data[~np.all(data == 0, axis=1)]
    data = data[:, ~np.all(data == 0, axis=0)]

    if data.sum() == 0:
        return 0, 1.0, 1, None

    return stats.chi2_contingency(data,
                                  correction=correction,
                                  lambda_="log-likelihood")


def cond_mutual_info(data, norm=False, ci_level=None, p=True):
    """
    Compute the conditional mutual information of two variables given a third

    Parameters
    ----------
    data :
        A 3-dimensional table. This method computes the mutual
        information between the first and second dimensions, given the third.

    norm :
        whether the MI should be normalized

    ci_level :
        level for confidence intervals (or None)

    p :
        whether a p-value should be computed (for efficiency reasons when
        bootstrapping)

    Returns
    -------
    max :
        upper bound of confidence interval

    min :
        lower bound of confidence interval

    pval :
        the corresponding p-value

    References
    ----------
    https://en.wikipedia.org/wiki/Conditional_mutual_information
    """

    assert not p or ci_level

    data = [ct.values if isinstance(ct, pd.DataFrame) else ct for ct in data]
    data = np.array([d for d in data if d.sum() > 0])

    N = data.sum()

    if p:
        G, pval, df = G_test_cond(data)
    else:
        pval = 0

    weights = map(lambda d: d.sum(), data)
    hxs = map(lambda d: stats.entropy(np.sum(d, axis=1)), data)
    cond_hx = np.average(hxs, axis=None, weights=weights)
    hys = map(lambda d: stats.entropy(np.sum(d, axis=0)), data)
    cond_hy = np.average(hys, axis=None, weights=weights)
    hxys = map(lambda d: stats.entropy(d.flatten()), data)
    cond_hxy = np.average(hxys, axis=None, weights=weights)

    mi = -cond_hxy + cond_hx + cond_hy

    # normalized mutual info
    if norm:
        if (cond_hx == 0) or (cond_hy == 0) or (mi == 0):
            mi = 0
        else:
            mi = mi/min(cond_hx, cond_hy)

    # no confidence levels, return single measure
    if not ci_level:
        return mi, pval

    p_low = 1-(1-ci_level)/2
    p_high = (1-ci_level)/2
    G_low = special.chndtrinc(G, df, p_low)
    G_high = special.chndtrinc(G, df, p_high)
    ci = ((G_low+df)/(2.0*N), (G_high+df)/(2.0*N))

    if pval > 1-ci_level:
        ci = (0, ci[1])

    if norm:
        ci = map(lambda x: x/min(cond_hx, cond_hy), ci)

    return max(ci[0], 0), min(ci[1], 1), pval


def G_test_cond(data):
    """
    Computes a conditional G-test

    Parameters
    ----------
    data :
        3-dimensional contingency table to test

    Returns
    -------
    G :
        G-test score

    p_val :
        The corresponding p value

    df :
        degrees of freedom

    References
    ----------
    https://en.wikipedia.org/wiki/G-test
    """
    G = 0

    # the degrees of freedom are (S-1)*(O-1)*K
    df = 0
    for group in data:
        G_temp, _, df_temp, _ = G_test(group)
        G += G_temp
        df = max(df, df_temp)

    df *= len(data)
    p_val = stats.chisqprob(G, df)

    return G, p_val, df


def Fisher_test(data):
    """
    Computes the exact Fisher test

    Parameters
    ----------
    data :
        contingency table to test

    Returns
    -------
    fisher_exact :
        Fisher exact test

    References
    ----------
    https://en.wikipedia.org/wiki/Fisher%27s_exact_test
    """
    assert data.shape == (2, 2)
    return stats.fisher_exact(data)


def mi_sigma(data):
    """
    standard deviation of mutual information (Basharin '59)

    Parameters
    ----------
    data :
        contingency table

    References
    ----------
    http://epubs.siam.org/doi/abs/10.1137/1104033
    """
    data = np.array(data, dtype='float')
    pxs = np.sum(data, axis=1)
    pys = np.sum(data, axis=0)

    pxs = pxs/sum(pxs)
    pys = pys/sum(pys)

    N = sum(sum(data))
    pxys = data/N

    mi_sqr = 0
    mi = 0
    for i in range(0, len(pxs)):
        px = pxs[i]
        if px != 0:
            for j in range(0, len(pys)):
                py = pys[j]

                pxy = pxys[i][j]
                if pxy != 0:
                    mi_sqr += pxy * pow((log(pxy)-log(px)-log(py)), 2)
                    mi += pxy * (log(pxy)-log(px)-log(py))

    return sqrt((mi_sqr-pow(mi, 2))/N)


def cramer_v(data, ci_level=0.95):
    """
    Confidence Intervals for Cramér's V

    Parameters
    ----------
    data :
        contingency table

    ci_level :
        confidence intervals levels

    Returns
    -------
    cv_low :
        lower bound of confidence interval

    cv_high :
        upper bound of confidence Interval

    References
    ----------
    http://psychology3.anu.edu.au/people/smithson/details/CIstuff/CI.html
    """
    # remove all-zero columns/rows
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    data = data[data.columns[(data != 0).any()]]
    data = data[(data.T != 0).any()]

    chi2, _, _, _ = stats.chi2_contingency(data, correction=False)
    dim = data.shape
    df = (dim[0]-1)*(dim[1]-1)
    n = data.sum().sum()

    cv = sqrt(chi2/(n*(min(dim[0], dim[1])-1)))

    if not ci_level:
        return cv

    p_low = 1-(1-ci_level)/2
    p_high = (1-ci_level)/2
    chi_low = special.chndtrinc(chi2, df, p_low)
    chi_high = special.chndtrinc(chi2, df, p_high)
    # print chi_low, chi_high
    (cv_low, cv_high) = map(lambda x: sqrt((x+df)/(n*(min(dim[0], dim[1])-1))),
                            (chi_low, chi_high))
    return cv_low, cv_high


def correlation(counts, ci_level=None):
    """
    Pearson correlation with confidence intervals

    Parameters
    ----------
    counts :
        statistics for correlation computation
        (sum_x, sum_x2, sum_y, sum_y2, sum_xy, n)

    ci_level :
        level for confidence intervals (or None)

    Returns
    -------
    conf_corr[0] :
        the lower bound of confidence interval

    conf_corr[1] :
        the upper bound of confidence interval

    p_val :
        the corresponding p value

    References
    ---------
    https://en.wikipedia.org/wiki/Correlation_and_dependence
    """
    assert len(counts) == 6
    sum_x = counts[0]
    sum_x2 = counts[1]
    sum_y = counts[2]
    sum_y2 = counts[3]
    sum_xy = counts[4]
    n = counts[5]

    corr = (n*sum_xy - sum_x*sum_y)/(sqrt(n*sum_x2 - sum_x**2) *\
            sqrt(n*sum_y2 - sum_y**2))

    # Fisher transform
    fisher = atanh(corr)
    std = 1.0/sqrt(n-3)
    z = fisher/std
    pval = 2*stats.norm.sf(abs(z))

    if ci_level:
        conf_fisher = stats.norm.interval(ci_level, loc=fisher, scale=std)

        # inverse transform
        conf_corr = [tanh(conf_fisher[0]), tanh(conf_fisher[1])]

        if np.isnan(conf_corr[0]) or np.isnan(conf_corr[1]) or np.isnan(pval):
            return -1, 1, 1.0

        return conf_corr[0], conf_corr[1], pval
    else:
        return abs(corr), pval


def bootstrap_ci_ct(data, stat, num_samples=10000, ci_level=0.95):
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

    ci_level :
        Confidence level for the interval

    Returns
    -------
    ci :
        the confidence interval

    References
    ----------
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    dim = data.shape
    data = data.flatten()
    data += 1
    n = data.sum()

    #print 'Bootstrap on data of size {}'.format(n)
    probas = (1.0*data)/n

    # Obtain `num_samples' random samples of `n' multinomial values, sampled
    # with replacement from {0, 1, ..., n-1}. For each sample, rebuild a
    # contingency table and compute the stat.
    temp = np.random.multinomial(n, probas, size=num_samples)
    bs_stats = [row.reshape(dim) for row in temp]
    bs_stats = map(lambda ct: stat(ct), bs_stats)

    alpha = 1-ci_level
    q_low = np.percentile(bs_stats, 100*alpha/2)
    q_high = np.percentile(bs_stats, 100*(1-alpha/2))

    ci = (q_low, q_high)
    return ci


def bootstrap_ci_ct_cond(data, stat, num_samples=10000, ci_level=0.95):
    """
    Bootstrap confidence interval computation on a 3-way contingency table

    Parameters
    ----------
    data :
        Contingency table collected from independent samples

    stat :
        Statistic to bootstrap. Takes a contingency table as argument

    num_samples :
        Number of bootstrap samples to generate

    ci_level :
        Confidence level for the interval

    Returns
    -------
    ci :
        the confidence interval

    References
    ----------
    """
    data = np.array([ct.values if isinstance(ct, pd.DataFrame)
                     else ct for ct in data])

    dim = data.shape
    data = [ct.flatten()+1 for ct in data]

    probas = [(1.0*ct)/ct.sum() for ct in data]

    # Obtain `num_samples' random samples of `n' multinomial values, sampled
    # with replacement from {0, 1, ..., n-1}. For each sample, rebuild a
    # contingency table and compute the stat.
    temp = np.dstack([np.random.multinomial(data[i].sum(),
                                            probas[i],
                                            size=num_samples)
                      for i in range(dim[0])])
    bs_stats = [row.reshape(dim) for row in temp]
    bs_stats = map(lambda ct: stat(ct), bs_stats)

    alpha = 1-ci_level
    q_low = np.percentile(bs_stats, 100*alpha/2)
    q_high = np.percentile(bs_stats, 100*(1-alpha/2))

    ci = (q_low, q_high)
    return ci


def bootstrap_ci_corr(x, y, stat, num_samples=10000, ci_level=0.95):
    """
    Bootstrap confidence interval computation for correlation

    Parameters
    ----------
    x :
        Values for the first dimension

    y :
        Values for the second dimension

    stat :
        Statistic to bootstrap. Takes values (x,y) as argument

    num_samples :
        Number of bootstrap samples to generate

    ci_level :
        Confidence level for the interval

    Returns
    -------
    ci :
        the confidence interval

    References
    ----------
    """
    ci = bs.ci((x, y), stat, alpha=1-ci_level, n_samples=num_samples)
    return ci


def permutation_test_ct_cond(data, num_samples=10000):
    """
    Monte-Carlo permutation test for a 3-way contingency table

    Parameters
    ----------
    data :
        the contingency table

    num_samples :
        the number of random permutations to perform

    Returns
    -------
    p :
        the p-value

    References
    ----------
    https://en.wikipedia.org/wiki/Resampling_(statistics)
    """
    data = np.array([ct.values if isinstance(ct, pd.DataFrame)
                     else ct for ct in data])

    dim = data.shape
    data_x = [[] for ct in data]
    data_y = [[] for ct in data]

    stat_0 = cond_mutual_info(data, norm=False, ci_level=None, p=False)[0]

    for e in range(0, dim[0]):
        for x in range(0, dim[1]):
            for y in range(0, dim[2]):
                data_x[e] += [x]*data[e, x, y]
                data_y[e] += [y]*data[e, x, y]

    k = 0
    N = data.sum()
    for i in range(num_samples):
        mi_cond = 0
        for e in range(0, dim[0]):
            np.random.shuffle(data_x[e])
            mi_cond += (1.0*len(data_x[e]))/N * \
                       metrics.mutual_info_score(data_x[e], data_y[e])
        k += stat_0 < mi_cond

    pval = (1.0*k) / num_samples
    return max(pval, 1.0/num_samples)


def permutation_test_ct(data, num_samples=10000):
    """
    Monte-Carlo permutation test for a contingency table

    Parameters
    ----------
    data :
        the contingency table

    num_samples :
        the number of random permutations to perform

    Returns
    -------
    p :
        the p-value

    Notes
    -----
    Uses the R 'coin' package that can directly handle contingency
    tables instead of having to convert into arrays x,y

    References
    ----------
    https://en.wikipedia.org/wiki/Resampling_(statistics)
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    #print 'permutation test of size {}'.format(data.sum())

    data = np.array(data, dtype='int')

    ro.globalenv['ct'] = data
    ro.r('res = chisq_test(as.table(ct), distribution=approximate(B={}))'.
         format(num_samples))
    pval = ro.r('pvalue(res)')[0]
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
    p :
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
    for j in range(num_samples):
        np.random.shuffle(z)
        k += abs(obs_0) < abs(stats.pearsonr(z[:len(x)], z[len(x):])[0])
    pval = (1.0*k) / num_samples
    return max(pval, 1.0/num_samples)
