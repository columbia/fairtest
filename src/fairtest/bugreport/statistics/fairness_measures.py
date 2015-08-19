# -*- coding: utf-8 -*-

import scipy.stats as stats
import scipy.special as special
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from math import sqrt, log, exp, atanh, tanh
from collections import Counter


#
# Compute an effect score for a Z-score
# This is the absolute value of the lower bound of the confidence interval, if the interval does not contain zero.
#
# @args ci_low  Lower bound of the confidence interval
# @args ci_high Upper bound of the confidence interval
#
def z_effect(ci_low, ci_high):
    return 0 if (ci_low * ci_high < 0) else min(abs(ci_low), abs(ci_high))


#
# Recompute an asymptotically normal confidence interval from a corrected p-value
#
# @args ci_low      Lower bound of the original confidence interval
# @args ci_high     Upper bound of the original confidence interval
# @args pval        The adjusted p-value
# @args ci_level    The confidence level
#
def z_ci_from_p(ci_low, ci_high, pval, ci_level):
    pval = max(pval, 1e-180)

    z = abs(stats.norm.ppf(pval))
    mean = ci_high-(ci_high-ci_low)/2
    std = abs(mean/z)

    return stats.norm.interval(ci_level, loc=mean, scale=std)


#
# Class representing a fairness measure
#
class Measure(object):
    # Types of measures
    DATATYPE_CT = 'ct'      # Measures on a contingency table
    DATATYPE_CORR = 'corr'  # Correlation measures
    DATATYPE_REG = 'reg'    # Regression measures

    def __init__(self, ci_level=0.95):
        self.ci_level = ci_level


#
# Normalized Mutual Information measure
#
class NMI(Measure):
    dataType = Measure.DATATYPE_CT

    def compute(self, data):
        return mutual_info(data, norm=True, ci_level=self.ci_level)
        
    def normalize_effect(self, res):
        if self.ci_level:
            (ci_low, ci_high, p) = res
            return ci_low
        else:
            return res
    
    def ci_from_p(self, ci_low, ci_high, pval):
        if pval == 1:
            return 0, 0, 1
        
        ci = z_ci_from_p(ci_low, ci_high, pval, self.ci_level)
        return max(0, ci[0]), min(ci[1], 1), pval


#
# Pearson Correlation measure
#
class CORR(Measure):
    dataType = Measure.DATATYPE_CORR

    def compute(self, data):
        return correlation(data, ci_level=self.ci_level)
        
    def normalize_effect(self, res):
        if self.ci_level:
            (ci_low, ci_high, p) = res
            return z_effect(ci_low, ci_high)
        else:
            return abs(res)
    
    def ci_from_p(self, ci_low, ci_high, pval):
        if pval == 1:
            return 0, 0, 1
        
        ci = z_ci_from_p(ci_low, ci_high, pval, self.ci_level)
        return max(-1,ci[0]), min(ci[1],1), pval


#
# Difference measure
#
class DIFF(Measure):
    dataType = Measure.DATATYPE_CT

    def compute(self, data):
        return difference(data, ci_level=self.ci_level)
        
    def normalize_effect(self, res):
        if self.ci_level:
            (ci_low, ci_high, p) = res
            return z_effect(ci_low, ci_high)
        else:
            return abs(res)
    
    def ci_from_p(self, ci_low, ci_high, pval):
        if pval == 1:
            return 0, 0, 1
        
        ci = z_ci_from_p(ci_low, ci_high, pval, self.ci_level)
        return max(-1,ci[0]), min(ci[1],1), pval


#
# Ratio measure
#
class RATIO(Measure):
    dataType = Measure.DATATYPE_CT

    def compute(self, data):
        return ratio(data, ci_level=self.ci_level)
        
    def normalize_effect(self, res):
        if self.ci_level:
            (ci_low, ci_high, p) = res
            return z_effect(log(ci_low), log(ci_high))
        else:
            return abs(log(res))
    
    def ci_from_p(self, ci_low, ci_high, pval):
        if pval == 1:
            return 1, 1, 1
        
        ci = z_ci_from_p(log(ci_low), log(ci_high), pval, self.ci_level)
        return exp(ci[0]), exp(ci[1]), pval


#
# Regression measure
#
class REGRESSION(Measure):
    dataType = Measure.DATATYPE_REG

    def __init__(self, ci_level=None, topK=10):
        Measure.__init__(self, ci_level)
        self.topK = topK

    def compute(self, data):
        y = data[data.columns[-1]]
        X = data[data.columns[0:-1]]

        # print 'Regressing from {}...{} to {}'.format(data.columns[0], data.columns[-2], data.columns[-1])

        reg = LogisticRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)

        # approximate the standard errors for all regression coefficients
        mse = np.mean((y - y_pred.T)**2)
        var_est = mse * np.diag(np.linalg.pinv(np.dot(X.T, X)))
        SE_est = np.sqrt(var_est)
        coeffs = reg.coef_[0].tolist()

        if not self.ci_level:
            return coeffs

        # compute confidence intervals and p-values for all coefficients
        results = pd.DataFrame(coeffs, columns=['coeff'])
        results['std err'] = SE_est
        results['z'] = abs(results['coeff']/results['std err'])
        results['p-value'] = 2*stats.norm.sf(results['z'])
        ci_s = stats.norm.interval(self.ci_level, loc=results['coeff'], scale=results['std err'])
        results['conf low'] = exp(ci_s[0])
        results['conf high'] = exp(ci_s[1])

        # compute a standardized effect size and return the topK coefficients
        results['effect'] = map(lambda (ci_low, ci_high): z_effect(ci_low, ci_high), zip(results['conf low'], results['conf high']))
        sorted_results = results.sort(columns=['effect'], ascending=False)
        return sorted_results[['conf low', 'conf high', 'p-value']].head(self.topK)

    def normalize_effect(self, res):
        if self.ci_level:
            return np.mean(map(lambda (ci_low, ci_high, p): z_effect(ci_low, ci_high), res.values))
        else:
            return np.mean(abs(res))

    def ci_from_p(self, ci_low, ci_high, pval):
        if pval == 1:
            return 0, 0, 1

        ci = z_ci_from_p(ci_low, ci_high, pval, self.ci_level)
        return ci[0], ci[1], pval


# 
# mutual information with or without normalization and confidence intervals
#
# @args data        the contingency table to compute the MI of
# @args norm        whether the MI should be normalized
# @args ci_level    level for confidence intervals (or None)
#
def mutual_info(data, norm=False, ci_level=None):
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # data smoothing
    data = data.copy()
    data[data == 0] = 1
    
    # row/column sums
    sum_x = np.sum(data, axis=1)
    sum_y = np.sum(data, axis=0)
        
    # row/columns probabilities
    px = np.array(sum_x, dtype=float)/sum(sum_x)
    py = np.array(sum_y, dtype=float)/sum(sum_y)
    
    # joint probabilities
    N = np.array(data).sum()

    pxy = np.array(data, dtype=float)/N
    
    # entropies    
    hx = stats.entropy(px)
    hy = stats.entropy(py)
    hxy = stats.entropy(pxy.flatten())

    mi = -hxy + hx + hy

    # normalized mutual info
    if norm:
        if (hx == 0) or (hy == 0) or (mi == 0):
            mi = 0
        else:
            mi = mi/min(hx, hy)
    
    # no confidence levels, return single measure
    if not ci_level:
        return mi

    # get asymptotic standard deviation for confidence interval
    # Brown'75, "The asymptotic standard errors of some estimates of uncertainty in the two-way contingency table"
    std = 0
    for i in range(0, len(px)):
        for j in range(0, len(py)):
            if data[i,j] != 0:
                if norm:
                    # std for normalized MI (Brown'75)
                    if sum_y[j] != 0 and sum_x[i] != 0:
                        if hx < hy and hx != 0:
                            std += data[i,j]*(hx*log((1.0*data[i,j])/sum_y[j])+((hy-hxy)*log((1.0*sum_x[i])/N)))**2/(N**2*hx**4);
                        elif hy != 0:
                            std += data[i,j]*(hy*log((1.0*data[i,j])/sum_x[i])+((hx-hxy)*log((1.0*sum_y[j])/N)))**2/(N**2*hy**4);
                else:
                    # std for MI
                    std += pxy[i,j] * pow((log(pxy[i,j])-log(px[i])-log(py[j])), 2)

    if norm:
        std = sqrt(std)
    else:
        std = sqrt((std-pow(mi, 2))/N)

    if std != 0:
        # compute asymptotic confidence interval and p-value
        #ci = stats.norm.interval(ci_level, loc=mi, scale=std)
        #z = mi/std
        #pval = 2*stats.norm.sf(abs(z))

        _, pval, _, _ = G_test(data)
        #print G,pval2,z,pval,z**2,mi,std

        ci = z_ci_from_p(mi, mi, pval, ci_level)

        return max(ci[0],0), min(ci[1],1), pval
    else:
        return mi, 0, 1


#
# statistical parity measure
#
# @args data    the contingency table of shape (?, 2)
#
def statistical_parity(data):
    assert data.shape[1] == 2
    
    # transform contingency table into probability table
    tot = np.sum(data, axis=0)
    data = np.array(np.array(data, dtype=float)/tot, dtype='float')
    
    sp = 0.5*sum(abs(data[:,0]-data[:,1]))
    return sp


# 
# Difference measure, possibly with confidence intervals
#
# @args data        contingency table  
# @args ci_level    level for confidence intervals (or None)
#
def difference(data, ci_level=0.95):
    assert (data.shape == (2,2))
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # transform contingency table into probability table
    tot = np.sum(data, axis=0)
    probas = np.array(data, dtype=float)/tot
    
    # Difference measure
    diff = probas[1,0]-probas[1,1]
    
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
    
        # confidence intervals
        ci_diff = stats.norm.interval(ci_level, loc=diff, scale=sigma_diff)
        
        z = diff/sigma_diff
        pval = 2*stats.norm.sf(abs(z))

        return max(ci_diff[0], -1), min(ci_diff[1], 1), pval
    else:
        return diff


# 
# Ratio measure, possibly with confidence intervals
#
# @args data        contingency table  
# @args ci_level    level for confidence intervals (or None)
#
def ratio(data, ci_level=0.95):
    assert (data.shape == (2,2))
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # data smoothing
    data = data.copy()
    data += 5
    
    # transform contingency table into probability table
    tot = np.sum(data, axis=0)
    probas = np.array(data, dtype=float)/tot
    
    # Slift measures
    ratio = probas[1,0]/probas[1,1]
    
    #
    # confidence levels as in Ruggieri et al. '10
    #
    if ci_level:
        # contingency table values
        n1 = tot[0]
        n2 = tot[1]
        r1 = data[1][0]
        r2 = data[1][1]
        
        # standard deviations
        sigma_log_ratio = sqrt(1.0/r1+1.0/r2-1.0/n1-1.0/n2)
        
        # confidence intervals
        ci_log_ratio = stats.norm.interval(ci_level, loc=log(ratio), scale=sigma_log_ratio)

        z = log(ratio)/sigma_log_ratio
        pval = 2*stats.norm.sf(abs(z))

        return exp(ci_log_ratio[0]), exp(ci_log_ratio[1]), pval
    else:
        return ratio


#
# Computes the G_test
#
# @args data        contingency table to test
# @args correction  whether to apply continuity corrections
# 
def G_test(data, correction=False):
    # remove all-zero columns/rows
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    data = data[data.columns[(data != 0).any()]]
    data = data[(data.T != 0).any()]

    return stats.chi2_contingency(data, correction=False, lambda_="log-likelihood")


#
# Compute the conditional mutual information of two variables given a third
#
# @args data   A 3-dimensional table. This method computes the mutual 
#               information between the first and second dimensions, given the 
#               third.
#        
def mi_cond(data):
    # total size of the data
    tot = sum(map(lambda group: group.sum().sum(), data))
    
    G, _ = G_test_cond(data)
    
    mi = G/(2.0*tot)
    return mi


#
# Computes a conditional G-test
#
# @args data    3-dimensional contingency table to test
#
def G_test_cond(data): 
    G = 0
    
    # the degrees of freedom are (S-1)*(O-1)*K
    df = 0
    for group in data:
        G_temp, _, df_temp, _ = G_test(group)
        G += G_temp
        df = max(df, df_temp)
    
    df *= len(data)
    p_val = stats.chisqprob(G, df)
    
    return G, p_val


#
# Computes the exact Fisher test
#
# @args data    contingency table to test
def Fisher_test(data):
    assert (data.shape == (2,2))
    return stats.fisher_exact(data)


#
# standard deviation of mutual information (Basharin '59)
#
# @args data    contingency table
def mi_sigma(data):
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
        if(px != 0):
            for j in range(0, len(pys)):
                py = pys[j]
                
                pxy = pxys[i][j]
                if (pxy != 0):
                    mi_sqr += pxy * pow((log(pxy)-log(px)-log(py)), 2)
                    mi += pxy * (log(pxy)-log(px)-log(py))
    
    return sqrt((mi_sqr-pow(mi,2))/N)


def cramer_v(data, ci_level=0.95):
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

    #
    # Confidence Intervals for Cramér's V are described here
    # http://psychology3.anu.edu.au/people/smithson/details/CIstuff/CI.html
    #
    p_low = 1-(1-ci_level)/2
    p_high= (1-ci_level)/2
    chi_low = special.chndtrinc(chi2, df, p_low)
    chi_high = special.chndtrinc(chi2, df, p_high)
    print chi_low, chi_high
    (cv_low, cv_high) = map(lambda x: sqrt((x+df)/(n*(min(dim[0], dim[1])-1))), (chi_low, chi_high))
    return (cv_low, cv_high)


#
# Pearson correlation with confidence intervals
#
# @args counts      statistics for correlation computation (sum_x, sum_y, sum_x2, sum_y2, sum_xy, n)
# @args ci_level    level for confidence intervals (or None)
#
def correlation(counts, ci_level=None):
    assert(len(counts) == 6)
    
    sum_x = counts[0]
    sum_x2 = counts[1]
    sum_y = counts[2]
    sum_y2 = counts[3]
    sum_xy = counts[4]
    n = counts[5]
    
    corr = (n*sum_xy - sum_x*sum_y)/(sqrt(n*sum_x2 - sum_x**2) * sqrt(n*sum_y2 - sum_y**2))
    
    if ci_level:
        # Fisher transform
        fisher = atanh(corr)
        std = 1.0/sqrt(n-3)
        z = fisher/std
        pval = 2*stats.norm.sf(abs(z))
        
        conf_fisher = stats.norm.interval(ci_level, loc=fisher, scale=std)
        
        # inverse transform
        conf_corr = [tanh(conf_fisher[0]), tanh(conf_fisher[1])]
        return conf_corr[0], conf_corr[1], pval
    else:
        return abs(corr)


#
# Perform a Monte-Carlo permutation test
#
# @args data        the contingency table to test
# @args n_samples   the number of random permutations to perform
#
def permutation_test(data, n_samples=1000):
    # observed statistic
    obs = G_test(data)[0]
    
    data = np.array(data)
    pool = []
    
    # create a list of all the outcomes
    for i in range(len(data)):
        pool.extend([i]*data[i].sum())
    
    # count the sensitive value frequencies
    counts = data.sum(axis=0)
    
    #
    # Sub-routine for permutation tests
    #
    # @args pool    a list of all outcomes
    # @args counts  the number of elements to sample for each group
    #    
    def run_permutation_test(pool,counts):
        np.random.shuffle(pool)
        
        ct = []
        
        # sample outcomes at random for each group
        for count in counts:
            ct.append(pool[0:count])
            pool = pool[count:-1]
        
        assert not pool
        
        # create a new contingency table and test
        ct = pd.DataFrame(map(lambda c: Counter(c), ct)).T
        return G_test(ct)[0]
    
    
    # perform N_SAMPLES randomized samplings
    estimates = np.array(map(lambda x: run_permutation_test(list(pool),counts),range(n_samples)))
    
    # compute the two-sided p-value
    diffCount = len(np.where(estimates <= obs)[0])
    p_val = 1.0 - (float(diffCount)/float(n_samples))
    return p_val


#
# Bootstrap confidence interval computation
#
# @args data        Contingency table collected from independent samples
# @args stat        Statistic to bootstrap. Takes a contingency table as argument
# @args num_samples Number of bootstrap samples to generate
# @args ci_level    Confidence level for the interval       
#
def bootstrap_ci(data, stat, num_samples=1000, ci_level=0.95):
    if isinstance(data, pd.DataFrame):
        data = data.values

    dim = data.shape
    data = data.flatten()
    n = data.sum()

    probas = (1.0*data)/n
    alpha = 1-ci_level

    #
    # Obtain `num_samples' random samples of `n' multinomial values, sampled
    # with replacement from {0, 1, ..., n-1}. For each sample, rebuild a
    # contingency table and compute the stat.
    #
    temp = np.random.multinomial(n, probas, size=num_samples)
    bs_stats = np.apply_along_axis(lambda row: stat(row.reshape(dim)), 1, temp)
    
    # Use the stats to get an estimator of the mean and std of the stat
    mean = np.mean(bs_stats)
    std = np.std(bs_stats, ddof=1)
    
    # get a confidence interval as (mean-z*std, mean+z*std)
    ci = stats.norm.interval(ci_level, loc=mean, scale=std)
    z = mean/std
    pval = 2*stats.norm.sf(abs(z))

    return ci, pval