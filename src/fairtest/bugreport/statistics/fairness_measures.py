# -*- coding: utf-8 -*-

import scipy.stats as stats
import scipy.special as special
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from math import sqrt, log, exp, atanh, tanh
from collections import Counter

class Measure(object):
    def __init__(self, ci_level=None):
        self.ci_level = ci_level
    
class NMI(Measure):
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
        
        z = abs(stats.norm.ppf(pval))
        mean = ci_high-(ci_high-ci_low)/2
        std = mean/z
        ci = stats.norm.interval(self.ci_level, loc=mean, scale=std)
        return max(0,ci[0]), min(ci[1],1), pval
        
class CORR(Measure):
    def compute(self, data):
        return correlation(data, ci_level=self.ci_level)
        
    def normalize_effect(self, res):
        if self.ci_level:
            (ci_low, ci_high, p) = res
            return 0 if (ci_low * ci_high < 0) else min(abs(ci_low), abs(ci_high))
        else:
            return abs(res)
    
    def ci_from_p(self, ci_low, ci_high, pval):
        if pval == 1:
            return 0, 0, 1
        
        z = abs(stats.norm.ppf(pval))
        mean = ci_high-(ci_high-ci_low)/2
        std = abs(mean/z)
        ci = stats.norm.interval(self.ci_level, loc=mean, scale=std)
        return max(-1,ci[0]), min(ci[1],1), pval
        
class DIFF(Measure):
    def compute(self, data):
        return difference(data, ci_level=self.ci_level)
        
    def normalize_effect(self, res):
        if self.ci_level:
            (ci_low, ci_high, p) = res
            return 0 if (ci_low * ci_high < 0) else min(abs(ci_low), abs(ci_high))
        else:
            return abs(res)
    
    def ci_from_p(self, ci_low, ci_high, pval):
        if pval == 1:
            return 0, 0, 1
        
        z = abs(stats.norm.ppf(pval))
        mean = ci_high-(ci_high-ci_low)/2
        std = abs(mean/z)
        ci = stats.norm.interval(self.ci_level, loc=mean, scale=std)
        return max(-1,ci[0]), min(ci[1],1), pval
        
class RATIO(Measure):
    def compute(self, data):
        return ratio(data, ci_level=self.ci_level)
        
    def normalize_effect(self, res):
        if self.ci_level:
            (ci_low, ci_high, p) = res
            return 0 if (log(ci_low) * log(ci_high) < 0) else min(abs(log(ci_low)), abs(log(ci_high)))
        else:
            return abs(log(res))
    
    def ci_from_p(self, ci_low, ci_high, pval):
        if pval == 1:
            return 1, 1, 1
        
        z = abs(stats.norm.ppf(pval))
        mean = log(ci_high)-(log(ci_high)-log(ci_low))/2
        std = abs(mean/z)
        
        ci = stats.norm.interval(self.ci_level, loc=mean, scale=std)
        return exp(ci[0]), exp(ci[1]), pval

# 
# mutual information with or without normalization
#
# @args data    the contingency table to compute the MI of
# @args norm    whether the MI should be normalized
# @args ci_level    level for confidence intervals (or None)
#
'''
def mutual_info(data, norm=False, ci_level=None):
    mi = metrics.mutual_info_score(None, None, contingency=data)
    
    #
    # Normalize I(X,Y) by dividing by the min entropy of X and Y
    #
    if (norm):
        # compute marginal distributions of X and Y
        px = np.sum(data, axis=1)
        py = np.sum(data, axis=0)
        
        px = np.array(px, dtype=float)/sum(px)
        py = np.array(py, dtype=float)/sum(py)
        
        hx = stats.entropy(px)
        hy = stats.entropy(py)
        
        mi = mi/min(hx,hy)
    
    # compute confidence interval (measure, delta)
    if (ci_level):
        if not norm:
            std = mi_sigma(data)
            conf_mi = stats.norm.interval(ci_level, loc=mi, scale=std)
            z = mi/std
            p_val = 2*(1 - stats.norm.cdf(abs(z)))
        else:   
            #p_val = 2*(1 - special.ndtr(abs(z)))
            #_, pval, _, _ = G_test(data)
            #conf_mi = ci_from_p(pval, mi, ci_level)
            conf_mi, p_val = bootstrap_ci(data, lambda x: mutual_info(x, norm), ci_level=ci_level)
        
        return mi, (conf_mi[1]-conf_mi[0])/2, p_val
        
    return mi
'''
    
def mutual_info(data, norm=False, ci_level=None):
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # data smoothing if there are very small values
    #if data.shape == (2,2):
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
        if (hx == 0 or hy == 0 or mi == 0):
            mi = 0
        else:
            mi = mi/min(hx, hy)
    
    # no confidence levels, return single measure
    if not ci_level:
        return mi
    
    # get asymptotic standard deviation for confidence interval 
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
        std = sqrt((std-pow(mi,2))/N)
    
    if std != 0:
        # compute asymptotic confidence interval and p-value
        ci = stats.norm.interval(ci_level, loc=mi, scale=std)
        z = mi/std
        pval = 2*stats.norm.sf(abs(z))
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
    
    sp =  0.5*sum(abs(data[:,0]-data[:,1]))
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
    
        #confidence intervals
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
        
        #confidence intervals
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

    return stats.chi2_contingency(data, correction=False, 
                                    lambda_="log-likelihood")
                                    
#
# Compute the conditional mutual information of two variables given a third
#
# @args data   A 3-dimensional table. This method computes the mutual 
#               information between the first and second dimensions, given the 
#               third.
#        
def mi_cond(data): 
    mi  = 0
    
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


# number of randomized samplings in the Monte Carlo test
N_SAMPLES = 1000


#
# Perform a Monte-Carlo permutation test
#
# @args data    the contingency table to test
#
def permutation_test(data):
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
    estimates = np.array(map(lambda x: run_permutation_test(list(pool),counts),range(N_SAMPLES)))
    
    # compute the two-sided p-value
    diffCount = len(np.where(estimates <= obs)[0])
    p_val = 1.0 - (float(diffCount)/float(N_SAMPLES))
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
    #stats = np.sort(np.apply_along_axis(lambda row: stat(row.reshape(dim)), 1, np.random.multinomial(n, probas, size=num_samples)))
    #return (stats[int((alpha/2.0)*num_samples)], stats[int((1-alpha/2.0)*num_samples)])
    
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
    #pval = 2*(1 - stats.norm.cdf(abs(z)))
    pval = 2*stats.norm.sf(abs(z))
    
    #print z, pval, pval2
    
    
    return ci, pval
    
# Computes a confidence interval from a p-value, assuming asymptotic normality
#
# @args pval        the p-value
# @args mean        the mean of the confidence interval (expected value of the asymptotic normal)
# @args ci_level    confidence level for the interval
#
def ci_from_p(pval, mean, ci_level):
    # get the z-statistic for a normal distribution test from the p-value
    z = min(stats.norm.ppf(1-pval/2), 100)
    
    # get the standard deviation of the normal
    std = abs(mean/z)

    # compute a confidence interval around mean
    ci = stats.norm.interval(ci_level, loc=mean, scale=std)
    
    print pval, z, std, ci
    
    return ci
     