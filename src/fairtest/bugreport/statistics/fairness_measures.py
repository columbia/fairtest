# -*- coding: utf-8 -*-

import scipy.stats as stats
import scipy.special as special
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from math import sqrt, log, exp, atanh, tanh
from collections import Counter

# 
# mutual information with or without normalization
#
# @args data    the contingency table to compute the MI of
# @args norm    whether the MI should be normalized
# @args ci_level    level for confidence intervals (or None)
#
def mutual_info(data, norm=False, ci_level=None):
    assert (not (norm and ci_level))
    
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
        std = mi_sigma(data)
        conf_mi = stats.norm.interval(ci_level, loc=mi, scale=std)
        z = mi/std
        pval = 2*(1 - stats.norm.cdf(abs(z)))
        return mi, conf_mi[1]-mi
        
    return mi

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
# slift measures, possibly with confidence intervals
#
# @args data        contingency table  
# @args ci_level    level for confidence intervals (or None)
#
def slifts(data, ci_level=0.95):
    assert (data.shape == (2,2))
    
    # transform contingency table into probability table
    tot = np.sum(data, axis=0)
    probas = np.array(np.array(data, dtype=float)/tot, dtype='float')
    
    # Slift measures
    slift = probas[1,0]/probas[1,1]
    slift_d = probas[1,0]-probas[1,1]
    
    #
    # confidence levels as in Ruggieri et al. '10
    #
    if ci_level:
        # contingency table values
        n1 = sum(data[1])
        n2 = sum(data[0])
        a1 = data[1][1]
        a2 = data[0][1]
        
        # proba of hired
        p = (1.0*(a1 + a2))/(n1+n2)
    
        # standard deviations
        sigma_diff = sqrt(p*(1-p)*(1.0/n1+1.0/n2))
        sigma_ratio = sqrt(1.0/a1-1.0/n1+1.0/a2-1.0/n2)
    
        #confidence intervals
        int_diff = stats.norm.interval(ci_level, loc=slift_d, scale=sigma_diff)
        int_ratio = stats.norm.interval(ci_level, loc=slift, scale=sigma_ratio)
        
        return (slift, slift_d, int_ratio[1]-slift, int_diff[1]-slift_d)
    else:
        return (slift, slift_d)

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
        conf_fisher = stats.norm.interval(ci_level, loc=fisher, scale=std)
        
        # inverse transform
        conf_corr = [tanh(conf_fisher[0]), tanh(conf_fisher[1])]
        
        return abs(corr), conf_corr[1]-corr         
        
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
def bootstrap_ci(data, stat, num_samples=10000, ci_level=0.95):
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
    bs_stats = np.apply_along_axis(lambda row: stat(row.reshape(dim)), 1, np.random.multinomial(n, probas, size=num_samples))
    
    # Use the stats to get an estimator of the mean and std of the stat
    mean = np.mean(bs_stats)
    std = np.std(bs_stats, ddof=1)
    
    # get a confidence interval as (mean-z*std, mean+z*std)
    ci = stats.norm.interval(ci_level, loc=mean, scale=std)
    z = mean/std
    pval = 2*(1 - stats.norm.cdf(abs(z)))
    return ci, pval
    
# Computes a confidence interval from a p-value, assuming asymptotic normality
#
# @args pval        the p-value
# @args mean        the mean of the confidence interval (expected value of the asymptotic normal)
# @args ci_level    confidence level for the interval
#
def ci_from_p(pval, mean, ci_level):
    # get the z-statistic for a normal distribution test from the p-value
    z = stats.norm.ppf(1-pval/2)
    
    # get the standard deviation of the normal
    std = abs(stat/z)
    
    # compute a confidence interval around mean
    ci = stats.norm.interval(ci_level, loc=mean, scale=std)
    return ci
     