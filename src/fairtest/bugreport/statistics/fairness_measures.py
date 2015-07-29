# -*- coding: utf-8 -*-

import scipy.stats as stats
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from math import sqrt, log
from collections import Counter

# 
# mutual information with or without normalization
#
# @args data    the contingency table to compute the MI of
# @args norm    whether the MI should be normalized
# @args ci      whether to compute confidence intervals
# @args level   confidence level
def mutual_info(data, norm=True, ci=False, level=0.95):
    mi = metrics.mutual_info_score(None, None, contingency=data)
    
    assert (not (norm and ci))
    
    #
    # Normalize I(X,Y) by dividing by the min entropy of X and Y
    #
    if (norm):
        # compute marginal distributions of X and Y
        px = np.sum(data, axis=1)
        py = np.sum(data, axis=0)
        
        px = px/sum(px)
        py = py/sum(py)
        
        hx = stats.entropy(px)
        hy = stats.entropy(py)
        
        mi = mi/min(hx,hy)
    
    # compute confidence interval
    if (ci):
        conf_mi = stats.norm.interval(level, loc=mi, scale=mi_sigma(data))
        return mi, conf_mi[1]-mi
        
        '''
        
        Method of Stefani et al. Does not seem to give good intervals
        
        def bin_entropy(x):
            return -x*log(x)-(1-x)*log(1-x)
            
        alpha = 1-level
        Mx = min(np.array(data).shape)
        My = max(np.array(data).shape)
        n = np.sum(data)
        epsilon = sqrt(2.0/n*(log(2**(Mx*My) - 2) - log(alpha)))

        if epsilon <= 2-2.0/Mx:
            mi_delta = epsilon/2*(log(Mx*My-1)+log(Mx-1)+log(My-1)) + 3*bin_entropy(epsilon/2)
        else:
            mi_delta = log(Mx)
        
        return mi, mi_delta
        '''
        
    return mi

#
# statistical parity measure
#
# @args data    the contingency table of shape (?, 2)
def statistical_parity(data):
    assert data.shape[1] == 2
    
    # transform contingency table into probability table
    tot = np.sum(data, axis=0)
    data = np.array(data/tot, dtype='float')
    
    sp =  0.5*sum(abs(data[:,0]-data[:,1]))
    return sp
    
# 
# slift measures, possibly with confidence intervals
#
# @args data    contingency table  
# @args ci      whether to compute confidence intervals
# @args level   confidence level
#
def slifts(data, ci=False, level=0.95):
    assert (data.shape == (2,2))
    
    # transform contingency table into probability table
    tot = np.sum(data, axis=0)
    data = np.array(data/tot, dtype='float')
    
    # Slift measures
    slift = data[1,0]/data[1,1]
    slift_d = data[1,0]-data[1,1]
    
    
    #
    # confidence levels as in Ruggieri et al. '10
    #
    if ci:
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
        int_diff = stats.norm.interval(level, loc=slift_d, scale=sigma_diff)
        int_ratio = stats.norm.interval(level, loc=slift, scale=sigma_ratio)
        
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
    
    '''
    for group in data:
        p = (1.0*group.sum().sum())/tot
        mi += p*fm.mutual_info(group, norm=False)
    '''
    
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
     