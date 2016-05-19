"""
Run multiple hypothesis tests
"""
import pandas as pd
import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
import logging
import multiprocessing
import rpy2.robjects as ro


def compute_all_stats(investigations, exact=True, conf=0.95):
    """
    Compute all statistics for all investigations and protected features

    Parameters
    ----------
    investigations :
        list of investigations

    exact :
        whether exact tests should be used

    conf :
        overall confidence level (1-familywise error rate)
    """

    # reserve the same "confidence budget" for each investigation, independently
    # of the number of hypotheses tested in each
    adj_conf = 1-(1-conf)/len(investigations)
    for inv in investigations:
        inv.stats = compute_investigation_stats(inv, exact, adj_conf)


def compute_investigation_stats(inv, exact=True, conf=0.95):
    """
    Compute all statistics for all protected features of an investigation

    Parameters
    ----------
    inv :
        the investigation

    exact :
        whether exact tests should be used

    conf :
        overall confidence level (1- familywise error rate)

    Returns
    -------
    all_stats:
        list of all statistics for the investigation
    """

    # count the number of hypotheses to test
    total_hypotheses = num_hypotheses(inv)
    logging.info('Testing %d hypotheses', total_hypotheses)

    #
    # Adjusted Confidence Level (Bonferroni)
    #
    adj_conf = 1-(1-conf)/total_hypotheses

    # statistics for all investigations
    all_stats = {sens: compute_stats(ctxts, exact, adj_conf, inv.random_state,
                                     inv.k, inv.m)
                 for (sens, ctxts) in sorted(inv.contexts.iteritems())}

    # flattened array of all p-values
    all_pvals = [max(stat[-1], 1e-180)
                 for sens_stats in all_stats.values()
                 for stat in sens_stats['stats']]

    # correct p-values
    _, pvals_corr, _, _ = multipletests(all_pvals,
                                        alpha=1-conf,
                                        method='holm')

    # replace p-values by their corrected value
    idx = 0

    # iterate over all protected features for the investigation
    for (sens, sens_contexts) in inv.contexts.iteritems():
        sens_stats = all_stats[sens]['stats']
        # iterate over all contexts for a protected feature
        for i in range(len(sens_stats)):
            old_stats = sens_stats[i]
            all_stats[sens]['stats'][i] = \
                np.append(old_stats[0:-1], pvals_corr[idx])
            idx += 1

    for (sens, sens_contexts) in inv.contexts.iteritems():
        metric = sens_contexts[0].metric
        # For regression, re-form the dataframes for each context
        if isinstance(metric.stats, pd.DataFrame):
            res = all_stats[sens]
            res = pd.DataFrame(res['stats'], index=res['index'],
                               columns=res['cols'])
            all_stats[sens] = \
                {'stats':
                     np.array_split(res, len(res)/len(metric.stats))}

    all_stats = {sens: sens_stats['stats']
                 for (sens, sens_stats) in all_stats.iteritems()}

    return all_stats


def num_hypotheses(inv):
    """
    Counts the number of hypotheses to be tested in a single investigation

    Parameters
    ----------
    inv :
        an investigation

    Returns
    -------
    tot :
        the total number of hypotheses to test
    """
    tot = 0
    for contexts in inv.contexts.values():
        metric = contexts[0].metric
        if isinstance(metric.stats, pd.DataFrame):
            tot += len(contexts)*len(metric.stats)
        else:
            tot += len(contexts)

    return tot


def _wrapper((context, conf, exact, seed, k, m)):
    """
    Helper, wrapper used for map_async callback

    Parameters
    ----------
    context :
        a discrimination context

    conf :
        confidence level

    exact :
        whether exact statistics should be computed

    seed :
        a seed for the random number generators

    Returns
    -------
    dict :
        discrimination statistics for the given context
    """

    # seed the PRNGs used to compute statistics
    logging.info('Computing stats for context %d' % context.num)
    ro.r('set.seed({})'.format(seed))
    np.random.seed(seed)
    return context.metric.compute(context.data, conf, k, m, exact=exact).stats


def compute_stats(contexts, exact, conf, seed, k, m):
    """
    Compute statistics for a list of contexts

    Parameters
    ----------
    contexts :
        a list of contexts

    exact :
        whether exact statistics should be computed

    conf :
        confidence level

    seed :
        seed for the PRNGs used to compute statistics

    Returns
    -------
    dict :
        a dictionary containing the computed statistics as well as index
        information if more than one hypothesis was tested in a context
    """
    metric = contexts[0].metric

    logging.info('Computing stats for %d contexts' % len(contexts))

    # P = multiprocessing.Pool(multiprocessing.cpu_count())
    # results = P.map_async(
    #     _wrapper,
    #     zip(contexts, [conf]*len(contexts), [exact]*len(contexts),
    #         [seed]*len(contexts), [k]*len(contexts), [m]*len(contexts))
    # )
    # stats = results.get()
    # P.close()
    # P.join()

    stats = [_wrapper((context, conf, exact, seed, k, m)) for context in contexts]

    #
    # The following block helps parallelization on a spark cluster
    #
    # from pyspark import SparkContext
    # sc = SparkContext("local[4]", "FairTest")
    # rdd = sc.parallelize(
    #     zip(contexts, [conf]*len(contexts),[exact]*len(contexts))
    # )
    # stats = result.collect()
    #

    #
    # When calling 'map_async', the Context object is pickled
    # (passed by value), so we have to apply the change to the stats here.
    # There's probably a cleaner way to do this.
    #
    for (c, c_stats) in zip(contexts, stats):
        c.metric.stats = c_stats

    # For regression, we have multiple p-values per context
    # (one per topK coefficient)
    if isinstance(metric.stats, pd.DataFrame):
        stats = pd.concat(stats)
        index = stats.index
        cols = stats.columns
        stats = stats.values
        return {'stats': stats, 'index': index, 'cols': cols}

    return {'stats': stats}
