"""
Filter and Rank Association Bugs.
"""
from fairtest.modules.metrics import Metric
import pandas as pd
import numpy as np
import logging


# Filters
FILTER_LEAVES_ONLY = 'leaves'
FILTER_ALL = 'all'
FILTER_ROOT_ONLY = 'root'
FILTER_BETTER_THAN_ANCESTORS = "better_than_ancestors"
NODE_FILTERS = [FILTER_ALL,
                FILTER_LEAVES_ONLY,
                FILTER_ROOT_ONLY,
                FILTER_BETTER_THAN_ANCESTORS]


def filter_rank_bugs(context_stats, node_filter=FILTER_BETTER_THAN_ANCESTORS,
                     conf=0.95):
    """
    Print all the contexts sorted by relevance

    Parameters
    -----------
    context_stats :
        list of all contexts and their statistics

    node_filter :
        way to filter the contexts

    conf :
        confidence level for filtering insignificant associations
    """

    logging.info('Filtering and ranking %d sub-contexts' % len(context_stats))

    if node_filter == FILTER_ROOT_ONLY:
        logging.info('0 sub-contexts printed')
        return []

    metric = context_stats[0][0].metric

    # Take all the non-root contexts that are statistically significant
    if metric.dataType == Metric.DATATYPE_REG:
        filtered_bugs = [(c, c_stats) for (c, c_stats) in context_stats if
                         not c.isroot and c.metric.abs_effect() > 0]
    else:
        if isinstance(context_stats[0][1], pd.DataFrame):
            filtered_bugs = [(c, c_stats) for (c, c_stats) in context_stats
                             if not c.isroot and
                             np.array(c_stats)[0][-1] <= 1-conf]
        else:
            filtered_bugs = [(c, c_stats) for (c, c_stats) in context_stats if
                             not c.isroot and c_stats[-1] <= 1-conf]


    logging.info('%d statistically significant sub-contexts'
                 % len(filtered_bugs))

    if filtered_bugs:
        logging.info('Size range: %d-%d'
                     % (min([c.size for (c,_) in filtered_bugs]),
                        max([c.size for (c,_) in filtered_bugs])))

    if node_filter == FILTER_LEAVES_ONLY:
        filtered_bugs = [(c, c_stats) for (c, c_stats) in context_stats if
                         c.isleaf]

    #
    # Only keep sub-populations that lead to a better bias
    #
    if node_filter == FILTER_BETTER_THAN_ANCESTORS:
        effects = {}
        for context, _ in context_stats:
            if context.parent is None:
                effects[context.num] = context.metric.abs_effect()
            else:
                effects[context.num] = max(context.metric.abs_effect(),
                                           effects[context.parent.num])
        filtered_bugs = [(c, c_stats) for (c, c_stats) in filtered_bugs if
                         c.metric.abs_effect() >= effects[c.num] and
                         c.metric.abs_effect() > 0]

    # sort by effect size
    filtered_bugs.sort(key=lambda (c, stats): c.metric.abs_effect(),
                       reverse=True)

    filtered_bugs = [(c, c_stats) for (c, c_stats) in filtered_bugs if
                         c_stats[2] < 0.05]

    logging.info('%d sub-contexts printed' % len(filtered_bugs))
    return filtered_bugs
