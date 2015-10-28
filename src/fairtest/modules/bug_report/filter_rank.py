"""
Filter and Rank Association Bugs.
"""
from fairtest.modules.metrics import Metric

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

    metric = context_stats[0][0].metric

    # Take all the non-root contexts that are statistically significant
    if metric.dataType == Metric.DATATYPE_REG:
        filtered_bugs = [(c, c_stats) for (c, c_stats) in context_stats if
                         not c.isroot and c.metric.abs_effect() > 0]
    else:
        filtered_bugs = [(c, c_stats) for (c, c_stats) in context_stats if
                         not c.isroot and c_stats[-1] <= 1-conf]

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

    return filtered_bugs
