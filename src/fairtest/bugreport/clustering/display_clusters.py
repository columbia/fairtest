from StringIO import StringIO
import prettytable
from statsmodels.sandbox.stats.multicomp import multipletests
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import numpy as np

from fairtest.bugreport.statistics import fairness_measures as fm
from fairtest.bugreport.statistics.fairness_measures import NMI

# Filters
FILTER_LEAVES_ONLY = 0
FILTER_ALL = 1
NODE_FILTERS = [FILTER_ALL, FILTER_LEAVES_ONLY]

# Sorting Method
SORT_BY_EFFECT = 0
SORT_BY_SIG = 1
SORT_METHODS = [SORT_BY_EFFECT, SORT_BY_SIG]


#
# Print all the clusters sorted by relevance
#
# @args clusters    list of all clusters
# @args sort_by     way to sort the clusters ('effect' => sort by lower bound
#                   on the mutual information score; 'sig' => sort by significance
#                   level)
# @args leaves_only consider tree leaves only
# @args conf_level  level for confidence intervals
#
def bug_report(clusters, columns=None, measure=NMI(ci_level=0.95), sort_by=SORT_BY_EFFECT, node_filter=FILTER_LEAVES_ONLY, fdr=None):
    #assert isinstance(measure, fm.Measure)
    assert sort_by in SORT_METHODS
    assert node_filter in NODE_FILTERS

    # compute effect sizes and p-values
    stats = map(lambda c: measure.compute(c.stats), clusters)

    # For regression, we have multiple p-values per cluster (one per topK coefficient)
    if isinstance(measure, fm.REGRESSION):
        stats = pd.concat(stats)
        stats_index = stats.index
        stats_columns = stats.columns
        stats = stats.values

    # correct p-values
    if fdr:
        pvals = map(lambda (low, high, p): max(p, 1e-180), stats)
        _, pvals_corr, _, _ = multipletests(pvals, alpha=fdr, method='holm')
        stats = [measure.ci_from_p(low, high, pval_corr) for ((low, high, _), pval_corr) in zip(stats, pvals_corr)]

    # For regression, re-form the dataframes for each cluster
    if isinstance(measure, fm.REGRESSION):
        stats = pd.DataFrame(stats, index=stats_index, columns=stats_columns)
        stats = np.array_split(stats, len(stats)/measure.topK)

    zipped = zip(clusters, stats)
    
    # Filter the clusters to show (All or Leaves & Root)
    if node_filter == FILTER_LEAVES_ONLY:
        zipped = filter(lambda (c, stat): c.isleaf or c.isroot, zipped)
    
    # print global stats (root of tree)
    (root, root_stats) = filter(lambda (c, _): c.isroot, zipped)[0]
    print 'Global Population of size {}'.format(root.size)
    print
    
    # print a contingency table or correlation analysis
    if measure.dataType == fm.Measure.DATATYPE_CT:
        print_cluster_ct(root, columns, root_stats, measure.__class__.__name__)
    elif measure.dataType == fm.Measure.DATATYPE_CORR:
        print_cluster_corr(root, root_stats, measure.__class__.__name__)
    else:
        print_cluster_reg(root, root_stats, measure.__class__.__name__, sort_by=sort_by)
    print '='*80
    print

    # Take all the non-root clusters
    zipped = filter(lambda (c, _): not c.isroot, zipped)

    # sort by significance
    if sort_by == SORT_BY_SIG:
        if isinstance(measure, fm.REGRESSION):
            # for regression, we sort by the p-value of the most significant coefficient
            zipped.sort(key=lambda (c, stats): min(stats['p-value']))
        else:
            zipped.sort(key=lambda (c, (low, high, p)): p)

    # sort by effect-size
    elif sort_by == SORT_BY_EFFECT:
        # get a normalized effect size and sort
        zipped.sort(key=lambda (c, stats): measure.normalize_effect(stats), reverse=True)

    # print clusters in order of relevance    
    for (cluster, cluster_stats) in zipped:
        print 'Sub-Population of size {}'.format(cluster.size)
        print 'Context = {}'.format(cluster.path)
        print

        if measure.dataType == fm.Measure.DATATYPE_CT:
            print_cluster_ct(cluster, columns, cluster_stats, measure.__class__.__name__)
        elif measure.dataType == fm.Measure.DATATYPE_CORR:
            print_cluster_corr(cluster, cluster_stats, measure.__class__.__name__)
        else:
            print_cluster_reg(cluster, cluster_stats, measure.__class__.__name__, sort_by=sort_by)
        print '-'*80
        print


def print_cluster_ct(cluster, columns, cluster_stats, effect_name):
    # pretty-print the contingency table
    output = StringIO()
    
    if columns:
        ct = cluster.stats[columns]
    else:
        ct = cluster.stats
    
    rich_ct(ct).to_csv(output)
    output.seek(0)
    pt = prettytable.from_csv(output)
    print pt
    print 

    (effect_low, effect_high, p_val) = cluster_stats
    # print p-value and confidence interval of MI
    print 'p-value = {:.2e} ; {} = [{:.4f}, {:.4f}]'.format(p_val, effect_name, effect_low, effect_high)


def print_cluster_corr(cluster, cluster_stats, effect_name):
    # print correlation graph
    data = cluster.data['values']

    out = data[data.columns[0]]
    sens = data[data.columns[1]]

    #plt.scatter(sens, out, color='blue', edgecolor='none')
    m, b = np.polyfit(sens, out, 1)
    plt.plot(sens, m*sens + b, '-', color='green')

    plt.hexbin(sens, out, gridsize=20, cmap=plt.get_cmap('Blues'))
    plt.colorbar()

    plt.xlabel(data.columns[1])
    plt.ylabel(data.columns[0])
    plt.show()

    (effect_low, effect_high, p_val) = cluster_stats
    # print p-value and confidence interval of correlation
    print 'p-value = {:.2e} ; {} = [{:.4f}, {:.4f}]'.format(p_val, effect_name, effect_low, effect_high)


def print_cluster_reg(cluster, stats, effect_name, sort_by='effect'):
    effect = fm.REGRESSION(ci_level=1.0).normalize_effect(stats)
    print 'Average (absolute) Log-Odds of top-{} labels: {}'.format(len(stats), effect)
    print
    labels = cluster.data['labels']

    stats.index = map(lambda idx: labels[idx], stats.index.values)

    if sort_by == 'effect':
        stats['effect'] = map(lambda (ci_low, ci_high): fm.z_effect(ci_low, ci_high), zip(stats['conf low'], stats['conf high']))
        sorted_results = stats.sort(columns=['effect'], ascending=False)
        sorted_results.drop('effect', axis=1)
    else:
        sorted_results = stats.sort(columns=['p-value'], ascending=True)

    print sorted_results


#
# Build a rich contingency table with proportions and marginals
#     
# @args ct  the contingency table
#   
def rich_ct(ct):

    # for each output, add its percentage over each sensitive group
    temp = ct.copy().astype(object)
    for col in ct.columns:
        tot_col = sum(ct[col])
        for row in ct.index:
            val = ct.loc[row][col]
            percent = (100.0*val)/tot_col
            temp.loc[row][col] = '{} ({:.1f}%)'.format(val, percent)
    
    total = ct.sum().sum()
    
    # add marginals
    sum_col = ct.sum(axis=1)
    temp.insert(len(temp.columns), 'Total', map(lambda val: '{} ({:.1f}%)'.format(val, (100.0*val)/total) , sum_col))
    sum_row = ct.sum(axis=0)
    sum_row['Total'] = total
    temp.loc['Total'] = map(lambda val: '{} ({:.1f}%)'.format(val, (100.0*val)/total) , sum_row)
    temp.loc['Total']['Total'] = '{} (100.0%)'.format(total)
    
    return temp