"""
Module for discpaying clusters
"""
from StringIO import StringIO
import prettytable
from statsmodels.sandbox.stats.multicomp import multipletests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import subprocess
import textwrap
from copy import copy

from fairtest.bugreport.statistics import fairness_measures as fm

# Filters
FILTER_LEAVES_ONLY = 'LEAVES_ONLY'
FILTER_ALL = 'ALL'
FILTER_ROOT_ONLY = 'ROOT ONLY'
NODE_FILTERS = [FILTER_ALL, FILTER_LEAVES_ONLY, FILTER_ROOT_ONLY]

# Sorting Method
SORT_BY_EFFECT = 'EFFECT'
SORT_BY_SIG = 'SIGNIFICANCE'
SORT_METHODS = [SORT_BY_EFFECT, SORT_BY_SIG]


def bug_report(clusters, columns=None, sort_by=SORT_BY_EFFECT,
               node_filter=FILTER_LEAVES_ONLY, new_measure=None,
               approx=True, fdr=None):
    """
    Print all the clusters sorted by relevance

    Parameters
    -----------
    clusters :
        list of all clusters

    sort_by :
        way to sort the clusters ('effect' => sort by lower bound
        on the mutual information score; 'sig' => sort by significance level)

    leaves_only :
        consider tree leaves only

    conf_level :
        level for confidence intervals
    """
    assert sort_by in SORT_METHODS
    assert node_filter in NODE_FILTERS

    if fdr:
        fdr_alpha = 1-fdr
    else:
        fdr_alpha = None

    if new_measure:
        measure = new_measure
        for cluster in clusters:
            cluster.clstr_measure = copy(new_measure)
    else:
        measure = clusters[0].clstr_measure
    measure_type = measure.dataType

    # Filter the clusters to show (All or Leaves & Root)
    if node_filter == FILTER_LEAVES_ONLY:
        clusters = filter(lambda c: c.isleaf or c.isroot, clusters)
    elif node_filter == FILTER_ROOT_ONLY:
        clusters = filter(lambda c: c.isroot, clusters)

    #
    # Adjusted Confidence Interval (Bonferroni)
    #
    # compute effect sizes and p-values
    adj_ci_level = None
    if fdr_alpha and measure.ci_level:
        adj_ci_level = 1-(1-fdr_alpha)/len(clusters)
        #print 'Adjusted CI level is {:.4f}'.format(adj_ci_level)

    if measure.dataType == fm.Measure.DATATYPE_CORR:
        stats = map(lambda c: c.clstr_measure.\
                                    compute(c.stats,
                                            data=c.data['values'],
                                            approx=approx,
                                            adj_ci_level=adj_ci_level).stats,
                    clusters)
    else:
        stats = map(lambda c: c.clstr_measure.\
                                    compute(c.stats,
                                            approx=approx,
                                            adj_ci_level=adj_ci_level).stats,
                    clusters)

    # For regression, we have multiple p-values per cluster
    # (one per topK coefficient)
    if measure_type == fm.Measure.DATATYPE_REG:
        stats = pd.concat(stats)
        stats_index = stats.index
        stats_columns = stats.columns
        stats = stats.values

    # correct p-values
    if fdr_alpha:
        pvals = map(lambda stat: max(stat[-1], 1e-180), stats)
        _, pvals_corr, _, _ = multipletests(pvals,
                                            alpha=fdr_alpha,
                                            method='holm')
        #stats = [clstr.clstr_measure.ci_from_p(low, high, pval_corr)
        # for ((clstr, (low, high, _)), pval_corr) in zip(zipped, pvals_corr)]
        stats = [np.append(stat[0:-1], pval_corr)\
                for (stat, pval_corr) in zip(stats, pvals_corr)]

    # For regression, re-form the dataframes for each cluster
    if measure_type == fm.Measure.DATATYPE_REG:
        stats = pd.DataFrame(stats, index=stats_index, columns=stats_columns)
        stats = np.array_split(stats, len(stats)/measure.topK)

    zipped = zip(clusters, stats)

    # print global stats (root of tree)
    (root, root_stats) = filter(lambda (c, _): c.isroot, zipped)[0]
    print 'Global Population of size {}'.format(root.size)
    print

    # print a contingency table or correlation analysis
    if measure_type == fm.Measure.DATATYPE_CT:
        print_cluster_ct(root, columns, root_stats, measure.__class__.__name__)
    elif measure_type == fm.Measure.DATATYPE_CORR:
        print_cluster_corr(root, root_stats, measure.__class__.__name__)
    else:
        print_cluster_reg(root, root_stats,
                          measure.__class__.__name__, sort_by=sort_by)
    print '='*80
    print

    # Take all the non-root clusters
    zipped = filter(lambda (c, _): not c.isroot, zipped)

    # sort by significance
    if sort_by == SORT_BY_SIG:
        if measure_type == fm.Measure.DATATYPE_REG:
            # for regression, we sort by the p-value of
            # the most significant coefficient
            zipped.sort(key=lambda (c, stats): min(stats['p-value']))
        else:
            zipped.sort(key=lambda (c, stats): stats[-1])

    # sort by effect-size
    elif sort_by == SORT_BY_EFFECT:
        # get a normalized effect size and sort
        zipped.sort(key=lambda (c, stats): c.clstr_measure.abs_effect(),
                    reverse=True)

    # print clusters in order of relevance
    for (cluster, cluster_stats) in zipped:
        print 'Sub-Population of size {}'.format(cluster.size)
        print 'Context = {}'.format(cluster.path)
        print

        if measure_type == fm.Measure.DATATYPE_CT:
            print_cluster_ct(cluster, columns, cluster_stats,
                             measure.__class__.__name__)
        elif measure_type == fm.Measure.DATATYPE_CORR:
            print_cluster_corr(cluster, cluster_stats,
                               measure.__class__.__name__)
        else:
            print_cluster_reg(cluster, cluster_stats,
                              measure.__class__.__name__, sort_by=sort_by)
        print '-'*80
        print


def print_cluster_ct(cluster, columns, cluster_stats, effect_name):
    """
    pretty-print the contingency table

    Parameters
    ----------
    cluster :
        List of all clusters

    columns :
        Columns of contingency table

    cluster_stats :
        Statistics of the cluster

    effect_name :
        The effect to sort by
    """
    shape = cluster.stats.shape

    if columns:
        if len(shape) == 3:
            contingency_table = map(lambda c: c[columns], cluster.stats)
        else:
            contingency_table = cluster.stats[columns]
    else:
        contingency_table = cluster.stats

    if len(shape) == 2:
        output = StringIO()
        rich_ct(contingency_table).to_csv(output)
        output.seek(0)
        pretty_table = prettytable.from_csv(output)
        print pretty_table
        print
    else:
        (expl, encoder_enc) = cluster.data['expl']

        for i in range(len(encoder_enc.classes_)):
            if cluster.stats[i].values.sum() > 0:
                size = cluster.stats[i].values.sum()
                weight = (100.0*size)/cluster.size
                print '{} = {} ({} - {:.2f}%):'.\
                        format(expl, encoder_enc.classes_[i], size, weight)

                ct_measure = fm.NMI(ci_level=cluster.clstr_measure.ci_level)
                (effect_low, effect_high, p_val) = \
                        ct_measure.compute(cluster.stats[i]).stats
                print '{} (non-adjusted) = [{:.4f}, {:.4f}]'.\
                        format('MI', effect_low, effect_high)

                contingency_table = pd.DataFrame(cluster.stats[i])
                output = StringIO()
                rich_ct(contingency_table).to_csv(output)
                output.seek(0)
                pretty_table = prettytable.from_csv(output)
                print pretty_table
                print

    if len(cluster_stats) == 3:
        # print p-value and confidence interval of MI
        (effect_low, effect_high, p_val) = cluster_stats
        print 'p-value = {:.2e} ; {} = [{:.4f}, {:.4f}]'.\
                format(p_val, effect_name, effect_low, effect_high)
    else:
        effect, p_val = cluster_stats
        print 'p-value = {:.2e} ; {} = {:.4f}'.\
                format(p_val, effect_name, effect)


def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def jitter(x, y, s=20, c='b', marker='o', cmap=None,
           norm=None, vmin=None, vmax=None, alpha=None, linewidths=None,
           verts=None, hold=None, **kwargs):
    return plt.scatter(rand_jitter(x), y, s=20, c='b', marker='o',
                       cmap=None, norm=None, vmin=None, vmax=None, alpha=None,
                       linewidths=None, verts=None, hold=None, **kwargs)


def print_cluster_corr(cluster, cluster_stats, effect_name):
    """
    print correlation graph

    Parameters
    ----------
    cluster :
        List of all clusters

    cluster_stats :
        Statistics of the cluster

    effect_name :
        The effect to sort by
    """
    data = cluster.data['values']

    out = data[data.columns[0]]
    sens = data[data.columns[1]]

    #jitter(sens, out, color='blue', edgecolor='none')
    m, b = np.polyfit(sens, out, 1)
    plt.plot(sens, m*sens + b, '-', color='green')

    plt.hexbin(sens, out, gridsize=20, cmap=plt.get_cmap('Blues'))
    plt.colorbar()

    plt.xlabel(data.columns[1])
    plt.ylabel(data.columns[0])
    plt.show()

    if len(cluster_stats) == 3:
        # print p-value and confidence interval of correlation
        (effect_low, effect_high, p_val) = cluster_stats
        print 'p-value = {:.2e} ; {} = [{:.4f}, {:.4f}]'.\
                format(p_val, effect_name, effect_low, effect_high)
    else:
        effect, p_val = cluster_stats
        print 'p-value = {:.2e} ; {} = {:.4f}'.\
                format(p_val, effect_name, effect)


def print_cluster_reg(cluster, stats, effect_name, sort_by='effect'):
    """
    Print regression graph

    Parameters
    ----------
    cluster :
        List of all clusters

    columns :
        Columns of contingency table

    cluster_stats :
        Statistics of the cluster

    effect_name :
        The effect to sort by
    """
    effect = cluster.clstr_measure.abs_effect()

    print 'Average (absolute) Log-Odds of top-{} labels: {}'.\
            format(len(stats), effect)
    print
    labels = cluster.data['labels']

    stats.index = map(lambda idx: labels[idx], stats.index.values)

    if sort_by == SORT_BY_EFFECT:
        if 'conf low' in stats.columns:
            stats['effect'] = map(lambda (ci_low, ci_high): fm.z_effect(ci_low, ci_high),
                                  zip(stats['conf low'], stats['conf high']))
            sorted_results = stats.sort(columns=['effect'], ascending=False)
            sorted_results.drop('effect', axis=1)
        else:
            sorted_results = stats.sort(columns=['coeff'], ascending=False)
    else:
        sorted_results = stats.sort(columns=['p-value'], ascending=True)

    print sorted_results
    print

    cluster_data = cluster.data['data_node']
    sens = cluster.data['sens']
    encoder = cluster.data['encoder_sens']

    for label in sorted_results.index:
        contingency_table = pd.DataFrame(0, index=cluster_data[label].unique(),
                                         columns=range(len(encoder.classes_)))
        # fill in available values
        contingency_table = contingency_table.\
                add(pd.crosstab(cluster_data[label], cluster_data[sens]),
                    fill_value=0)

        # replace numbers by original labels
        contingency_table.index.name = label
        contingency_table.columns = encoder.classes_
        contingency_table.columns.name = sens

        output = StringIO()
        rich_ct(contingency_table).to_csv(output)
        output.seek(0)
        pretty_table = prettytable.from_csv(output)

        print prettytable
        print


def rich_ct(contingency_table):
    """
    Build a rich contingency table with proportions and marginals

    Parameters
    ----------
    contingency_table :

    Returns
    -------
    temp :
        Enhanced contingency table

    """
    # for each output, add its percentage over each sensitive group
    temp = contingency_table.copy().astype(object)
    for col in contingency_table.columns:
        tot_col = sum(contingency_table[col])
        for row in contingency_table.index:
            val = contingency_table.loc[row][col]
            percent = (100.0*val)/tot_col
            temp.loc[row][col] = '{} ({:.1f}%)'.format(val, percent)

    total = contingency_table.sum().sum()

    # add marginals
    sum_col = contingency_table.sum(axis=1)
    temp.insert(len(temp.columns), 'Total',
                map(lambda val: '{} ({:.1f}%)'.\
                        format(val, (100.0*val)/total), sum_col))
    sum_row = contingency_table.sum(axis=0)
    sum_row['Total'] = total
    temp.loc['Total'] = map(lambda val: '{} ({:.1f}%)'.\
                                format(val, (100.0*val)/total), sum_row)
    temp.loc['Total']['Total'] = '{} (100.0%)'.format(total)

    return temp


def print_report_info(data, measure, tree_params, display_params):
    """
    Prints reports information
    """
    print '='*80
    print 'Commit Hash: \t{}'.format(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='..').strip())
    print
    print 'Dataset: \t{}'.format(data.filepath)
    print 'Training Size: \t{}'.format(len(data.data_train))
    print 'Testing Size: \t{}'.format(len(data.data_test))
    print 'Attributes: \t{}'.format("\n\t\t".join(textwrap.wrap(str(data.features.tolist()), 60)))
    print 'Protected: \t{}'.format(data.SENS)
    print 'Explanatory: \t{}'.format(data.EXPL)
    print 'Target: \t{}'.format(data.OUT)
    print
    print 'Tree Params: \t{}'.format(tree_params)
    print 'Metric: \t{}'.format(measure)
    print
    print 'Report Params: \t{}'.format(display_params)
    print '='*80
    print
