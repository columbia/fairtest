"""
Module for displaying clusters
"""
from StringIO import StringIO
import prettytable
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import pandas as pd
import numpy as np
import subprocess
import textwrap
import os
from fairtest.bugreport.statistics import fairness_measures as fm
from fairtest.bugreport.clustering import tree_clustering as tc

# Filters
FILTER_LEAVES_ONLY = 'LEAVES_ONLY'
FILTER_ALL = 'ALL'
FILTER_ROOT_ONLY = 'ROOT ONLY'
FILTER_BETTER_THAN_ANCESTORS = "BETTER_THAN_ANCESTORS"
NODE_FILTERS = [FILTER_ALL, FILTER_LEAVES_ONLY,
                FILTER_ROOT_ONLY, FILTER_BETTER_THAN_ANCESTORS]

# Sorting Method
SORT_BY_EFFECT = 'EFFECT'
SORT_BY_SIG = 'SIGNIFICANCE'
SORT_METHODS = [SORT_BY_EFFECT, SORT_BY_SIG]


class Namer:
    def __init__(self, sens, expl, output, encoders):
        self.encoders = encoders
        self.sens = sens
        self.expl = expl
        self.output = output

    def get_feature_val(self, feature, cat):
        if self.encoders:
            if feature in self.encoders:
                return self.encoders[feature].inverse_transform([cat])[0]
        return cat

    def get_sens_feature_vals(self, default):
        if self.encoders:
            if self.sens in self.encoders:
                return self.encoders[self.sens].classes_
        return range(default)

    def get_expl_feature_vals(self, default):
        if self.encoders:
            if self.expl in self.encoders:
                return self.encoders[self.expl].classes_
        return range(default)

    def get_target_vals(self, target, default):
        if self.encoders:
            if target in self.encoders:
                return self.encoders[target].classes_
        return range(default)


def print_summary(all_clusters, displ_clusters, namer):
    """
    Hierarchical printing of context

    Parameters
    -----------
    all_clusters :
        list of all clusters

    displ_clusters :
        list of all clusters that were displayed
    """
    print "Hierarchical printing of subpopulations (summary)"
    print
    print "="*80
    print

    root = filter(lambda c: c.isroot, all_clusters)[0]

    def recurse(node, indent):
        if node.num in displ_clusters:
            if node.clstr_measure.dataType != fm.Measure.DATATYPE_REG:
                print '{} Context = {} ; CI = [{:.4f}, {:.4f}]'.\
                    format(' '*indent, print_context(node.path, namer),
                           node.clstr_measure.stats[0],
                           node.clstr_measure.stats[1])
            else:
                print '{} Context = {} ; Avg Effect = {:.4f}'.\
                    format(' '*indent, node.path,
                           node.clstr_measure.abs_effect())
            indent += 2
        for child in node.children:
            recurse(child, indent)

    recurse(root, 0)

    print '-'*80
    print


def bug_report(clusters, stats, sens, expl, output, sort_by=SORT_BY_EFFECT,
               node_filter=FILTER_LEAVES_ONLY, encoders=None):
    """
    Print all the clusters sorted by relevance

    Parameters
    -----------
    clusters :
        list of all clusters

    sort_by :
        way to sort the clusters ('effect' => sort by lower bound
        on the mutual information score; 'sig' => sort by significance level)

    node_filter :
        way to sort the clusters

    approx :
        whether to use approximate asymptotic statistical measures
        or exact methods

    fdr :
        false discovery rate
    """
    #assert sort_by in SORT_METHODS
    #assert node_filter in NODE_FILTERS

    measure = clusters[0].clstr_measure
    measure_type = measure.dataType
    zipped = zip(clusters, stats)

    namer = Namer(sens, expl, output, encoders)

    # print global stats (root of tree)
    (root, root_stats) = filter(lambda (c, _): c.isroot, zipped)[0]
    print 'Global Population of size {}'.format(root.size)
    print

    if len(root_stats) == 3:
        (_, root_effect_high, _) = root_stats

    # print a contingency table or correlation analysis
    if measure_type == fm.Measure.DATATYPE_CT:
        print_cluster_ct(root, root_stats, measure.__class__.__name__, namer)
    elif measure_type == fm.Measure.DATATYPE_CORR:
        print_cluster_corr(root, root_stats, measure.__class__.__name__, namer)
    else:
        print_cluster_reg(root, root_stats,
                          measure.__class__.__name__, namer, sort_by=sort_by)
    print '='*80
    print

    # Take all the non-root clusters
    zipped = filter(lambda (c, _): not c.isroot, zipped)

    #
    # Only keep sub-populations that lead to a better bias
    #
    if node_filter == FILTER_BETTER_THAN_ANCESTORS:
        effects = {}
        for cluster in clusters:
            if cluster.parent is None:
                effects[cluster.num] = cluster.clstr_measure.abs_effect()
            else:
                effects[cluster.num] = max(cluster.clstr_measure.abs_effect(),
                                           effects[cluster.parent.num])
        zipped = filter(lambda (c, _):
                        c.clstr_measure.abs_effect() >= effects[c.num] and
                        c.clstr_measure.abs_effect() > 0, zipped)

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

    displ_clusters = [root.num] + [x[0].num for x in zipped]

    # print clusters in order of relevance
    for (cluster, cluster_stats) in zipped:

        print 'Sub-Population of size {}'.format(cluster.size)
        print 'Context = {}'.format(print_context(cluster.path, namer))
        print

        if measure_type == fm.Measure.DATATYPE_CT:
            print_cluster_ct(cluster, cluster_stats,
                             measure.__class__.__name__, namer)
        elif measure_type == fm.Measure.DATATYPE_CORR:
            print_cluster_corr(cluster, cluster_stats,
                               measure.__class__.__name__, namer)
        else:
            print_cluster_reg(cluster, cluster_stats,
                              measure.__class__.__name__, namer, sort_by=sort_by)
        print '-'*80
        print

    print_summary(clusters, displ_clusters, namer)


def print_context(path, namer):
    new_path = {
        key: val if isinstance(val, tc.Bound) else
        namer.get_feature_val(key, val) for (key, val) in path.iteritems()
        }

    return new_path


def print_cluster_ct(cluster, cluster_stats, effect_name, namer):
    """
    pretty-print the contingency table

    Parameters
    ----------
    cluster :
        List of all clusters

    cluster_stats :
        Statistics of the cluster

    effect_name :
        The effect to sort by
    """
    out = namer.output.names[0]

    if not namer.expl:
        ct = cluster.stats
        output = StringIO()
        ct.index = namer.get_target_vals(out, len(ct.index))
        ct.index.name = out
        ct.columns = namer.get_sens_feature_vals(len(ct.columns))
        ct.columns.name = namer.sens
        rich_ct(ct).to_csv(output)
        output.seek(0)
        pretty_table = prettytable.from_csv(output)
        print pretty_table
        print
    else:
        expl_values = namer.get_expl_feature_vals(len(cluster_stats))
        for i in range(len(cluster.stats)):
            if cluster.stats[i].sum() > 0:
                size = cluster.stats[i].sum()
                weight = (100.0*size)/cluster.size
                print '{} = {} ({} - {:.2f}%):'.\
                    format(namer.expl, expl_values[i], size, weight)

                ct_measure = fm.NMI(ci_level=cluster.clstr_measure.ci_level)
                (effect_low, effect_high, p_val) = \
                        ct_measure.compute(cluster.stats[i]).stats
                print '{} (non-adjusted) = [{:.4f}, {:.4f}]'.\
                        format('MI', effect_low, effect_high)

                ct = pd.DataFrame(cluster.stats[i])
                ct.index = namer.get_target_vals(out, len(ct.index))
                ct.index.name = out
                ct.columns = namer.get_sens_feature_vals(len(ct.columns))
                ct.columns.name = namer.sens
                output = StringIO()
                rich_ct(ct).to_csv(output)
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


# type of correlation plot ("BOXPLOT", "JITTER" or "HEXBIN")
CORR_PLOT = 'BOXPLOT'


def print_cluster_corr(cluster, cluster_stats, effect_name, namer):
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

    # avoid matplotlib overflow
    if len(out) > 100000:
        (out, sens) = zip(*random.sample(zip(out, sens), 100000))
        out = np.array(out)
        sens = np.array(sens)

    m, b = np.polyfit(sens, out, 1)
    plt.plot(sens, m*sens + b, '-', color='green')

    if CORR_PLOT == 'JITTER':
        jitter(sens, out, color='blue', edgecolor='none')
    elif CORR_PLOT == 'HEXBIN':
        plt.hexbin(sens, out, gridsize=20, norm=colors.LogNorm(),
                   cmap=plt.get_cmap('Blues'))
        plt.colorbar()
    elif CORR_PLOT == 'BOXPLOT':
        grouped = data.groupby(data.columns[1])
        keys = [key for (key, group) in grouped]
        groups = [group[data.columns[0]].values for (key, group) in grouped]
        min_key_diff = min([keys[i + 1]-keys[i] for i in xrange(len(keys)-1)])
        plt.boxplot(groups, positions=keys, widths=(1.0*min_key_diff)/2)

    plt.xlabel(data.columns[1])
    plt.ylabel(data.columns[0])
    plt.xlim(np.min(sens)-0.2*np.std(sens), np.max(sens)+0.2*np.std(sens))
    plt.ylim(np.min(out)-0.2*np.std(out), np.max(out)+0.2*np.std(out))
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


def print_cluster_reg(cluster, stats, effect_name, namer, sort_by='effect'):
    """
    Print regression stats

    Parameters
    ----------
    cluster :
        List of all clusters

    stats :
        Statistics of the cluster

    effect_name :
        The effect to sort by

    sort_by :
        The way regression coefficients should be sorted
    """
    effect = cluster.clstr_measure.abs_effect()

    print 'Average MI of top-{} labels: {}'.format(len(stats), effect)
    print
    labels = namer.output.names

    stats = stats.copy()
    stats.index = map(lambda idx: labels[idx], stats.index.values)

    if sort_by == SORT_BY_EFFECT:
        if 'conf low' in stats.columns:
            sorted_results = stats.sort(columns=['conf low'], ascending=False)
        else:
            sorted_results = stats.sort(columns=['coeff'], ascending=False)
    else:
        sorted_results = stats.sort(columns=['p-value'], ascending=True)

    pd.set_option('display.max_rows', cluster.clstr_measure.topK)
    print sorted_results
    print

    cluster_data = cluster.data['data_node']
    sens = namer.sens

    for label in sorted_results.index:
        ct = pd.DataFrame(0, index=cluster_data[label].unique(), columns=[0,1])
        # fill in available values
        ct = ct.add(pd.crosstab(np.array(cluster_data[label]),
                                np.array(cluster_data[sens])),
                    fill_value=0)

        # replace numbers by original labels
        ct.index.name = label
        ct.columns = namer.get_sens_feature_vals(2)
        ct.columns.name = sens

        output = StringIO()
        rich_ct(ct).to_csv(output)
        output.seek(0)
        pretty_table = prettytable.from_csv(output)

        print pretty_table
        print


def rich_ct(contingency_table):
    """
    Build a rich contingency table with proportions and marginals

    Parameters
    ----------
    contingency_table : the contingency table

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
    if os.path.exists('../.git'):
        print 'Commit Hash: \t{}'.format(
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='..').strip())
    else:
        print 'Commit Hash: Not A Git Repository'
    print
    print 'Dataset: \t{}'.format(data.filepath)
    print 'Training Size: \t{}'.format(len(data.data_train))
    print 'Testing Size: \t{}'.format(len(data.data_test))
    print 'Attributes: \t{}'.format("\n\t\t".join(
        textwrap.wrap(str(data.features.tolist()), 60)))
    print 'Protected: \t{}'.format(data.sens)
    print 'Explanatory: \t{}'.format(data.expl)
    print 'Target: \t{}'.format(data.out)
    print
    print 'Tree Params: \t{}'.format(tree_params)
    print 'Metric: \t{}'.format(measure)
    print
    print 'Report Params: \t{}'.format(display_params)
    print '='*80
    print
