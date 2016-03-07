"""
Report association bugs
"""
from StringIO import StringIO
import datetime
import prettytable
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rcParams
import random
import pandas as pd
import numpy as np
import subprocess
import textwrap
import os
import errno
import re
import pickle
from fairtest.modules.metrics import Metric
from fairtest.modules.context_discovery.tree_parser import Bound
import fairtest.modules.bug_report.filter_rank as filter_rank


class Namer(object):
    """
    Converter between encoded features and their original names
    """
    def __init__(self, sens, expl, output, encoders):
        self.encoders = encoders
        self.sens = sens
        self.expl = expl
        self.output = output

    def get_feature_val(self, feature, cat):
        """
        Gets the original name of a categorical feature value

        Parameters
        -----------
        feature :
            the categorical feature

        cat :
            the encoded value

        Returns
        -------
        val :
            the original value if found, or the provided encoded value

        """
        if self.encoders:
            if feature in self.encoders:
                return self.encoders[feature].inverse_transform([cat])[0]
        return cat

    def get_sens_feature_vals(self, default):
        """
        Gets the original names of the categories of the sensitive feature

        Parameters
        -----------
        default :
            values to return if no encoding found

        Returns
        -------
        values :
            the original values if found, or the provided default values

        """
        if self.encoders:
            if self.sens in self.encoders:
                return self.encoders[self.sens].classes_
        return range(default)

    def get_expl_feature_vals(self, default):
        """
        Gets the original names of the categories of the explanatory feature

        Parameters
        -----------
        default :
            values to return if no encoding found

        Returns
        -------
        values :
            the original values if found, or the provided default values

        """
        if self.encoders:
            if self.expl in self.encoders:
                return self.encoders[self.expl].classes_
        return range(default)

    def get_target_vals(self, target, default):
        """
        Gets the original names of the categories of the target feature

        Parameters
        -----------
        default :
            values to return if no encoding found

        Returns
        -------
        values :
            the original values if found, or the provided default values

        """
        if self.encoders:
            if target in self.encoders:
                return self.encoders[target].classes_
        return range(default)


def print_summary(all_contexts, displayed_contexts, namer, output_stream):
    """
    Hierarchical printing of contexts

    Parameters
    -----------
    all_contexts :
        list of all contexts

    displayed_contexts :
        list of all contexts that were displayed

    namer :
        a Namer object to use to decode categorical features

    output_stream :
        stream to output the data to
    """
    print >> output_stream, "Hierarchical printing of subpopulations (summary)"
    print >> output_stream
    print >> output_stream, "="*80
    print >> output_stream

    root = [c for c in all_contexts if c.isroot][0]

    context_list = []

    def recurse(node, indent):
        """
        Recursively traverse tree structure and print out contexts

        Parameters
        ----------
        node :
            the current tree node

        indent :
            the current indentation
        """
        if node.num in displayed_contexts:
            if node.metric.dataType != Metric.DATATYPE_REG:
                if isinstance(node.metric.stats, pd.DataFrame):
                    (ci_low, ci_high, _) = node.metric.stats.loc[0]
                else:
                    (ci_low, ci_high, _) = node.metric.stats

                print >> output_stream, \
                    '{} Context = {} ; CI = [{:.4f}, {:.4f}] ; Size = {}'.\
                    format(' '*indent, print_context(node.path, namer),
                           ci_low, ci_high, node.size)
                context_list.append(print_context(node.path, namer))
            else:
                print >> output_stream, \
                    '{} Context = {} ; Avg Effect = {:.4f}'.\
                        format(' '*indent, print_context(node.path, namer),
                               node.metric.abs_effect())
                context_list.append(print_context(node.path, namer))
            indent += 2
        for child in node.children:
            recurse(child, indent)

    recurse(root, 0)

    print >> output_stream, '-'*80
    print >> output_stream

    return context_list


def bug_report(contexts, stats, sens, expl, output, output_stream,
               node_filter=filter_rank.FILTER_BETTER_THAN_ANCESTORS, conf=0.95,
               encoders=None, plot_dir=None):
    """
    Print all the association bugs sorted by effect size

    Parameters
    -----------
    contexts :
        list of all contexts

    stats :
        the statistics for all the contexts

    sens :
        the name of the sensitive feature

    expl :
        the name of the explanatory feature

    output :
        the target feature

    output_stream :
        stream to output the data to

    node_filter :
        method to use to filter contexts

    conf :
        confidence level for filtering

    encoders :
        data encoders used to numerize categorical features

    plot_dir :
        directory to save plots
    """

    metric = contexts[0].metric
    metric_type = metric.dataType
    contexts_stats = zip(contexts, stats)
    namer = Namer(sens, expl, output, encoders)

    # print global stats (root of tree)
    (root, root_stats) = [(c, stat) for (c, stat) in contexts_stats
                          if c.isroot][0]
    print >> output_stream, 'Global Population {} of size {}'.format(root.num,
                                                                     root.size)
    print >> output_stream

    # print a contingency table, correlation analysis or regression summary
    if metric_type == Metric.DATATYPE_CT:
        print_context_ct(root, root_stats, metric.__class__.__name__, namer,
                         output_stream)
    elif metric_type == Metric.DATATYPE_CORR:
        print_context_corr(root, root_stats, metric.__class__.__name__, namer,
                           output_stream, plot_dir)
    else:
        print_context_reg(root, root_stats, namer, output_stream)
    print >> output_stream, '='*80
    print >> output_stream

    # filter and rank bugs
    ranked_bugs = filter_rank.filter_rank_bugs(contexts_stats,
                                               node_filter=node_filter,
                                               conf=conf)

    # print contexts in order of relevance
    for (context, context_stats) in ranked_bugs:

        print >> output_stream, \
            'Sub-Population {} of size {}'.format(context.num, context.size)
        print >> output_stream, \
            'Context = {}'.format(print_context(context.path, namer))
        print >> output_stream

        if metric_type == Metric.DATATYPE_CT:
            print_context_ct(context, context_stats,
                             metric.__class__.__name__, namer, output_stream)
        elif metric_type == Metric.DATATYPE_CORR:
            print_context_corr(context, context_stats,
                               metric.__class__.__name__, namer, output_stream,
                               plot_dir)
        else:
            print_context_reg(context, context_stats, namer, output_stream)
        print >> output_stream, '-'*80
        print >> output_stream

    # get the list of bugs that are displayed
    displayed_bugs = [root.num] + [x[0].num for x in ranked_bugs]
    return print_summary(contexts, displayed_bugs, namer, output_stream)


def print_context(path, namer):
    """
    Prints a description of an association context.

    Parameters
    -----------
    path :
        the predicate path for this context

    namer:
        a Namer object to decode categorical feature values

    Returns
    -------
    new_path :
        the path with categorical features decoded
    """
    new_path = {
        key: val if isinstance(val, Bound) else namer.get_feature_val(key, val)
        for (key, val) in path.iteritems()}

    return new_path


def print_context_ct(context, context_stats, metric_name, namer, output_stream):
    """
    Print an association context based on a contingency table

    Parameters
    ----------
    context :
        the context

    context_stats :
        statistics of the context

    metric_name :
        the name of the metric

    namer :
        a Namer object to decode categorical feature values

    output_stream :
        the stream to output data to
    """
    out = namer.output.names[0]

    if not namer.expl:
        ct = context.data
        ct.index = namer.get_target_vals(out, len(ct.index))
        ct.index.name = out
        ct.columns = namer.get_sens_feature_vals(len(ct.columns))
        ct.columns.name = namer.sens
        print >> output_stream, pretty_ct(ct)
        print >> output_stream
    else:
        expl_values = namer.get_expl_feature_vals(len(context_stats))
        for i in range(len(context.data)):
            if context.data[i].sum() > 0:
                size = context.data[i].sum()
                weight = (100.0*size)/context.size
                print >> output_stream, '> {} = {} ; size {} ({:.2f}%):'.\
                    format(namer.expl, expl_values[i], size, weight)

                (effect_low, effect_high, p_val) = context_stats.loc[i+1]
                print >> output_stream, \
                    'p-value = {:.2e} ; {} = [{:.4f}, {:.4f}]'.\
                    format(p_val, 'DIFF', effect_low, effect_high)

                ct = pd.DataFrame(context.data[i])
                ct.index = namer.get_target_vals(out, len(ct.index))
                ct.index.name = out
                ct.columns = namer.get_sens_feature_vals(len(ct.columns))
                ct.columns.name = namer.sens
                print >> output_stream, pretty_ct(ct)
                print >> output_stream

        context_stats = context_stats.loc[0]

    # print p-value and confidence interval of MI
    (effect_low, effect_high, p_val) = context_stats
    print >> output_stream, 'p-value = {:.2e} ; {} = [{:.4f}, {:.4f}]'.\
        format(p_val, metric_name, effect_low, effect_high)


def rand_jitter(data):
    """
    Adds random jitter to data for nicer correlation plots

    Parameters
    ----------
    data :
        the data to perturb

    Returns
    -------
    noisy_data :
        the data with random noise added
    """
    stdev = .01*(max(data)-min(data))
    return data + np.random.randn(len(data)) * stdev


def jitter(x, y, **kwargs):
    """
    Draws a scatter plot with random jitter

    Parameters
    ----------
    x :
        first axis of the data
    y :
        second axis of the data
    **kwargs :
        additional arguments for the scatter plot
    """
    return plt.scatter(rand_jitter(x), y, **kwargs)


# type of correlation plot ("BOXPLOT", "JITTER" or "HEXBIN")
CORR_PLOT = 'BOXPLOT'


def mkdir_p(path):
    """
    Creates a directory recursively creating any missing directories on the path

    Parameters
    ----------
    path :
        the directory path
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def print_context_corr(context, context_stats, metric_name, namer,
                       output_stream, plot_dir):
    """
    Print an association context based on correlation

    Parameters
    ----------
    context :
        the context

    context_stats :
        statistics of the context

    metric_name :
        the name of the metric

    namer :
        a Namer object to decode categorical feature values

    output_stream :
        the stream to output data to

    plot_dir :
        directory to be used to store plots
    """

    if not namer.expl:
        data = context.data

        out = data[data.columns[0]]
        sens = data[data.columns[1]]

        rcParams.update({'figure.autolayout': True})

        # avoid matplotlib overflow
        if len(out) > 100000:
            (out, sens) = zip(*random.sample(zip(out, sens), 100000))
            out = np.array(out)
            sens = np.array(sens)

        if plot_dir:
            try:
                mkdir_p(plot_dir)
            except OSError:
                # directory already exists
                pass
            plot_name = os.path.join(plot_dir,
                                     'context_{}.png'.format(context.num))
        else:
            plot_name = None

        fig = plt.figure()
        m, b = np.polyfit(sens, out, 1)
        plt.plot(sens, m*sens + b, '-', color='green', linewidth=3)

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

            if len(keys) < 2:
                min_key_diff = 1
            else:
                min_key_diff = min([keys[i + 1]-keys[i]
                                    for i in xrange(len(keys)-1)])

            if len(keys) > 10:
                fig.set_size_inches(20, 10)

            plt.boxplot(groups, positions=keys, widths=(1.0*min_key_diff)/2,
                        sym='')

        plt.rcParams.update({'font.size': 22})
        if namer.sens in namer.encoders:
            ax = plt.gca()
            ax.set_xticklabels(namer.get_sens_feature_vals(
                len(data[data.columns[1]].unique())))
        else:
            plt.xlim(np.min(sens) - 0.4*np.std(sens),
                     np.max(sens) + 0.4*np.std(sens))
            plt.ylim(np.min(out) - 0.4*np.std(out),
                     np.max(out) + 0.4*np.std(out))
        plt.xlabel(data.columns[1])
        plt.ylabel(data.columns[0])

        if plot_name:
            pickle.dump(fig, file(os.path.splitext(plot_name)[0]+'.pkl', 'w'))
            plt.savefig(plot_name)
            plt.close(fig)
        else:
            plt.show()
    else:
        context_stats = context_stats.values
        expl_values = namer.get_expl_feature_vals(len(context_stats))
        for i in range(len(context.data)):
            if len(context.data[i]) > 0:
                size = len(context.data[i])
                weight = (100.0*size)/context.size
                print >> output_stream, '> {} = {} ; size {} ({:.2f}%):'.\
                    format(namer.expl, expl_values[i], size, weight)

                data = context.data[i]
                out = data[data.columns[0]]
                sens = data[data.columns[1]]

                fig = plt.figure()
                m, b = np.polyfit(sens, out, 1)
                plt.plot(sens, m*sens + b, '-', color='green', linewidth=3)

                rcParams.update({'figure.autolayout': True})

                grouped = data.groupby(data.columns[1])
                keys = [key for (key, group) in grouped]
                groups = [group[data.columns[0]].values
                          for (key, group) in grouped]
                min_key_diff = min([keys[_ + 1]-keys[_]
                                    for _ in xrange(len(keys)-1)])

                plt.boxplot(groups, positions=keys,
                            widths=(1.0*min_key_diff)/2, sym='')

                plt.rcParams.update({'font.size': 22})
                if namer.sens in namer.encoders:
                    ax = plt.gca()
                    ax.set_xticklabels(namer.get_sens_feature_vals(
                        len(data[data.columns[1]].unique())))
                else:
                    plt.xlim(np.min(sens) - 0.4*np.std(sens),
                             np.max(sens) + 0.4*np.std(sens))
                    plt.ylim(np.min(out) - 0.4*np.std(out),
                             np.max(out) + 0.4*np.std(out))

                plt.xlabel(data.columns[1])
                plt.ylabel(data.columns[0])
                plt.show()

            (effect_low, effect_high, p_val) = context_stats[i+1]
            print >> output_stream, \
                'p-value = {:.2e} ; {} = [{:.4f}, {:.4f}]'.\
                format(p_val, 'CORR', effect_low, effect_high)
            print >> output_stream

        context_stats = context_stats[0]

    # print p-value and confidence interval of correlation
    (effect_low, effect_high, p_val) = context_stats
    print >> output_stream, 'p-value = {:.2e} ; {} = [{:.4f}, {:.4f}]'.\
        format(p_val, metric_name, effect_low, effect_high)


def print_context_reg(context, context_stats, namer, output_stream):
    """
    Print an association context based on regression

    Parameters
    ----------
    context :
        the context

    context_stats :
        statistics of the context

    namer :
        a Namer object to decode categorical feature values

    output_stream :
        the stream to output data to
    """
    effect = context.metric.abs_effect()

    print >> output_stream, \
        'Average Effect of top-{} labels: {}'.format(len(context_stats), effect)
    print >> output_stream
    labels = namer.output.names

    stats = context_stats.copy()
    stats.index = [labels[idx] for idx in stats.index.values]

    # sort the labels by p-value as all the tests were performed on the same
    # sample size
    sorted_results = stats.sort(columns=['pval'], ascending=True)

    pd.set_option('display.max_rows', context.metric.topk)
    print >> output_stream, sorted_results
    print >> output_stream

    context_data = context.additional_data['data_node']
    sens = namer.sens

    for label in sorted_results.index:
        ct = pd.DataFrame(0, index=context_data[label].unique(), columns=[0, 1])
        # fill in available values
        ct = ct.add(pd.crosstab(np.array(context_data[label]),
                                np.array(context_data[sens])),
                    fill_value=0)

        # replace numbers by original labels
        ct.index.name = re.sub(r'\W+', ' ', label)
        ct.columns = namer.get_sens_feature_vals(2)
        ct.columns.name = sens

        print >> output_stream, pretty_ct(ct)
        print >> output_stream


def pretty_ct(ct):
    """
    Pretty-print a contingency table

    Parameters
    ----------
    ct :
        the contingency table

    Returns
    -------
    pretty_table :
        a fancier string representation of the table
    """
    output = StringIO()
    rich_ct(ct).to_csv(output)
    output.seek(0)
    pretty_table = prettytable.from_csv(output)
    pretty_table.padding_width = 0
    pretty_table.align = 'r'
    pretty_table.align[pretty_table.field_names[0]] = 'l'
    return pretty_table


def rich_ct(contingency_table):
    """
    Build a rich contingency table with proportions and marginals

    Parameters
    ----------
    contingency_table :
        the contingency table

    Returns
    -------
    rich_ct :
        enhanced contingency table

    """
    total = contingency_table.sum().sum()

    # for each output, add its percentage over each sensitive group
    rich_table = contingency_table.copy().astype(object)
    for col in contingency_table.columns:
        tot_col = sum(contingency_table[col])

        # largest percentage in the column
        max_percent_len = len('{:.0f}'.format((100.0*tot_col)/total))
        for row in contingency_table.index:
            val = contingency_table.loc[row][col]
            percent = (100.0*val)/tot_col
            percent_len = len('{:.0f}'.format(percent))
            delta = max_percent_len - percent_len
            rich_table.loc[row][col] = '{}{}({:.0f}%)'.\
                format(val, ' '*delta, percent)

    # add marginals
    sum_col = contingency_table.sum(axis=1)
    lens = [len('{:.0f}'.format((100.0*val)/total)) for val in sum_col]

    rich_table.insert(len(rich_table.columns), 'Total',
                      ['{}{}({:.0f}%)'.format(val, ' '*(3-l), (100.0*val)/total)
                       for (val, l) in zip(sum_col, lens)])
    sum_row = contingency_table.sum(axis=0)
    sum_row['Total'] = total
    rich_table.loc['Total'] = ['{}({:.0f}%)'.format(val, (100.0*val)/total)
                               for val in sum_row]
    rich_table.loc['Total']['Total'] = '{}(100%)'.format(total)

    return rich_table


def print_report_info(dataset, train_size, test_size, sensitive, contextual,
                      expl, target, train_params, test_params, display_params,
                      output_stream):
    """
    Prints report information.

    Parameters
    -----------
    dataset :
        the dataset used

    train_size :
        size of the training set

    test_size :
        size of the testing set

    sensitive :
        list of sensitive features

    contextual :
        list of contextual features

    expl :
        explanatory feature

    target :
        the target feature(s)

    train_params :
        parameters used for training

    test_params :
        parameters used for testing

    display_params :
        parameters used for reporting

    output_stream :
        stream to output the data to
    """
    print >> output_stream, '='*80
#    if os.path.exists('../.git'):
#        print >> output_stream, 'Commit Hash: \t{}'.\
#            format(subprocess.check_output(['git', 'rev-parse', 'HEAD'],
#                                           cwd='..').strip())
#    else:
#        print >> output_stream, 'Commit Hash: Not A Git Repository'
    print >> output_stream, 'Report Creation time:',\
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print >> output_stream
    print >> output_stream, 'Dataset: {}'.format(dataset)
    print >> output_stream, 'Train Size: {}'.format(train_size)
    print >> output_stream, 'Test Size: {}'.format(test_size)
    print >> output_stream, 'S: {}'.format(sensitive)
    print >> output_stream, 'X: {}'.format("\n\t".join(
        textwrap.wrap(str(contextual), 60)))
    print >> output_stream, 'E: {}'.format(expl)
    if len(target) > 10:
        target = '[{} ... {}]'.format(target[0], target[-1])
    print >> output_stream, 'O: {}'.format(target)
    print >> output_stream
    print >> output_stream, 'Train Params: \t{}'.format(train_params)
    print >> output_stream, 'Test Params: \t{}'.format(test_params)
    print >> output_stream, 'Report Params: \t{}'.format(display_params)
    print >> output_stream, '='*80
    print >> output_stream
