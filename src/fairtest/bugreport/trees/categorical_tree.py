# -*- coding: utf-8 -*-
"""
Module for building categorical trees
"""
from fairtest.bugreport.statistics import fairness_measures as fm
from fairtest.bugreport.statistics.fairness_measures import Measure

import pydot
import operator
import numpy as np
from collections import Counter
from ete2 import Tree
from sklearn.externals import six
from sklearn.externals.six import StringIO
from copy import copy


def find_thresholds(data, features, categorical, num_bins):
    """
    Find thresholds for continuous features (quantization)

    Parameters
    ----------
    data :
        the dataset

    features :
        the list of features

    categorical :
        the list of categorical features

    num_bins :
        the maximum number of bins

    Returns
    -------
    thresholds :
        dictionary of thresholds
    """
    thresholds = {}

    for feature in features:
        # consider only continuous features
        if feature not in categorical:

            # count frequency of each value of the feature
            counts = Counter(data[feature])
            values = sorted(counts.keys())
            if len(values) <= num_bins:
                # there are less than num_bins values
                thresholds[feature] = (np.array(values[0:-1]) +
                                       np.array(values[1:]))/2.0
            else:
                # binning algorithm from Spark. Build 'num_bins' bins of roughly
                # equal sample size
                approx_size = (1.0*len(data[feature])) / (num_bins + 1)
                feature_thresholds = []

                current_count = counts[values[0]]
                index = 1
                target_count = approx_size
                while index < len(counts):
                    previousCount = current_count
                    current_count += counts[values[index]]
                    previous_gap = abs(previousCount - target_count)
                    curent_gap = abs(current_count - target_count)

                    if previous_gap < curent_gap:
                        feature_thresholds.\
                                append((values[index] + values[index-1])/2.0)
                        target_count += approx_size
                    index += 1

                thresholds[feature] = feature_thresholds
    return thresholds


class ScoreParams:
    """
    Split-scoring parameters
    """
    # Child-score aggregation (weighted average, average or max)
    WEIGHTED_AVG = 'WEIGHTED_AVG'
    AVG = 'AVG'
    MAX = 'MAX'
    AGG_TYPES = [WEIGHTED_AVG, AVG, MAX]

    def __init__(self, measure, agg_type, expl=None):
        self.measure = measure
        self.agg_type = agg_type
        self.expl = expl


class SplitParams:
    """
    Split parameters
    """
    def __init__(self, targets, sens, dim, categorical,
                 thresholds, min_leaf_size, expl=None):
        self.targets = targets
        self.sens = sens
        self.dim = dim
        self.categorical = categorical
        self.thresholds = thresholds
        self.min_leaf_size = min_leaf_size
        self.expl = expl


def build_tree(dataset,
               categorical,
               max_depth=5,
               min_leaf_size=100,
               measure=fm.NMI(ci_level=0.95),
               agg_type="WEIGHTED_AVG", max_bins=10):
    """
    Builds a decision tree for finding nodes with high bias

    Parameters
    ----------
    dataset :
        The dataset object

    categorical :
        List of categorical features

    max_depth :
        Maximum depth of the decision-tree

    min_leaf_size :
        Minimum size of a leaf

    measure :
        Fairness measure to use

    agg_type :
        Aggregation method for children scores

    max_bins :
        Maximum number of bins to use when binning continuous features

    Returns
    -------
    tree :
        the tree built
    """
    tree = Tree()

    data = dataset.data_train

    # Check if there are multiple labeled outputs
    if dataset.out_type == 'labeled':
        targets = dataset.labels.tolist()
    else:
        targets = [dataset.out]

    sens = dataset.sens
    expl = dataset.expl
    features = data.columns.difference(targets).difference([sens])

    if expl:
        assert isinstance(measure, fm.CondNMI)
        features = features.difference([expl])

    # check the data dimensions
    if isinstance(measure, fm.CORR):
        if expl:
            dim = (len(dataset.encoders[expl].classes_), 6)
        else:
            dim = 6
    else:
        assert dataset.out_type in ['cat', 'labeled'] \
               and dataset.sens_type == 'cat'

        # get the dimensions of the OUTPUT x SENSITIVE contingency table
        if expl:
            dim = (len(dataset.encoders[expl].classes_),
                   len(dataset.encoders[dataset.out].classes_),
                   len(dataset.encoders[dataset.sens].classes_))
        else:
            dim = (len(dataset.encoders[dataset.out].classes_),
                   len(dataset.encoders[dataset.sens].classes_))

    # bin the continuous features
    cont_thresholds = find_thresholds(data, features, categorical, max_bins)

    score_params = ScoreParams(measure, agg_type, expl)
    split_params = SplitParams(targets, sens, dim, categorical,
                               cont_thresholds, min_leaf_size, expl)

    # get a measure for the root
    if measure.dataType == Measure.DATATYPE_CT:
        target = targets[0]
        stats = [count_values(data, sens, target, expl, dim)[0]]
    elif measure.dataType == Measure.DATATYPE_CORR:
        target = targets[0]
        # compute summary statistics for each child
        stats = [corr_values(data, sens, target)[0]]
    else:
        # aggregate all the data for each child for regression
        stats = [data[targets+[sens]]]

    root_score, root_measure = score(stats, score_params)
    tree.add_features(measure=root_measure[0])

    #
    # Builds up the tree recursively. Selects the best feature to split on,
    # in order to maximize the average bias (mutual information) in all
    # sub-trees.
    def rec_build_tree(node_data, node, pred,
                       node_features, depth, parent_score):
        node.add_features(size=len(node_data))

        # make a new leaf
        if (depth == max_depth) or (len(node_features) == 0):
            return

        # select the best feature to split on
        split_score, best_feature, threshold, to_drop, child_measures = \
            select_best_feature(node_data, node_features,
                                split_params, score_params, parent_score)

        # no split found, make a leaf
        if not best_feature:
            return

        # print 'splitting on {} with threshold {} at pred {}'.\
        #   format(best_feature, threshold, pred)

        if threshold:
            # binary split
            data_left = node_data[node_data[best_feature] <= threshold]
            data_right = node_data[node_data[best_feature] > threshold]

            # predicates for sub-trees
            pred_left = "{} <= {}".format(best_feature, threshold)
            pred_right = "{} > {}".format(best_feature, threshold)

            # add new nodes to the underlying tree structure
            left_child = node.add_child(name=str(pred_left))
            left_child.add_features(feature_type='continuous',
                                    feature=best_feature,
                                    threshold=threshold,
                                    is_left=True,
                                    measure=child_measures['left'])

            right_child = node.add_child(name=str(pred_right))
            right_child.add_features(feature_type='continuous',
                                     feature=best_feature,
                                     threshold=threshold,
                                     is_left=False,
                                     measure=child_measures['right'])

            # recursively build the tree
            rec_build_tree(data_left,
                           left_child,
                           pred+[pred_left],
                           node_features.drop(to_drop),
                           depth+1,
                           split_score)
            rec_build_tree(data_right,
                           right_child,
                           pred+[pred_right],
                           node_features.drop(to_drop),
                           depth+1,
                           split_score)
        else:
            # categorical split
            for val in node_data[best_feature].unique():
                data_child = node_data[node_data[best_feature] == val]

                if len(data_child) >= min_leaf_size:

                    # predicate for the current sub-tree
                    new_pred = "{} = {}".format(best_feature, val)

                    # add a node to the underlying tree structure
                    child = node.add_child(name=str(new_pred))
                    child.add_features(feature_type='categorical',
                                       feature=best_feature,
                                       category=val,
                                       measure=child_measures[val])

                    # recursively build the tree
                    rec_build_tree(data_child, child, pred+[new_pred],
                                   node_features.drop(to_drop + [best_feature]),
                                   depth+1, split_score)

    rec_build_tree(data, tree, [], features, 0, 0)
    return tree


def select_best_feature(node_data, features, split_params,
                        score_params, parent_score):
    """
    Selects the optimal non-sensitive feature to split on to maximize bias

    Parameters
    ----------
    node_data :
        The current data

    features :
        The features to consider

    split_params :
        The splitting parameters

    score_params :
        The split scoring parameters

    parent_score :
        The score of the parent node

    Returns
    -------
    best_feature :
        The best feature

    best_threshold :
        The best threshol

    to_drop :
        A List of features to drop

    best_measures :
        The best measures
    """
    best_feature = None
    best_threshold = None
    best_measures = None
    best_better_than_parent = False
    max_score = 0

    # keep track of useless features (no splits available)
    to_drop = []

    categorical = split_params.categorical
    targets = split_params.targets
    sens = split_params.sens
    expl = split_params.expl

    # iterate over all available non-sensitive features
    for feature in features:
        feature_list = [feature, sens] + targets
        if expl:
            feature_list.append(expl)

        # determine type of split
        if feature in categorical:
            split_score, measures = test_cat_feature(node_data[feature_list],
                                                     feature, split_params,
                                                     score_params)
            threshold = None
        else:
            split_score, threshold, measures = \
                    test_cont_feature(node_data[feature_list],
                                      feature,
                                      split_params,
                                      score_params)

        # the feature produced no split and can be dropped for future sub-trees
        if not split_score:
            to_drop.append(feature)
            continue

        curr_better_than_parent = \
            len(filter(lambda measure: measure.abs_effect() > parent_score,
                       measures.values())) > 0

        new_best = False

        if curr_better_than_parent:
            # No better-than-parent split was found yet. Automatically new best
            if not best_better_than_parent:
                new_best = True
                best_better_than_parent = True
            # Better than the previous better-than-parent split
            elif split_score > max_score:
                new_best = True
                best_better_than_parent = True
        elif not best_better_than_parent and split_score > max_score:
            # not better than parent but highest score
            new_best = True

        # check quality of split
        if new_best:
            max_score = split_score
            best_feature = feature
            best_threshold = threshold
            best_measures = measures

    return max_score, best_feature, best_threshold, to_drop, best_measures


def count_values(data, sens, target, expl, dim):
    """
    Count occurrences of target values and reshape as a contingency table

    Parameters
    ----------
    data :
        the data to count

    sens :
        The sensitive feature

    target :
        The targeted feature

    expl :
        A potentially explanatory feature

    dim :
        The dimensions of the sensitive and targeted features
    """
    values = np.zeros(dim)

    if expl:
        groups = [zip(group[target], group[sens])
                  for (_, group) in data.groupby(expl)]
        counters = [Counter(group) for group in groups]

        for k in range(len(counters)):
            for i in range(dim[1]):
                for j in range(dim[2]):
                    values[k, i, j] = counters[k].get((i, j), 0)

        return values, min(map(lambda g: len(g), groups))

    else:
        counter = Counter(zip(data[target], data[sens]))
        for i in range(dim[0]):
            for j in range(dim[1]):
                values[i, j] = counter.get((i, j), 0)

        return values, len(data)


def corr_values(data, sens, target):
    """
    Get statistics for Pearson-correlation measure

    Parameters
    ----------
    data :
        The data

    sens :
        The sensitive feature

    target :
        The targeted feature
    """
    (x, y) = (np.array(data[sens]), np.array(data[target]))
    # sum(x), sum(x^2), sum(y), sum(y^2), sum(xy)
    return np.array([x.sum(),
                     np.dot(x, x),
                     y.sum(),
                     np.dot(y, y),
                     np.dot(x, y), x.size]), len(data)


def test_cat_feature(node_data, feature, split_params, score_params):
    """
    Find the best split for a categorical feature

    Parameters
    ----------
    node_data :
        The current data

    feature :
        The feature to consider

    split_params :
        The splitting parameters

    score_params :
        The split scoring parameters

    Returns
    -------
    split_score :
        the score of the current split
    """
    #print 'testing categorical feature {}'.format(feature)
    targets = split_params.targets
    sens = split_params.sens
    dim = split_params.dim
    expl = split_params.expl
    min_leaf_size = split_params.min_leaf_size
    data_type = score_params.measure.dataType

    if data_type == Measure.DATATYPE_CT:
        target = targets[0]
        # build a contingency table for each child
        contigency_table = [(key, count_values(group, sens, target, expl, dim))
                            for key, group in node_data.groupby(feature)]
    elif data_type == Measure.DATATYPE_CORR:
        target = targets[0]
        # compute summary statistics for each child
        contigency_table = [(key, corr_values(group, sens, target))\
                for key, group in node_data.groupby(feature)]
    else:
        # aggregate all the data for each child for regression
        contigency_table = [(key, (group[targets+[sens]], len(group)))\
                for key, group in node_data.groupby(feature)]

    # prune small sub-trees
    contigency_table = [(key, group) for (key, (group, size))
                        in contigency_table if size >= min_leaf_size]

    split_score = None
    # compute the split score
    if len(contigency_table) > 1:
        values, contigency_table = zip(*contigency_table)
        split_score, measures = score(contigency_table, score_params)
        #print split_score
        return split_score, dict(zip(values, measures))
    else:
        return split_score, None


def test_cont_feature(node_data, feature, split_params, score_params):
    """
    Find the best split for a continuous feature

    Parameters
    ----------
    node_data:
        The current data

    feature :
        The feature to consider

    split_params :
        The splitting parameters

    score_params :
        The split scoring parameters

    Returns
    -------
    max_score :
        maximum score achieved

    best_threshold :
        best threshold found

    best_measures :
        best measures
    """
    #print 'testing continuous feature {}'.format(feature)
    targets = split_params.targets
    sens = split_params.sens
    dim = split_params.dim
    expl = split_params.expl
    min_leaf_size = split_params.min_leaf_size
    thresholds = split_params.thresholds[feature]
    data_type = score_params.measure.dataType

    max_score = None
    best_threshold = None
    best_measures = None

    #
    # If we want to do a regression for each child, simply keep all the data
    # and check the split-score for each threshold
    #
    if data_type == Measure.DATATYPE_REG:
        for threshold in thresholds:
            # print '    testing threshold {}'.format(threshold)
            data_left = node_data[node_data[feature] <= threshold]
            data_right = node_data[node_data[feature] > threshold]

            size_left = len(data_left)
            size_right = len(data_right)

            if (size_left >= min_leaf_size) and (size_right >= min_leaf_size):
                split_score, measures = score([data_left[targets+[sens]],
                                               data_right[targets+[sens]]],
                                              score_params)
                if split_score > max_score:
                    max_score = split_score
                    best_threshold = threshold
                    best_measures = dict(zip(['left', 'right'], measures))

        return max_score, best_threshold, best_measures

    # split data based on the bin thresholds
    groups = node_data.groupby(np.digitize(node_data[feature],
                                           thresholds,
                                           right=True))

    if data_type == Measure.DATATYPE_CT:
        target = targets[0]
        # aggregate all the target counts for each bin
        temp = map(lambda (key, group): (key, count_values(group,
                                                           sens,
                                                           target,
                                                           expl,
                                                           dim)),
                   groups)
    elif data_type == Measure.DATATYPE_CORR:
        target = targets[0]
        # correlation score
        temp = map(lambda (key, group): (key, corr_values(group, sens, target)),
                   groups)

    # get the indices of the bin thresholds
    keys, temp = zip(*temp)

    # get the bins and their sizes
    bins, sizes = zip(*temp)
    total_size = sum(sizes)

    # aggregate of target counts for the complete data
    total = reduce(operator.add, bins, np.zeros(dim))

    #print 'total = {}'.format(total)

    # split on the first threshold
    (data_left, size_left) = (bins[0], sizes[0])
    (data_right, size_right) = (total - data_left, total_size - size_left)

    #print 'data left = {}'.format(data_left)
    #print 'data right = {}'.format(data_right)

    # check score if split is valid
    if (size_left >= min_leaf_size) and (size_right >= min_leaf_size):
        #print 'testing threshold {}'.format(thresholds[keys[0]])
        split_score, measures = score([data_left, data_right], score_params)
        max_score = split_score
        best_threshold = thresholds[keys[0]]
        best_measures = dict(zip(['left', 'right'], measures))

    # check all further splits in order
    for i in range(1, len(bins)):
        (ct_i, size_i) = (bins[i], sizes[i])
        data_left += ct_i
        data_right -= ct_i

        #print 'contigency_table {} = {}'.format(i, ct_i)
        #print 'data left {} = {}'.format(i, data_left)
        #print 'data right {} = {}'.format(i, data_right)

        size_left += size_i
        size_right -= size_i

        if (size_left >= min_leaf_size) and (size_right >= min_leaf_size):
            #print 'testing threshold {}'.format(thresholds[keys[i]])
            split_score, measures = score([data_left, data_right], score_params)

            if split_score > max_score:
                max_score = split_score
                best_threshold = thresholds[keys[i]]
                best_measures = dict(zip(['left', 'right'], measures))
    #if max_score:
    #    print max_score, best_threshold, map(lambda m: m.stats[0],
    #                                         best_measures.values())
    return max_score, best_threshold, best_measures


def score(stats, score_params):
    """
    Compute the score for a split

    Parameters
    ----------
    stats :
        Statistics for all the children

    score_params :
        Split scoring parameters
    """
    measure = score_params.measure
    agg_type = score_params.agg_type

    measures = [copy(measure) for _ in stats]

    zip_w_measure = zip(stats, measures)

    # compute a score for each child
    score_list = map(lambda (child, measure_copy): measure_copy.compute(child, approx=True).abs_effect(), zip_w_measure)

    # take the average or maximum of the child scores
    if agg_type == ScoreParams.WEIGHTED_AVG:
        totals = map(lambda group: group.sum().sum(), stats)
        probas = map(lambda tot: (1.0*tot)/sum(totals), totals)
        return np.dot(score_list, probas), measures
    elif agg_type == ScoreParams.AVG:
        return np.mean(score_list), measures
    elif agg_type == ScoreParams.MAX:
        return max(score_list), measures


def export_graphviz(decision_tree,
                    encoders,
                    out_file="tree.dot",
                    is_spark=False):
    """
    Export a tree to a file (adapted from scikit source code)

    Parameters
    ----------
    decision_tree :
        the tree to export

    encoders :
        the encoders used to encode categorical features

    out_file :
        the output file

    is_spark :
        if the tree was produced by Spark
    """
    # print node information
    def node_to_str(node):
        pred = 'Root'

        if not node.is_root():
            feature = node.feature

            if node.feature_type == 'continuous':
                threshold = node.threshold
                if node.is_left:
                    pred = feature + '<=' + str(threshold)
                else:
                    pred = feature + '>' + str(threshold)
            else:
                category = node.category
                pred = feature + '=' + \
                        str(encoders[feature].inverse_transform([category])[0])

        node_size = node.size
        return "%s\\nsamples = %s" % (pred, node_size)

    def node_to_str_spark(node):
        """
        print Spark node information
        """
        pred = 'Root'

        if not node.is_root():
            pred = node.name

        node_size = node.size
        return "%s\\nsamples = %s" % (pred, node_size)

    def recurse(node, parent_id=None):

        children = node.get_children()
        node_id = node.id

        # Add node with description
        if is_spark:
            node_str = node_to_str_spark(node)
        else:
            node_str = node_to_str(node)
        out_file.write('%d [label="%s", shape="box"] ;\n' % (node_id, node_str))

        if parent_id is not None:
            # Add edge to parent
            out_file.write('%d -> %d ;\n' % (parent_id, node_id))

        for child in children:
            recurse(child, node_id)

    own_file = False
    try:
        if isinstance(out_file, six.string_types):
            if six.PY3:
                out_file = open(out_file, "w", encoding="utf-8")
            else:
                out_file = open(out_file, "wb")
            own_file = True

        out_file.write("digraph Tree {\n")

        node_id = 0
        for node in decision_tree.traverse("levelorder"):
            node.add_features(id=node_id)
            node_id += 1

        recurse(decision_tree, None)
        out_file.write("}")

    finally:
        if own_file:
            out_file.close()


def print_tree(tree, outfile, encoders, is_spark=False):
    """
    Print a tree to a file

    Parameters
    ----------
    tree :
        The tree structure

    outfile :
        The output file

    encoders :
        The encoders used to encode categorical features

    is_spark :
        If the tree was produced by Spark or not
    """
    dot_data = StringIO()
    export_graphviz(tree, encoders, out_file=dot_data, is_spark=is_spark)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(outfile)
