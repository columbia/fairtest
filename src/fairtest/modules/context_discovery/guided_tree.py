"""
Guided Tree Construction Algorithm.
"""
from fairtest.modules.metrics import Metric
import operator
import numpy as np
from collections import Counter
from ete2 import Tree
from sklearn.externals import six
from sklearn.externals.six import StringIO
from copy import copy
import logging
import multiprocessing


def find_thresholds(data, features, feature_info, num_bins):
    """
    Find thresholds for continuous features (quantization)

    Parameters
    ----------
    data :
        the dataset

    features :
        the list of features

    feature_info :
        feature information

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
        if feature_info[feature].arity is None:

            # count frequency of each value of the feature
            counts = Counter(data[feature])
            values = sorted(counts.keys())
            if len(values) <= num_bins:
                # there are less than num_bins values
                thresholds[feature] = (np.array(values[0:-1]) +
                                       np.array(values[1:]))/2.0
            else:
                # Binning algorithm from Spark. Build 'num_bins' bins of roughly
                # equal sample size
                approx_size = (1.0*len(data[feature])) / (num_bins + 1)
                feature_thresholds = []

                current_count = counts[values[0]]
                index = 1
                target_count = approx_size
                while index < len(counts):
                    previous_count = current_count
                    current_count += counts[values[index]]
                    previous_gap = abs(previous_count - target_count)
                    curent_gap = abs(current_count - target_count)

                    if previous_gap < curent_gap:
                        feature_thresholds.\
                                append((values[index] + values[index-1])/2.0)
                        target_count += approx_size
                    index += 1

                thresholds[feature] = feature_thresholds
    return thresholds


class ScoreParams(object):
    """
    Split-scoring parameters
    """

    # Child-score aggregation (weighted average, average or max)
    WEIGHTED_AVG = 'weighted_avg'
    AVG = 'avg'
    MAX = 'max'
    AGG_TYPES = [WEIGHTED_AVG, AVG, MAX]

    def __init__(self, metric, agg_type, conf):
        assert agg_type in ScoreParams.AGG_TYPES
        self.metric = metric
        self.agg_type = agg_type
        self.conf = conf


class SplitParams(object):
    """
    Split parameters
    """
    def __init__(self, targets, sens, expl, dim, feature_info,
                 thresholds, min_leaf_size):
        self.targets = targets
        self.sens = sens
        self.expl = expl
        self.dim = dim
        self.feature_info = feature_info
        self.thresholds = thresholds
        self.min_leaf_size = min_leaf_size


def build_tree(data, feature_info, sens, expl, output, metric, conf,
               max_depth, min_leaf_size=100, agg_type='avg', max_bins=10):
    """
    Builds a decision tree guided towards nodes with high bias

    Parameters
    ----------
    data :
        the dataset

    feature_info :
        information about user features

    sens :
        name of the sensitive feature

    expl :
        name of the explanatory feature

    output :
        the target feature

    metric :
        the fairness metric to use

    conf :
        the confidence level

    max_depth :
        maximum depth of the decision-tree

    min_leaf_size :
        minimum size of a leaf

    agg_type :
        aggregation method for children scores

    max_bins :
        maximum number of bins to use when binning continuous features

    Returns
    -------
    tree :
        the tree built by the algorithm
    """
    logging.info('Building a Guided Decision Tree')
    tree = Tree()

    # Check if there are multiple labeled outputs
    targets = data.columns[-output.num_labels:].tolist()
    logging.debug('Targets: %s', targets)

    features = set(data.columns.tolist())-set([sens, expl])-set(targets)
    logging.debug('Contextual Features: %s', features)

    # check the data dimensions
    if metric.dataType == Metric.DATATYPE_CORR:
        if expl:
            dim = (feature_info[expl].arity, 6)
        else:
            dim = 6
    else:
        # get the dimensions of the OUTPUT x SENSITIVE contingency table
        if expl:
            dim = (feature_info[expl].arity, output.arity,
                   feature_info[sens].arity)
        else:
            dim = (output.arity, feature_info[sens].arity)

    logging.debug('Data Dimension for Metric: %s', dim)

    # bin the continuous features
    cont_thresholds = find_thresholds(data, features, feature_info, max_bins)

    score_params = ScoreParams(metric, agg_type, conf)
    split_params = SplitParams(targets, sens, expl, dim, feature_info,
                               cont_thresholds, min_leaf_size)

    # get a measure for the root
    if metric.dataType == Metric.DATATYPE_CT:
        stats = [count_values(data, sens, targets[0], expl, dim)[0]]
    elif metric.dataType == Metric.DATATYPE_CORR:
        stats = [corr_values(data, sens, targets[0])[0]]
    else:
        stats = [data[targets+[sens]]]

    _, root_metric = score(stats, score_params)
    tree.add_features(metric=root_metric[0])

    #
    # Builds up the tree recursively. Selects the best feature to split on,
    # in order to maximize the average bias (mutual information) in all
    # sub-trees.
    def rec_build_tree(node_data, node, pred, split_features, depth,
                       parent_score, pool):
        """
        Recursive tree building.

        Parameters
        ----------
        node_data :
            the data for the current node

        pred :
            the predicate defining the current context

        split_features :
            the features on which a split can occur

        depth :
            the current depth

        parent_score :
            the metric score at the parent

        Returns
        -------
        tree :
            the tree built by the algorithm
        """

        node.add_features(size=len(node_data))

        # make a new leaf if recursion is stopped
        if (depth == max_depth) or (len(split_features) == 0):
            return

        logging.debug('looking for splits at pred %s', pred)

        # select the best feature to split on
        split_score, best_feature, threshold, to_drop, child_metrics = \
            select_best_feature(node_data, split_features, split_params,
                                score_params, parent_score, pool)

        # no split found, make a leaf
        if best_feature is None:
            return

        logging.info('splitting on %s (score=%s) with threshold %s at pred %s',
                     best_feature, split_score, threshold, pred)

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
                                    metric=child_metrics['left'])

            right_child = node.add_child(name=str(pred_right))
            right_child.add_features(feature_type='continuous',
                                     feature=best_feature,
                                     threshold=threshold,
                                     is_left=False,
                                     metric=child_metrics['right'])

            # recursively build the tree
            rec_build_tree(data_left, left_child, pred+[pred_left],
                           split_features-set(to_drop), depth+1, split_score, pool)
            rec_build_tree(data_right, right_child, pred+[pred_right],
                           split_features-set(to_drop), depth+1, split_score, pool)

        else:
            # categorical split
            for val in node_data[best_feature].unique():

                # check if this child was pruned or not
                if val in child_metrics:
                    # predicate for the current sub-tree
                    new_pred = "{} = {}".format(best_feature, val)

                    # add a node to the underlying tree structure
                    child = node.add_child(name=str(new_pred))
                    child.add_features(feature_type='categorical',
                                       feature=best_feature,
                                       category=val,
                                       metric=child_metrics[val])

                    child_data = node_data[node_data[best_feature] == val]

                    # recursively build the tree
                    rec_build_tree(child_data, child, pred+[new_pred],
                                   split_features-set(to_drop + [best_feature]),
                                   depth+1, split_score, pool)

    #
    # When contextual features are just a few there is
    # no actual benefit out of parallelization. In fact,
    # contemption introduces a slight overhead. Hence,
    # use only one thread to score less than 10 features.
    #
    if len(features) < 10:
        pool_size = 1
    else:
        pool_size = multiprocessing.cpu_count() - 2

    pool = multiprocessing.Pool(pool_size)
    rec_build_tree(data, tree, [], features, 0, 0, pool)
    pool.close()
    pool.join()

    return tree


def score_feature(args):
    """
    Scores a particular feature

    Parameters
    ----------
    args:
        a tuple of aguments

    Returns
    -------
    dict:
        a dictionary of feature score info
    """
    # unpack a long tuple of arguments
    feature,\
    sens,\
    targets,\
    expl,\
    feature_info,\
    node_data,\
    split_params,\
    score_params,\
    parent_score,\
    best_better_than_parent,\
    max_score =  args

    to_drop = []
    feature_list = [feature, sens] + targets
    if expl:
        feature_list.append(expl)

    # determine type of split
    if feature_info[feature].arity:
        split_score, metrics = test_cat_feature(node_data[feature_list],
                                                feature, split_params,
                                                score_params)
        threshold = None
    else:
        split_score, threshold, metrics = \
                test_cont_feature(node_data[feature_list], feature,
                                  split_params, score_params)

    logging.debug('feature %s: score %s', feature, split_score)

    # the feature produced no split and can be dropped in sub-trees
    if split_score is None or np.isnan(split_score):
        to_drop.append(feature)
        return

    # check if there is a child with higher score than the parent
    curr_better_than_parent = \
        len([metric for metric in metrics.values()
             if metric.abs_effect() > parent_score]) > 0

    new_best = False
    if curr_better_than_parent:
        if not best_better_than_parent:
            # No previous split resulted in a child with a higher score
            # than the parent. The current split is the best.
            new_best = True
            best_better_than_parent = True
        elif split_score > max_score:
            # There was a previous split that resulted in a child with a
            # higher score than the parent, but the current split is even
            # better
            new_best = True
            best_better_than_parent = True
    elif not best_better_than_parent and split_score > max_score:
        # No split so far resulted in a child with a higher score
        # than the parent, but the current split has the highest score
        new_best = True

    return {
        'split_score': split_score,
        'feature': feature,
        'threshold': threshold,
        'to_drop': to_drop,
        'metrics': metrics
    }


def select_best_feature(node_data, features, split_params,
                        score_params, parent_score, pool):
    """
    Selects the optimal contextual feature to split on to maximize bias

    Parameters
    ----------
    node_data :
        the current data

    features :
        the features to consider

    split_params :
        the splitting parameters

    score_params :
        the split scoring parameters

    parent_score :
        the score of the parent node

    Returns
    -------
    max_score :
        the score achieved by the best split

    best_feature :
        the best feature

    best_threshold :
        the best threshold (for continuous features)

    to_drop :
        a List of features to drop

    best_metrics :
        the metrics for all the sub-trees induced by the best split
    """
    best_feature = None
    best_threshold = None
    best_metrics = None
    best_better_than_parent = False
    max_score = 0

    # keep track of useless features (no splits available)
    to_drop = []

    feature_info = split_params.feature_info
    sens = split_params.sens
    expl = split_params.expl
    targets = split_params.targets

    # create a list of argument tuples
    args = zip(
        features,
        [sens]*len(features),
        [targets]*len(features),
        [expl]*len(features),
        [feature_info]*len(features),
        [node_data]*len(features),
        [split_params]*len(features),
        [score_params]*len(features),
        [parent_score]*len(features),
        [best_better_than_parent]*len(features),
        [max_score]*len(features)
    )

    # parallelize scoring of features
    results = pool.map_async(score_feature, args).get()

    # drop None
    results = filter(lambda item: item, results)

    # pick best score; also merge "to_drop" features
    if results:
        best =  sorted(results, key=lambda d: d['split_score'], reverse=True)[0]
        to_drop =  reduce(lambda a,b:a+b, map(lambda d: d['to_drop'], results))

        max_score = best['split_score']
        best_feature = best['feature']
        best_threshold = best['threshold']
        best_metrics = best['metrics']

    return max_score, best_feature, best_threshold, to_drop, best_metrics


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

    Returns
    -------
    values :
        the contingency table

    size :
        the size of the data
    """
    values = np.zeros(dim)

    if expl:
        # group by value of the explanatory feature
        groups = [zip(group[target], group[sens])
                  for (_, group) in data.groupby(expl)]
        counters = [Counter(group) for group in groups]

        # build a three-way contingency table
        for k in range(len(counters)):
            for i in range(dim[1]):
                for j in range(dim[2]):
                    values[k, i, j] = counters[k].get((i, j), 0)

        return values, min([len(g) for g in groups])

    else:
        # build a two-dimensional contingency table
        counter = Counter(zip(data[target], data[sens]))
        for i in range(dim[0]):
            for j in range(dim[1]):
                values[i, j] = counter.get((i, j), 0)

        return values, len(data)


def corr_values(data, sens, target):
    """
    Get statistics for correlation measures

    Parameters
    ----------
    data :
        the data

    sens :
        the sensitive feature

    target :
        the targeted feature

    Returns
    -------
    values :
        an array of aggregate statistics
    """
    (x, y) = (np.array(data[sens]), np.array(data[target]))
    # sum(x), sum(x^2), sum(y), sum(y^2), sum(xy)
    return np.array([x.sum(),
                     np.dot(x, x),
                     y.sum(),
                     np.dot(y, y),
                     np.dot(x, y), x.size]), len(x)


def test_cat_feature(node_data, feature, split_params, score_params):
    """
    Find the best split for a categorical feature

    Parameters
    ----------
    node_data :
        the current data

    feature :
        the feature to consider

    split_params :
        the splitting parameters

    score_params :
        the split scoring parameters

    Returns
    -------
    split_score :
        the score of the current split

    dict :
        a dictionary of child metrics
    """
    logging.debug('testing categorical feature %s', feature)
    sens = split_params.sens
    dim = split_params.dim
    expl = split_params.expl
    targets = split_params.targets
    min_leaf_size = split_params.min_leaf_size
    data_type = score_params.metric.dataType

    if data_type == Metric.DATATYPE_CT:
        # build a contingency table for each child
        child_stats = [(key, count_values(group, sens, targets[0], expl, dim))
                       for key, group in node_data.groupby(feature)]
    elif data_type == Metric.DATATYPE_CORR:
        # compute summary statistics for each child
        child_stats = [(key, corr_values(group, sens, targets[0]))
                       for key, group in node_data.groupby(feature)]
    else:
        # aggregate all the data for each child for regression
        child_stats = [(key, (group[targets+[sens]], len(group)))
                       for key, group in node_data.groupby(feature)]

    n = len(node_data)
    n_children = sum([size for (key, (group, size)) in child_stats
                      if size >= min_leaf_size])

    # prune small sub-trees
    child_stats = [(key, group) for (key, (group, size))
                   in child_stats if size >= min_leaf_size]

    split_score = None
    # compute the split score
    if len(child_stats) > 1:
        children, child_stats = zip(*child_stats)
        split_score, metrics = \
            score(child_stats, score_params, weight=(1.0*n_children/n))

        logging.debug('split score: %s', split_score)
        return split_score, dict(zip(children, metrics))
    else:
        return split_score, None


def test_cont_feature(node_data, feature, split_params, score_params):
    """
    Find the best split for a continuous feature

    Parameters
    ----------
    node_data:
        the current data

    feature :
        the feature to consider

    split_params :
        the splitting parameters

    score_params :
        the split scoring parameters

    Returns
    -------
    max_score :
        maximum score achieved

    best_threshold :
        best threshold found

    best_metrics :
        metrics for the child trees
    """
    logging.debug('testing continuous feature %s', feature)
    sens = split_params.sens
    dim = split_params.dim
    expl = split_params.expl
    targets = split_params.targets
    min_leaf_size = split_params.min_leaf_size
    thresholds = split_params.thresholds[feature]
    data_type = score_params.metric.dataType

    max_score = None
    best_threshold = None
    best_metrics = None

    #
    # If we want to do a regression for each child, simply keep all the data
    # and check the split-score for each threshold
    #
    if data_type == Metric.DATATYPE_REG:
        for threshold in thresholds:
            logging.debug('testing threshold %s', threshold)
            data_left = node_data[node_data[feature] <= threshold]
            data_right = node_data[node_data[feature] > threshold]

            size_left = len(data_left)
            size_right = len(data_right)

            if (size_left >= min_leaf_size) and (size_right >= min_leaf_size):
                split_score, metrics = score([data_left[targets+[sens]],
                                              data_right[targets+[sens]]],
                                             score_params)
                logging.debug('split score: %s', split_score)
                if split_score > max_score:
                    max_score = split_score
                    best_threshold = threshold
                    best_metrics = dict(zip(['left', 'right'], metrics))

        return max_score, best_threshold, best_metrics

    # split data based on the bin thresholds
    groups = node_data.groupby(np.digitize(node_data[feature],
                                           thresholds, right=True))

    if data_type == Metric.DATATYPE_CT:
        # aggregate all the target counts for each bin
        temp = [(key, count_values(group, sens, targets[0], expl, dim))
                for (key, group) in groups]
    elif data_type == Metric.DATATYPE_CORR:
        # correlation scores
        temp = [(key, corr_values(group, sens, targets[0]))
                for (key, group) in groups]

    # get the indices of the bin thresholds
    keys, temp = zip(*temp)

    # get the bins and their sizes
    bins, sizes = zip(*temp)
    total_size = sum(sizes)

    # aggregate of target counts for the complete data
    total = reduce(operator.add, bins, np.zeros(dim))

    # split on the first threshold
    (data_left, size_left) = (bins[0], sizes[0])
    (data_right, size_right) = (total - data_left, total_size - size_left)

    # check score if split is valid
    if (size_left >= min_leaf_size) and (size_right >= min_leaf_size):
        split_score, metrics = score([data_left, data_right], score_params)
        logging.debug('testing threshold %s', thresholds[keys[0]])
        logging.debug('split score: %s', split_score)

        max_score = split_score
        best_threshold = thresholds[keys[0]]
        best_metrics = dict(zip(['left', 'right'], metrics))

    # check all further splits in order by summing bins
    for i in range(1, len(bins)):
        (ct_i, size_i) = (bins[i], sizes[i])
        data_left += ct_i
        data_right -= ct_i

        size_left += size_i
        size_right -= size_i

        if (size_left >= min_leaf_size) and (size_right >= min_leaf_size):
            split_score, metrics = score([data_left, data_right], score_params)
            logging.debug('testing threshold %s', thresholds[keys[i]])
            logging.debug('split score: %s', split_score)

            if split_score > max_score:
                max_score = split_score
                best_threshold = thresholds[keys[i]]
                best_metrics = dict(zip(['left', 'right'], metrics))

    return max_score, best_threshold, best_metrics


def score(stats, score_params, weight=1):
    """
    Compute the score for a split

    Parameters
    ----------
    stats :
        statistics for all the children

    score_params :
        split scoring parameters

    weight :
        weight to apply to the score

    Returns
    -------
    score :
        aggregate of all child scores

    metrics :
        metrics used for each child
    """

    metric = score_params.metric
    agg_type = score_params.agg_type
    conf = score_params.conf

    metrics = [copy(metric) for _ in stats]

    zip_w_metric = zip(stats, metrics)

    # compute a score for each child
    score_list = [metric_copy.compute(child, conf, exact=False).abs_effect()
                  for (child, metric_copy) in zip_w_metric]

    logging.debug('split score list: %s', score_list)

    # take the average or maximum of the child scores
    if agg_type == ScoreParams.WEIGHTED_AVG:
        totals = [group.sum().sum() for group in stats]
        probas = [(1.0*tot)/sum(totals) for tot in totals]
        return weight * np.dot(score_list, probas), metrics
    elif agg_type == ScoreParams.AVG:
        return weight * np.mean(score_list), metrics
    elif agg_type == ScoreParams.MAX:
        return max(score_list), metrics


def export_graphviz(decision_tree, encoders, filename="tree.dot"):
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

    """

    def node_to_str(curr_node):
        """
        Returns a node's information

        Parameters
        ----------
        curr_node :
            the current node

        Returns
        -------
        str :
            the node information
        """

        pred = 'Root'

        if not curr_node.is_root():
            feature = curr_node.feature

            if curr_node.feature_type == 'continuous':
                threshold = curr_node.threshold
                if curr_node.is_left:
                    pred = feature + '<=' + str(threshold)
                else:
                    pred = feature + '>' + str(threshold)
            else:
                category = curr_node.category
                pred = feature + '=' + \
                        str(encoders[feature].inverse_transform([category])[0])

        node_size = curr_node.size
        return "%s\\nsamples = %s" % (pred, node_size)

    def recurse(node, parent_id=None):
        """
        Recursive traversal of the tree

        Parameters
        ----------
        node :
            the current node

        parent_id :
            the id of the parent node
        """

        children = node.get_children()
        node_id = node.id

        # Add node with description
        node_str = node_to_str(node)
        out_file.write('%d [label="%s", shape="box"] ;\n' % (node_id, node_str))

        if parent_id is not None:
            # Add edge to parent
            out_file.write('%d -> %d ;\n' % (parent_id, node_id))

        for child in children:
            recurse(child, node_id)

    own_file = False
    try:
        if isinstance(filename, six.string_types):
            if six.PY3:
                out_file = open(filename, "w", encoding="utf-8")
            else:
                out_file = open(filename, "wb")
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


def print_tree(tree, outfile, encoders):
    """
    Print a tree to a file

    Parameters
    ----------
    tree :
        the tree structure

    outfile :
        the output file

    encoders :
        the encoders used to encode categorical features
    """
    import pydot
    dot_data = StringIO()
    export_graphviz(tree, encoders, filename=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(outfile)
