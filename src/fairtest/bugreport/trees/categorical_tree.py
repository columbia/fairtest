# -*- coding: utf-8 -*-

from fairtest.bugreport.statistics import fairness_measures as fm
from fairtest.bugreport.statistics.fairness_measures import NMI

import pydot
import operator
import numpy as np
import pandas as pd
from collections import Counter
from ete2 import Tree, TreeStyle, AttrFace, TextFace, faces
from sklearn.externals import six
from sklearn.externals.six import StringIO 
from math import sqrt

#
# Find thresholds for continuous features (quantization)
#
# @args data        the dataset
# @args features    the list of features
# @args categorical the list of categorical features
# @args num_bins    the maximum number of bins
#
def find_thresholds(data, features, categorical, num_bins):
    thresholds = {}

    for feature in features:
        # consider only continuous features
        if feature not in categorical:
        
            # count frequency of each value of the feature
            counts = Counter(data[feature])
            values = sorted(counts.keys())
            if len(values) <= num_bins:
                # there are less than num_bins values
                thresholds[feature] = (np.array(values[0:-1]) + np.array(values[1:]))/2.0
            else:
                # binning algorithm from Spark. Build 'num_bins' bins of roughly
                # equal sample size
                approx_size = (1.0*len(data[feature])) / (num_bins + 1)
                feature_thresholds = []
                
                currentCount = counts[values[0]]
                index = 1
                targetCount = approx_size
                while (index < len(counts)) :
                    previousCount = currentCount
                    currentCount += counts[values[index]]
                    previousGap = abs(previousCount - targetCount)
                    currentGap = abs(currentCount - targetCount)
                    
                    if (previousGap < currentGap) :
                        feature_thresholds.append((values[index] + values[index-1])/2.0)
                        targetCount += approx_size
                    index += 1
                
                thresholds[feature] = feature_thresholds
    return thresholds
#
# Split-scoring parameters
#                    
class ScoreParams:
    
    # Child-score aggregation (weighted average, average or max)
    WEIGHTED_AVG = 'WEIGHTED_AVG'
    AVG = 'AVG'
    MAX = 'MAX'
    AGG_TYPES = [WEIGHTED_AVG, AVG, MAX]

    def __init__(self, measure, agg_type):
        self.measure = measure
        self.agg_type = agg_type

#
# Split parameters
#
class SplitParams:
    def __init__(self, target, sens, dim, categorical, thresholds, min_leaf_size):
        self.target = target
        self.sens = sens
        self.dim = dim
        self.categorical = categorical
        self.thresholds = thresholds
        self.min_leaf_size = min_leaf_size

#
# Builds a decision tree for finding nodes with high bias
#
# @args data            The dataset
# @args dim             The output dimension
# @args categorical     List of categorical features
# @param max_depth      Maximum depth of the decision-tree
# @param min_leaf_size  Minimum size of a leaf
# @param measure        Dependency measure to use
# @param agg_type       Aggregation method for children scores
# @param conf           Confidence level
# @param max_bins       Maximum number of bins to use when binning continuous features
#
def build_tree(data, dim, categorical, max_depth=5, min_leaf_size=100, measure=fm.NMI(ci_level=0.95), agg_type="WEIGHTED_AVG", max_bins=10):
    t = Tree()
    
    target = data.columns[0]
    sens = data.columns[1]
    features = data.columns[2:]
    
    # bin continuous features
    cont_thresholds = find_thresholds(data, features, categorical, max_bins)
    
    score_params = ScoreParams(measure, agg_type)
    split_params = SplitParams(target, sens, dim, categorical, cont_thresholds, min_leaf_size)
    
    #
    # Builds up the tree recursively. Selects the best feature to split on,
    # in order to maximize the average bias (mutual information) in all 
    # sub-trees.
    #
    def rec_build_tree(node_data, node, pred, node_features, depth):
        node.add_features(size=len(node_data))
        
        # make a new leaf
        if ((depth == max_depth) or (len(node_features) == 0)):
            return
        
        # select the best feature to split on
        best_feature, threshold, to_drop = select_best_feature(node_data, node_features, split_params, score_params)

        # no split found, make a leaf
        if not best_feature:
            return
        
        #print 'splitting on {} with threshold {} at pred {}'.format(best_feature, threshold, pred)
        
        if threshold:
            # binary split
            data_left = node_data[node_data[best_feature] <= threshold]
            data_right = node_data[node_data[best_feature] > threshold]
            
            # predicates for sub-trees
            pred_left = "{} <= {}".format(best_feature, threshold)
            pred_right = "{} > {}".format(best_feature, threshold)
            
            # add new nodes to the underlying tree structure
            left_child = node.add_child(name=str(pred_left))
            left_child.add_features(feature_type='continuous', feature=best_feature, threshold=threshold, is_left=True)
            
            right_child = node.add_child(name=str(pred_right))
            right_child.add_features(feature_type='continuous', feature=best_feature, threshold=threshold, is_left=False)
            
            # recursively build the tree
            rec_build_tree(data_left, left_child, pred+[pred_left], node_features.drop(to_drop), depth+1)
            rec_build_tree(data_right, right_child, pred+[pred_right], node_features.drop(to_drop), depth+1)
        else:
            # categorical split
            for val in node_data[best_feature].unique():
                data_child = node_data[node_data[best_feature] == val]
                
                if (len(data_child) >= min_leaf_size):
                
                    # predicate for the current sub-tree
                    new_pred = "{} = {}".format(best_feature, val)
                    
                    # add a node to the underlying tree structure
                    child = node.add_child(name=str(new_pred))
                    child.add_features(feature_type='categorical', feature=best_feature, category=val)
                    
                    # recursively build the tree
                    rec_build_tree(data_child, child, pred+[new_pred], node_features.drop(to_drop).drop(best_feature), depth+1)
            
    rec_build_tree(data, t, [], features, 0)
    return t

#
# Selects the optimal non-sensitive feature to split on to maximize bias
#
# @args node_data      The current data
# @args features       The features to consider
# @args split_params   The splitting parameters
# @args score_params   The split scoring parameters
#
def select_best_feature(node_data, features, split_params, score_params)  :
    best_feature = None
    best_threshold = None
    max_score = 0

    # keep track of useless features (no splits available)
    to_drop = []
    
    categorical = split_params.categorical
    target = split_params.target
    sens = split_params.sens
    
    # iterate over all available non-sensitive features
    for feature in features:
        # determine type of split
        if feature in categorical:
            split_score = test_cat_feature(node_data[[feature, target, sens]], feature, split_params, score_params)
            threshold = None
        else:
            split_score, threshold = test_cont_feature(node_data[[feature, target, sens]], feature, split_params, score_params)
        
        # the feature produced no split and can be dropped for future sub-trees
        if not score:
            to_drop.append(feature)

        # check quality of split            
        if split_score > max_score:
            max_score = split_score
            best_feature = feature
            best_threshold = threshold
                
    return best_feature, best_threshold, to_drop

#
# Count occurrences of target values and reshape as a contingency table
#
# @args data    the data to count
# @args dim     the dimensions of the contingency table
#
def countValues(data, dim):
    values = np.zeros(dim)
    counter = Counter(data)
    for i in range(dim[0]):
        for j in range(dim[1]):
            values[i,j] = counter.get((i,j), 0)

    return values, len(data)
    
def corrValues(x, y):
    # sum(x), sum(x^2), sum(y), sum(y^2), sum(xy)
    return np.array([x.sum(), np.dot(x, x), y.sum(), np.dot(y, y), np.dot(x, y), x.size]), x.size

#
# Find the best split for a categorical feature
#
# @args node_data      The current data
# @args feature        The feature to consider
# @args split_params   The splitting parameters
# @args score_params   The split scoring parameters
#
def test_cat_feature(node_data, feature, split_params, score_params):
    
    target = split_params.target
    sens = split_params.sens
    dim = split_params.dim
    min_leaf_size = split_params.min_leaf_size
    
    if dim:
        ct = {key: countValues(zip(group[target], group[sens]), dim) for key, group in node_data.groupby(feature)}
    else:
        ct = {key: corrValues(np.array(group[target]), np.array(group[sens])) for key, group in node_data.groupby(feature)}
     
    # prune small sub-trees
    ct = {key:group for (key,(group,size)) in ct.iteritems() if size >= min_leaf_size}

    split_score = None
    # compute the split score
    if len(ct) > 1:
        split_score = score(ct.values(), score_params)
        return split_score


#
# Find the best split for a continuous feature
#
# @args node_data      The current data
# @args feature       The feature to consider
# @args split_params   The splitting parameters
# @args score_params   The split scoring parameters
#
def test_cont_feature(node_data, feature, split_params, score_params):
    target = split_params.target
    sens = split_params.sens
    dim = split_params.dim
    min_leaf_size = split_params.min_leaf_size
    thresholds = split_params.thresholds[feature]
    
    max_score = None
    best_threshold = None
    
    # split data based on the bin thresholds
    groups = node_data.groupby(np.digitize(node_data[feature], thresholds, right=True))
    
    if dim:
        # aggregate all the target counts for each bin
        #temp = map(lambda (key, group): (key, countValues(pd.crosstab(group[target], group[sens]), dim)), groups)
        temp = map(lambda (key, group): (key, countValues(zip(group[target], group[sens]), dim)), groups)
    else:
        # correlation score
        temp = map(lambda (key, group): (key, corrValues(np.array(group[target]), np.array(group[sens]))), groups)
        
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
        split_score = score([data_left, data_right], score_params)
        max_score = split_score
        best_threshold = thresholds[keys[0]]
    
    # check all further splits in order
    for i in range(1, len(bins)):    
        (ct_i, size_i) = (bins[i], sizes[i])
        data_left += ct_i
        data_right -= ct_i
        size_left += size_i
        size_right -= size_i
        
        if (size_left >= min_leaf_size) and (size_right >= min_leaf_size):
            split_score = score([data_left, data_right], score_params)
        
            if split_score > max_score:
                max_score = split_score
                best_threshold = thresholds[keys[i]]
            
    return max_score, best_threshold

#
# Compute the score for a split
#
# @args child_cts       Contingency tables for all the children
# @args score_params    Split scoring parameters
# 
def score(stats, score_params):
    measure = score_params.measure
    agg_type = score_params.agg_type
    
    score_list = map(lambda ct: measure.normalize_effect(measure.compute(ct)), stats)
    
    # take the average or maximum of the child scores
    if agg_type == ScoreParams.WEIGHTED_AVG:
        totals = map(lambda group: group.sum().sum(), stats)
        probas = map(lambda tot: (1.0*tot)/sum(totals), totals)
        return np.dot(score_list, probas)
    elif agg_type == ScoreParams.AVG:
        return np.mean(score_list)
    elif agg_type == ScoreParams.MAX:
        return max(score_list)

#
# Export a tree to a file (adapted from scikit source code)
#
# @args decision_tree   the tree to export
# @args encoders        the encoders used to encode categorical features
# @args out_file        the output file
# @args is_spark        if the tree was produced by Spark
#        
def export_graphviz(decision_tree, encoders, out_file="tree.dot", is_spark=False):
    
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
                pred = feature + '=' + str(encoders[feature].inverse_transform([category])[0])
        
        node_size = node.size
        return "%s\\nsamples = %s" % (pred, node_size)
    
    # print Spark node information    
    def node_to_str_spark(node):
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
        out_file.write('%d [label="%s", shape="box"] ;\n' %
                        (node_id, node_str))

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

#
# Print a tree to a file
#
# @args tree        The tree structure
# @args outfile     The output file
# @args encoders    The encoders used to encode categorical features
# @args is_spark    If the tree was produced by Spark or not
#      
def print_tree(tree, outfile, encoders, is_spark=False):
    dot_data = StringIO()
    export_graphviz(tree, encoders, out_file=dot_data, is_spark=is_spark) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    graph.write_pdf(outfile) 
        