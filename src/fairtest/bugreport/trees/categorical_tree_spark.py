# -*- coding: utf-8 -*-

from fairtest.bugreport.statistics import fairness_measures as fm

import pydot
import operator
import numpy as np
from collections import Counter
from ete2 import Tree, TreeStyle, AttrFace, TextFace, faces
from sklearn.externals import six
from sklearn.externals.six import StringIO 
from pyspark.sql import SQLContext

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
            counts = data.select(feature).rdd.countByValue()
            counts = {row[0]: count for row, count in counts.items()}
            values = sorted(counts.keys())

            if len(values) <= num_bins:
                # there are less than num_bins values
                thresholds[feature] = (np.array(values[0:-1]) + np.array(values[1:]))/2.0
            else:
                # binning algorithm from Spark. Build 'num_bins' bins of roughly
                # equal sample size
                approx_size = (1.0*data.count()) / (num_bins + 1)
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
    
    # Dependency measure (Mutual Info or Correlation)
    MI = 'MI'
    CORR = 'CORR'
    MEASURES = [MI, CORR]
    
    # Child-score aggregation (weighted average, average or max)
    WEIGHTED_AVG = 'WEIGHTED_AVG'
    AVG = 'AVG'
    MAX = 'MAX'
    AGG_TYPES = [WEIGHTED_AVG, AVG, MAX]

    def __init__(self, measure, agg_type, conf):
        self.measure = measure
        self.agg_type = agg_type
        self.conf = conf

#
# Split parameters
#
class SplitParams:
    def __init__(self, target, dim, categorical, thresholds, min_leaf_size):
        self.target = target
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
def build_tree(sc, data, dim, categorical, max_depth=5, min_leaf_size=100, measure="MI", agg_type="WEIGHTED_AVG", conf=None, max_bins=20):
    t = Tree()
    
    sqlContext = SQLContext(sc)
    data = sqlContext.createDataFrame(data)
    
    target = data.columns[0]
    features = set(data.columns[1:])
    
    # bin continuous features
    cont_thresholds = find_thresholds(data, features, categorical, max_bins)
    
    print cont_thresholds
    
    score_params = ScoreParams(measure, agg_type, conf)
    split_params = SplitParams(target, dim, categorical, cont_thresholds, min_leaf_size)
    
    #
    # Builds up the tree recursively. Selects the best feature to split on,
    # in order to maximize the average bias (mutual information) in all 
    # sub-trees.
    #
    def rec_build_tree(node_data, node, pred, node_features, depth):
        node.add_features(size=node_data.count())
        
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
            data_left = node_data.filter(node_data[best_feature] <= threshold)
            data_right = node_data.filter(node_data[best_feature] > threshold)
            
            # predicates for sub-trees
            pred_left = "{} <= {}".format(best_feature, threshold)
            pred_right = "{} > {}".format(best_feature, threshold)
            
            # add new nodes to the underlying tree structure
            left_child = node.add_child(name=str(pred_left))
            left_child.add_features(feature_type='continuous', feature=best_feature, threshold=threshold, is_left=True)
            
            right_child = node.add_child(name=str(pred_right))
            right_child.add_features(feature_type='continuous', feature=best_feature, threshold=threshold, is_left=False)
            
            # recursively build the tree
            rec_build_tree(data_left, left_child, pred+[pred_left], node_features.difference(set(to_drop)), depth+1)
            rec_build_tree(data_right, right_child, pred+[pred_right], node_features.difference(set(to_drop)), depth+1)
        else:
            # categorical split
            values = node_data.select(best_feature).distinct().collect()
            values = map(lambda row: int(row[0]), values)
            
            for val in values:
                data_child = node_data.filter(node_data[best_feature] == val)
                
                if (data_child.count() >= min_leaf_size):
                
                    # predicate for the current sub-tree
                    new_pred = "{} = {}".format(best_feature, val)
                    
                    # add a node to the underlying tree structure
                    child = node.add_child(name=str(new_pred))
                    child.add_features(feature_type='categorical', feature=best_feature, category=val)
                    
                    # recursively build the tree
                    rec_build_tree(data_child, child, pred+[new_pred], node_features.difference(set(to_drop)).difference(set(best_feature)), depth+1)
            
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
    max_mi = 0

    # keep track of useless features (no splits available)
    to_drop = []
    
    categorical = split_params.categorical
    target = split_params.target
    
    # iterate over all available non-sensitive features
    for feature in features:
        # determine type of split
        if feature in categorical:
            mi_cond = test_cat_feature(node_data[[feature, target]], feature, split_params, score_params)
            threshold = None
        else:
            mi_cond, threshold = test_cont_feature(node_data[[feature, target]], feature, split_params, score_params)
        
        # the feature produced no split and can be dropped for future sub-trees
        if not mi_cond:
            to_drop.append(feature)
        
        # check quality of split            
        if mi_cond > max_mi:
            max_mi = mi_cond
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
    values = [0]*(dim[0]*dim[1])
    counter = Counter(data)
    for i in range(len(values)):
        values[i] = counter.get(i, 0)
    
    return (np.array(values).reshape((dim[0], dim[1])), sum(values))

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
    dim = split_params.dim
    min_leaf_size = split_params.min_leaf_size
    
    # split data based on the feature value, and aggregate targets
    #ct = {key: countValues(group[target], dim) for key, group in node_data.groupby(feature)}    
    crosstab = node_data.crosstab(feature, target).collect()
    
    ct = {}
    
    for row in crosstab:
        key = row[0]
        counter = row.asDict()
        
        values = [0]*(dim[0]*dim[1])
        for i in range(len(values)):
            values[i] = counter.get(str(i), 0)
        ct[int(key)] = ((np.array(values).reshape((dim[0], dim[1])), sum(values)))
        
    # prune small sub-trees
    ct = {key:group for (key,(group,size)) in ct.iteritems() if size >= min_leaf_size}
    
    mi = None
    # compute the split score
    if len(ct) > 1:
        mi = score(ct.values(), score_params)
    
    return mi


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
    dim = split_params.dim
    min_leaf_size = split_params.min_leaf_size
    thresholds = split_params.thresholds[feature]
    
    max_mi = None
    best_threshold = None
    
    # split data based on the bin thresholds
    # groups = node_data.groupby(np.digitize(node_data[feature], thresholds, right=True))
    # aggregate all the target counts for each bin
    # temp = map(lambda (key, group): (key, countValues(group[target], dim)), groups)
    
    groups = {}
    for key in range(len(thresholds)):
        if (key == 0):
            group = node_data.filter(node_data[feature] <= thresholds[0])
        elif (key == len(thresholds) - 1):
            group = node_data.filter(node_data[feature] > thresholds[key])
        else:
            group = node_data.filter((node_data[feature] > thresholds[key]) & (node_data[feature] <= thresholds[key+1]))
            
            
        counter = dict(map(lambda row: (row[0], row[1]), group.groupby(target).count().collect()))
        values = [0]*(dim[0]*dim[1])
        for i in range(len(values)):
            values[i] = counter.get(str(i), 0)
        groups[key] = ((np.array(values).reshape((dim[0], dim[1])), sum(values)))
    
    # get the indices of the bin thresholds
    keys = groups.keys()
    temp = groups.values()
    
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
        mi = score([data_left, data_right], score_params)
        max_mi = mi
        best_threshold = thresholds[keys[0]]
    
    # check all further splits in order
    for i in range(1, len(bins)):    
        (ct_i, size_i) = (bins[i], sizes[i])
        data_left += ct_i
        data_right -= ct_i
        size_left += size_i
        size_right -= size_i
        
        if (size_left >= min_leaf_size) and (size_right >= min_leaf_size):
            mi = score([data_left, data_right], score_params)
        
            if mi > max_mi:
                max_mi = mi
                best_threshold = thresholds[keys[i]]
            
    return max_mi, best_threshold

#
# Compute the score for a split
#
# @args child_cts       Contingency tables for all the children
# @args score_params    Split scoring parameters
# 
def score(child_cts, score_params):
    measure = score_params.measure
    agg_type = score_params.agg_type
    conf = score_params.conf
    
    # use confidence intervals or not
    if conf:
        mi_list = map(lambda ct: fm.mutual_info(ct, norm=False, ci=True, level=conf), child_cts)
        mi_list = map(lambda (mi, delta): max(0, mi-delta), mi_list)
    else:
        mi_list = map(lambda ct: fm.mutual_info(ct, norm=False), child_cts)
    
    # take the (weighted) average or maximum of the children scores
    if agg_type == ScoreParams.WEIGHTED_AVG:
        totals = map(lambda ct: ct.sum().sum(), child_cts)
        probas = map(lambda tot: (1.0*tot)/sum(totals), totals)
        return np.dot(mi_list, probas)
    elif agg_type == ScoreParams.AVG:
        return np.mean(mi_list)
    elif agg_type == ScoreParams.MAX:
        return max(mi_list)

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
        