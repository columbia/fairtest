# -*- coding: utf-8 -*-

import fairtest.bugreport.statistics.fairness_measures as fm
from fairtest.bugreport.statistics.fairness_measures import Measure
import pandas as pd
import numpy as np
import ete2
from copy import deepcopy


#
# class for storing a cluster
# 
class Cluster:
    #
    # Constructor
    #
    # @args num         the cluster's number
    # @args path        the predicate path leading from the root to the cluster
    # @args isleaf      if the cluster is a tree leaf
    # @args isroot      if the cluster is the tree root
    # @args stats       the statistics for this cluster
    # @args size        the cluster size
    # @args data        any additional data required for fairness measures
    #
    def __init__(self, num, path, isleaf, isroot, stats, size, data=None):
        self.num = num
        self.path = path
        self.isleaf = isleaf
        self.isroot = isroot
        self.size = size
        self.stats = stats
        self.data = data


#
# A class to represent a bound for a continuous feature
#
class Bound:
    def __init__(self):
        self.lower = -float('inf')
        self.upper = float('inf')
    
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        ret = '(' + str(self.lower) + ', ' + str(self.upper)
        if self.upper == float('inf'):
            ret += ')'
        else:
            ret += ']'
        return ret


#
# Update the predicate for a continuous feature by updating the Bound object
#
def update_cont_path(feature_path, feature, lower_bound=None, upper_bound=None):
    bound = feature_path.get(feature, Bound())
    if lower_bound:
        bound.lower = lower_bound
    else:
        bound.upper = upper_bound
    
    feature_path[feature] = bound


#
# Traverse a categorical tree and output clusters for each node
#
# @args tree        the tree to traverse
# @args data        the dataset object
# @args train_set   if true, finds clusters in the training set
#        
def find_clusters_cat(tree, data, measure=fm.NMI(ci_level=0.95), train_set=False):
    # list of clusters
    clusters = []
    
    out = data.OUT
    sens = data.SENS
    encoders = data.encoders
    labels = data.LABELS
    
    if train_set:
        data = data.data_train
    else:
        data = data.data_test
    
    # assign an id to each node
    node_id = 0
    for tree_node in tree.traverse("levelorder"):
        tree_node.add_features(id=node_id)
        node_id += 1
    
    #
    # Simple BFS to traverse the tree
    #
    # @args node         the current node
    # @args data_node    the sub-dataset rooted at this node
    # @args feature_path The predicate path from the root to this node
    #
    def bfs(node, data_node, feature_path):
        is_root = node.is_root()
        is_leaf = node.is_leaf()
        
        # current node
        if not is_root:
            feature = node.feature
            
            # check type of feature split
            if node.feature_type == 'continuous':
                threshold = node.threshold
                
                # update the bound on the continuous feature
                if node.is_left:
                    update_cont_path(feature_path, feature, upper_bound=threshold)
                    data_node = data_node[data_node[feature] <= threshold]
                else:
                    update_cont_path(feature_path, feature, lower_bound=threshold)
                    data_node = data_node[data_node[feature] > threshold]
            else:
                # categorical split
                category = node.category
                feature_path[feature] = encoders[feature].inverse_transform([category])[0]
                data_node = data_node[data_node[feature] == category]

        if measure.dataType == Measure.DATATYPE_CT:
            # categorical data
            # create an empty contingency table
            ct = pd.DataFrame(0, index=range(len(encoders[out].classes_)), columns=range(len(encoders[sens].classes_)))

            # fill in available values
            ct = ct.add(pd.crosstab(data_node[out], data_node[sens]), fill_value=0)

            # replace numbers by original labels
            ct.index = encoders[out].classes_
            ct.index.name = out
            ct.columns = encoders[sens].classes_
            ct.columns.name = sens
            stats = ct
            size = len(data_node)
            cluster_data = None

        elif measure.dataType == Measure.DATATYPE_CORR:
            # continuous data
            # get data statistics for correlation computation
            sum_x = data_node[out].sum()
            sum_x2 = np.dot(data_node[out], data_node[out])
            sum_y = data_node[sens].sum()
            sum_y2 = np.dot(data_node[sens], data_node[sens])
            sum_xy = np.dot(data_node[out], data_node[sens])
            n = len(data_node)

            stats = [sum_x, sum_x2, sum_y, sum_y2, sum_xy, n]
            size = n
            cluster_data = {'values': data_node[[out, sens]]}

        else:
            # regression measure
            # keep all the data
            label_list = labels.tolist()
            stats = data_node[label_list + [sens]]
            size = len(data_node)
            cluster_data = {'labels': label_list}
        
        # build a cluster class and store it in the list
        clstr = Cluster(node.id, feature_path, is_leaf, is_root, stats, size, cluster_data)
        clusters.append(clstr)
        
        # recurse in children
        for child in node.get_children():
            bfs(child, data_node, deepcopy(feature_path))
    
    # start bfs from the root with the full dataset and an empty path           
    bfs(tree, data, {})
    return clusters


#
# Convert a scala list to a python list
#
def toList(scalaList):
    iterator = scalaList.iterator()
    ret = []
    while iterator.hasNext():
        ret.append(int(iterator.next()))
    
    return ret


#
# Traverse a spark tree and output clusters for each node
#
# @args topNode     the tree root
# @args data        the dataset object
# @args train_set   if true, finds clusters in the training set
#  
def find_clusters_spark(topNode, data, measure=fm.NMI(ci_level=0.95), train_set=False):
    # list of clusters
    clusters = []
    
    # replicate the Scala tree with an ete2 tree (for pretty printing)
    t = ete2.Tree()
    
    out = data.OUT
    sens = data.SENS
    encoders = data.encoders
    
    if train_set:
        data = data.data_train
    else:
        data = data.data_test
    
    # feature names
    features = data.columns.drop([out, sens])
    
    #
    # @param node           the current node
    # @param tree_node      the corresponding node in the ete2 tree
    # @param data_node      the data rooted at this node
    # @param feature_path   predicates leading to this node
    #
    def bfs(node, tree_node, data_node, feature_path):
        isleaf = node.isLeaf()
        isroot = node == topNode

        if measure.dataType == Measure.DATATYPE_CT:
            # categorical data
            # create and empty contingency table
            ct = pd.DataFrame(0, index=range(len(encoders[out].classes_)), columns=range(len(encoders[sens].classes_)))

            # fill in available values
            ct = ct.add(pd.crosstab(data_node[out], data_node[sens]), fill_value=0)

            # replace numbers by original labels
            ct.index = encoders[out].classes_
            ct.index.name = out
            ct.columns = encoders[sens].classes_
            ct.columns.name = sens
            stats = ct
            size = len(data_node)
            cluster_data = None
        else:
            # continuous data
            # get statistics for correlation
            sum_x = data_node[out].sum()
            sum_x2 = np.dot(data_node[out], data_node[out])
            sum_y = data_node[sens].sum()
            sum_y2 = np.dot(data_node[sens], data_node[sens])
            sum_xy = np.dot(data_node[out], data_node[sens])
            n = len(data_node)

            stats = [sum_x, sum_x2, sum_y, sum_y2, sum_xy, n]
            size = n
            cluster_data = data_node[[out, sens]]

        # build a cluster object and store it        
        clstr = Cluster(node.id(), feature_path, isleaf, isroot, stats, size, cluster_data)
        clusters.append(clstr)
        
        # append the cluster size to the ete2 tree node for future printing
        tree_node.add_features(size=len(data_node))
        
        if not isleaf:
            # get the number and name of the splitting feature
            feature_num = node.split().get().feature()
            feature = features[feature_num]
            
            # check type of feature (continuous or categorical)
            if node.split().get().featureType().toString() == 'Continuous':
                
                # continuous feature splitting threshold
                threshold = node.split().get().threshold()
                
                # predicates for the ete2 tree
                pred_left = "{} <= {}".format(feature, threshold)
                pred_right = "{} > {}".format(feature, threshold)
                
                # split data on threshold
                data_left = data_node[data_node[feature] <= threshold]
                data_right = data_node[data_node[feature] > threshold]
                
                # copy path dictionaries
                path_left = deepcopy(feature_path)
                path_right = deepcopy(feature_path)
                
                # update [low, high] bounds for the feature
                update_cont_path(path_left, feature, lower_bound=None, upper_bound=threshold)
                update_cont_path(path_right, feature, lower_bound=threshold, upper_bound=None)
            else:
                # categorical split
                categories = toList(node.split().get().categories())
                
                # split data based on categories
                data_left = data_node[data_node[feature].isin(categories)]
                data_right = data_node[~data_node[feature].isin(categories)]
                
                # check if split on same feature already occurred
                if feature in feature_path:
                    prev_categories = feature_path[feature]
                else:
                    prev_categories = set(encoders[feature].classes_)
                
                # update sub-lists of categories
                categories_names_left = set(encoders[feature].inverse_transform(categories))
                categories_names_right = prev_categories.difference(categories_names_left)
                
                # copy and update path dictionaries
                path_left = deepcopy(feature_path)
                path_right = deepcopy(feature_path)
                path_left[feature] = categories_names_left
                path_right[feature] = categories_names_right
                
                # predicates for ete2 tree
                pred_left = "{} in {}".format(feature, categories_names_left)
                pred_right = "{} in {}".format(feature, categories_names_right)
            
            # recurse on left and right sub-trees    
            left_child = tree_node.add_child(name=str(pred_left))
            right_child = tree_node.add_child(name=str(pred_right))
            bfs(node.leftNode().get(), left_child, data_left, path_left)
            bfs(node.rightNode().get(), right_child, data_right, path_right)
    
    # start bfs from the root with the full dataset and an empty path           
    bfs(topNode, t, data, {})
    
    return clusters, t