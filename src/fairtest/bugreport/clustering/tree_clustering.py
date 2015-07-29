# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from ete2 import Tree
import copy

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
    # @args ct          the contingency table (SENS x OUT) for this cluster
    def __init__(self, num, path, isleaf, isroot, ct):
        self.num = num
        self.path = path
        self.isleaf = isleaf
        self.isroot = isroot
        self.size = ct.sum().sum()
        self.ct = ct

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
def find_clusters_cat(tree, data, train_set=False):
    # list of clusters
    clusters = []
    
    out = data.OUT
    sens = data.SENS
    encoders = data.encoders
    
    if train_set:
        data = data.data_train
    else:
        data = data.data_test
    
    # assign an id to each node
    node_id = 0
    for node in tree.traverse("levelorder"):
        node.add_features(id=node_id)
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
        
        #current node
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
                #categorical split
                category = node.category
                feature_path[feature] = encoders[feature].inverse_transform([category])[0]
                data_node = data_node[data_node[feature] == category]
        
        # create an empty contingency table
        ct = pd.DataFrame(0, index=range(len(encoders[out].classes_)), columns=range(len(encoders[sens].classes_)))
        
        # fill in available values
        ct = ct.add(pd.crosstab(data_node[out], data_node[sens]), fill_value=0)
        
        # replace numbers by original labels
        ct.index = encoders[out].classes_
        ct.columns = encoders[sens].classes_
        
        # build a cluster class and store it in the list
        clstr = Cluster(node.id, feature_path, is_leaf, is_root, ct)
        clusters.append(clstr)
        
        # recurse in children
        for child in node.get_children():
            bfs(child, data_node, copy.deepcopy(feature_path))
    
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
def find_clusters_spark(topNode, data, train_set=False):
    # list of clusters
    clusters = []
    
    # replicate the Scala tree with an ete2 tree (for pretty printing)
    t = Tree()

    out = data.OUT
    sens = data.SENS
    encoders = data.encoders
    
    if train_set:
        data = data.data_train
    else:
        data = data.data_test
    
    # feature names
    features = data.columns.drop([out,sens])
    
    #
    # @param node           the current node
    # @param tree_node      the corresponding node in the ete2 tree
    # @param data_node      the data rooted at this node
    # @param feature_path   predicates leading to this node
    #
    def bfs(node, tree_node, data_node, feature_path):
        isleaf = node.isLeaf()
        isroot = node == topNode
                
        # create and empty contingency table
        ct = pd.DataFrame(0, index=range(len(encoders[out].classes_)), columns=range(len(encoders[sens].classes_)))
        
        # fill in available values
        ct = ct.add(pd.crosstab(data_node[out], data_node[sens]), fill_value=0)
        
        # replace numbers by original labels
        ct.index = encoders[out].classes_
        ct.columns = encoders[sens].classes_
        
        # build a cluster object and store it        
        clstr = Cluster(node.id(), feature_path, isleaf, isroot, ct)
        clusters.append(clstr)
        
        # append the cluster size to the ete2 tree node for future printing
        tree_node.add_features(size=len(data_node))
        
        if not isleaf:
            # get the number and name of the splitting feature
            feature_num = node.split().get().feature()
            feature = features[feature_num]
            
            # check type of feature (continuous or categorical)
            if (node.split().get().featureType().toString() == 'Continuous'):
                
                # continuous feature splitting threshold
                threshold = node.split().get().threshold()
                
                # predicates for the ete2 tree
                pred_left = "{} <= {}".format(feature, threshold)
                pred_right = "{} > {}".format(feature, threshold)
                
                # split data on threshold
                data_left = data_node[data_node[feature] <= threshold]
                data_right = data_node[data_node[feature] > threshold]
                
                # copy path dictionaries
                path_left = copy.deepcopy(feature_path)
                path_right = copy.deepcopy(feature_path)
                
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
                path_left = copy.deepcopy(feature_path)
                path_right = copy.deepcopy(feature_path)
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