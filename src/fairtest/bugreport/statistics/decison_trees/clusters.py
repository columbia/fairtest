import random
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.tree._tree import TREE_LEAF
from scipy.stats import chi2_contingency, chisqprob, fisher_exact

# Cluster types
C_LEAF      = 1
C_ROOT      = 2
C_INTERNAL  = 3

class Cluster:
    '''
    Class for storing and analyzing a cluster
    '''
    def __init__(self, num, path, ctype, data, sens, out, business, df):
        '''
        @num:       the cluster's number
        @path:      the predicate path leading from the root to the cluster
        @ctype:     the cluster type
        @data:      the data contained in this cluster
        @sens:      the name of the sensitive feature
        @out:       the name of the algorithmic output feature
        @business:  the name of the business feature
        @df:        the number of degrees of freedom
        '''
        self.num = num
        self.path = path
        self.ctype = ctype
        self.size = len(data)
        self.df = df

        # build a two-way or three-way contingency table 
        if business == None:
            self.ct = [pd.crosstab(data[out], data[sens])]
        else:
            grouped = data.groupby(business)
            self.ct = [pd.crosstab(d[out], d[sens]) for (_, d) in grouped]

    def analyse_ct(self, correction=False, exact=False):
        '''
        Analyze the cluster's contingency table

        @correction:   If true, Yate's correction for continuity is applied
        '''
        self.mi     = 0
        self.G      = 0
        self.p      = 0
        self.adj_p  = 0

        # loop over all contingency tables (multiple tables occur when a
        # business necessity attribute is provided)
        for i in range(len(self.ct)):
            ct = self.ct[i]

            # size of the table
            tot = sum([np.array(ct_).sum() for ct_ in self.ct])

            # 1-dimensional tables have a p-value of 0 
            if 1 not in ct.shape:

                # compute the probability of this table
                p_ct = (1.0*np.array(ct).sum())/tot

                # compute the mutual information
                self.mi += p_ct*metrics.mutual_info_score(None,
                                                          None,
                                                          contingency=ct)

                G, _, _, _ = chi2_contingency(ct,
                                              correction,
                                              lambda_="log-likelihood")
                # sum up all G statistics
                self.G += G
        # compute the p-value
        self.p = chisqprob(self.G, self.df)
        if (exact and len(self.ct) == 1):
            _, self.p = fisher_exact(self.ct[0])

    def __str__(self):
         ctype = "LEAF" if (self.ctype == C_LEAF)\
            else "ROOT" if (self.ctype == C_ROOT)\
            else "INTERNAL"
         ret  = '{} node {} of size {}\n'.format(ctype, self.num, self.size)
         ret += '{}\n'.format(self.path)
         ret += 'G-Test = {},\
                 p-value = {},\
                 p-value adj = {},\
                 MI = {}\n'.format(self.G,self.p,self.adj_p,self.mi)
         ret += '\n'.join([str(ct_) for ct_ in self.ct])
         return ret


def find_clusters(tree, data, feature_names, sens, out, business=None):
    '''
    Traverse the tree and output clusters for each node

    @out:              the name of the algorithmic output feature
    @tree:             the decision tree
    @data:             the dataset
    @sens:             the name of the sensitive feature
    @business:         the name of the business feature
    @feature_names:    the names of the dataset features
    '''
    child_left_array      = tree.children_left
    child_right_array     = tree.children_right
    threshold_array       = tree.threshold
    features_array        = [feature_names[i] for i in tree.feature]

    # list of clusters
    clusters = [None]*tree.node_count
    print clusters
    print tree.node_count
    # compute degrees of freedom
    df = (len(data[sens].unique())-1)*(len(data[out].unique())-1)
    if business != None:
        df *= (len(data[business].unique())-1)

    def bfs(node, data_temp, feature_path):
        '''
        Simple BFS to traverse the tree

        @node:         the current node
        @data_temp:    the sub-dataset rooted at this node
        @feature_path: The predicate path from the root to this node
        '''
        print feature_path
        #current node
        feature = features_array[node]
        threshold = threshold_array[node]

        #check node type
        ctype = C_ROOT if (node == 0)\
            else C_LEAF if (child_left_array[node] == TREE_LEAF)\
            else C_INTERNAL

        # build a cluster class and store it in the list
        clstr = Cluster(node, feature_path, ctype, data_temp,
                        sens, out, business, df)
        clusters[node] = clstr

        if ctype is not C_LEAF:
            # recurse left, split data based on predicate 
            # and add predicate to path
            data_left = data_temp[data_temp[feature] <= threshold]
            bfs(child_left_array[node], data_left, 
                feature_path + [feature + ' <= ' + str(threshold)])

            # recurse left, split data based on predicate and add predicate
            # to path
            data_right = data_temp[data_temp[feature] > threshold]
            bfs(child_right_array[node], data_right, 
                feature_path + [feature + ' > ' + str(threshold)])

    # start bfs from the root with the
    # full dataset and an empty path
    bfs(0, data, [])
    return clusters
