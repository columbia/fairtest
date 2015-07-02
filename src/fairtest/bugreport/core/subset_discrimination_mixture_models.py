import pandas as pd
import numpy as np
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
from sklearn.externals.six import StringIO 
from statsmodels.sandbox.stats.multicomp import multipletests
import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
import random
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import sklearn.tree as tree
from sklearn.tree._tree import TREE_LEAF
from scipy.stats import chi2_contingency, chisqprob, fisher_exact

## Cluster types
#C_LEAF      = 1
#C_ROOT      = 2
#C_INTERNAL  = 3
#
##
## class for storing and analyzing a cluster
##
#class Cluster:
#
#    #
#    # Constructor
#    #
#    # @args num              the cluster's number
#    # @args path             the predicate path leading from the root to the cluster
#    # @args ctype            the cluster type
#    # @args data             the data contained in this cluster
#    # @args sens             the name of the sensitive feature
#    # @args out              the name of the algorithmic output feature
#    # @args business         the name of the business feature
#    # @args df               the number of degrees of freedom
#    #
#    def __init__(self, num, path, ctype, data, sens, out, business, df):
#        self.num = num
#        self.path = path
#        self.ctype = ctype
#        self.size = len(data)
#        self.df = df
#
#        # build a two-way or three-way contingency table 
#        if business == None:
#            self.ct = [pd.crosstab(data[out], data[sens])]
#        else:
#            grouped = data.groupby(business)
#            self.ct = [pd.crosstab(d[out], d[sens]) for (_, d) in grouped]
#
#    #
#    # Analyze the cluster's contingency table
#    #
#    # @args correction   If true, Yate's correction for continuity is applied
#    #
#    def analyse_ct(self, correction=False, exact=False):
#        self.mi     = 0
#        self.G      = 0
#        self.p      = 0
#        self.adj_p  = 0
#
#        # loop over all contingency tables (multiple tables occur when a
#        # business necessity attribute is provided)
#        for i in range(len(self.ct)):
#            ct = self.ct[i]
#
#            # size of the table
#            tot = sum([np.array(ct_).sum() for ct_ in self.ct])
#
#            # 1-dimensional tables have a p-value of 0 
#            if 1 not in ct.shape:
#
#                # compute the probability of this table
#                p_ct = (1.0*np.array(ct).sum())/tot
#
#                # compute the mutual information
#                self.mi += p_ct*metrics.mutual_info_score(None, None, contingency=ct)
#
#                for i in range(0,1275120):
#                    print i
#                    G, _, _, _ = chi2_contingency(ct, correction, 
#                                      lambda_="log-likelihood")
#
#                # sum up all G statistics
#                self.G += G
#
#        # compute the p-value
#        self.p = chisqprob(self.G, self.df)
#
#        if (exact and len(self.ct) == 1):
#            _, self.p = fisher_exact(self.ct[0])
#
#    def __str__(self):
#         ctype = "LEAF" if (self.ctype == C_LEAF)\
#            else "ROOT" if (self.ctype == C_ROOT)\
#            else "INTERNAL"
#    
#         ret  = '{} node {} of size {}\n'.format(ctype, self.num, self.size)
#         ret += '{}\n'.format(self.path)
#         ret += 'G-Test = {} ;\
#                p-value = {} ;\
#                p-value adj = {} ;\
#                MI = {}\n'.format(self.G,self.p,self.adj_p,self.mi)
#         ret += '\n'.join([str(ct_) for ct_ in self.ct])
#         return ret
##
## Traverse the tree and output clusters for each node
##
## @targs ree             the decision tree
## @args data             the dataset
## @args feature_names    the names of the dataset features
## @args sens             the name of the sensitive feature
## @args out              the name of the algorithmic output feature
## @args business         the name of the business feature
##        
#def find_clusters(tree, data, feature_names, sens, out, business=None):
#        child_left_array      = tree.children_left
#        child_right_array     = tree.children_right
#        threshold_array       = tree.threshold
#        features_array        = [feature_names[i] for i in tree.feature]
#
#
#        # list of clusters
#        clusters = [None]*tree.node_count
#        print clusters
#        print tree.node_count
#        # compute degrees of freedom
#        df = (len(data[sens].unique())-1)*(len(data[out].unique())-1)
#        if business != None:
#            df *= (len(data[business].unique())-1)
#
#        #
#        # Simple BFS to traverse the tree
#        #
#        # @node         the current node
#        # @data_temp    the sub-dataset rooted at this node
#        # @feature_path The predicate path from the root to this node
#        #
#        def bfs(node, data_temp, feature_path):
#            print "---------->", feature_path
#            #current node
#            feature = features_array[node]
#            threshold = threshold_array[node]
#
#            #check node type
#            ctype = C_ROOT if (node == 0)\
#                else C_LEAF if (child_left_array[node] == TREE_LEAF)\
#                else C_INTERNAL
#
#            # build a cluster class and store it in the list
#            clstr = Cluster(node, feature_path, ctype, data_temp, sens, out, business, df)
#            clusters[node] = clstr
#
#            if ctype is not C_LEAF:
#                # recurse left, split data based on predicate and add predicate
#                # to path
#                data_left = data_temp[data_temp[feature] <= threshold]
#                bfs(child_left_array[node], data_left, 
#                    feature_path + [feature + ' <= ' + str(threshold)])
#
#                # recurse left, split data based on predicate and add predicate
#                # to path
#                data_right = data_temp[data_temp[feature] > threshold]
#                bfs(child_right_array[node], data_right, 
#                    feature_path + [feature + ' > ' + str(threshold)])    
#
#        # start bfs from the root with the full dataset and an empty path           
#        bfs(0, data, [])
#        return clusters
#################################################################
import sys

def main(argv=sys.argv):
    BUSINESS = None

    if len(argv) != 2:
        usage(argv)
    data = argv[1]
    '''
    original_data = pd.read_csv(
        data,
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Race", "Gender", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")
    
    sensitive_attr = "Gender"
    sensitive_attr_vals = {}
    sensitive_attr_vals['Gender'] = ["Female", "Male"]
    output  = "Target"
    output_vals = ["<=50K", ">50K"]
    '''
    
    '''
    original_data = pd.read_csv(
        data,
        names=[
            "Gender", "Department", "Admitted"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")
    
    sensitive_attr = "Gender"
    sensitive_attr_vals = {}
    sensitive_attr_vals['Gender'] = ["Female", "Male"]
    output  = "Admitted"
    output_vals = ["No", "Yes"]
    '''
    
    original_data = pd.read_csv(
        data,
        names=[
            "Distance", "Gender", "Race", "Income", "Price"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")
    
    output = "Price"
    
    sensitive_attr = 'Race'
    sensitive_attr_vals = {}
    sensitive_attr_vals['Race'] = ["Some Other Race",\
            "Native Hawaiian and Other Pacific Islander",\
            "Hispanic or Latino", "Black or African American",\
            "Asian", "American Indian and Alaska Native",\
            "White Not Hispanic or Latino",\
            "Two or More Races"]
    
    sensitive_attr = 'Income'
    sensitive_attr_vals['Income'] = ["10000<=income<20000",\
            "160000<=income<320000", "20000<=income<40000",\
            "320000<=income", "40000<=income<80000",\
            "5000<=income<10000", "80000<=income<160000",\
            "income<5000"]
    
    output_vals = ["low", "high"]
    
    '''
    original_data = pd.read_csv(
        data,
        names=[
            "Gender", "Purpose", "Credit"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")
    
    output = 'Credit'
    sensitive_attr = 'Gender'
    sensitive_attr_vals = {}
    sensitive_attr_vals['Gender'] = ["Female", "Male"]
    output_vals = ["Yes", "No"]
    '''
    
    BUSINESS = None
    original_data.tail()
    original_data = original_data.reindex(
            np.random.permutation(original_data.index)
            )
    binary_data = pd.get_dummies(original_data)
    binary_data_copy =  binary_data.copy()
    
    #remove sensitive attributes
    for attr in sensitive_attr_vals:
        for attr_val in sensitive_attr_vals[attr]:
           del binary_data_copy[attr + "_" + attr_val]
    
    binary_data_copy = 10 * binary_data_copy
    components = 2
    covariance_type = 'diag'
    dpgmm = mixture.DPGMM(n_components=components,covariance_type=covariance_type)
    dpgmm.fit(binary_data_copy)
    
    for i in range(0, components):
    
        freq_dict = {}
        for attr_val in sensitive_attr_vals[sensitive_attr]:
            freq_dict[attr_val] = {}
            for output_val in output_vals:
                freq_dict[attr_val][output_val] = 0
    
        current = dpgmm.predict(binary_data_copy)
    
        #print zip(original_data[current == i][output], original_data[current == i][sensitive_attr])
        for (a, b) in zip(original_data[current == i][output],\
                          original_data[current == i][sensitive_attr]):
            freq_dict[b][a] += 1
    
        # convert dictionary into a table
        freq_table = [ [freq_dict[attr_val][val] for val in freq_dict[attr_val]]\
                        for attr_val in freq_dict]
    
        print "\n"
        print "Cluster:", i, "\nlength:", len(original_data[current == i])
        #print freq_table
        columns = list(binary_data_copy.columns.values)
        for j, mean in enumerate(dpgmm.means_[i]):
            if mean > 5:
                print columns[j],':', mean
        print "----"
        for attr_val in freq_dict:
            for output_val in freq_dict[attr_val]:
                print output_val, "\t",
            print "----",
            print ""
            break
    
        for attr_val in freq_dict:
            for output_val in freq_dict[attr_val]:
                print freq_dict[attr_val][output_val], "\t", 
            print attr_val,
            print ""
        
        try:
            (_, p_value, _, _) = chi2_contingency(freq_table, lambda_="log-likelihood")
            print "p-value:", p_value
        except Exception, error:
            print error


def usage(argv):
    print ("Usage:<%s> <filename.data>") % argv[0]
    sys.exit(-1)

if __name__ == '__main__':
    sys.exit(main())
