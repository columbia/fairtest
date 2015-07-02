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

# Cluster types
C_LEAF      = 1
C_ROOT      = 2
C_INTERNAL  = 3

#
# class for storing and analyzing a cluster
#
class Cluster:

    #
    # Constructor
    #
    # @args num              the cluster's number
    # @args path             the predicate path leading from the root to the cluster
    # @args ctype            the cluster type
    # @args data             the data contained in this cluster
    # @args sens             the name of the sensitive feature
    # @args out              the name of the algorithmic output feature
    # @args business         the name of the business feature
    # @args df               the number of degrees of freedom
    #
    def __init__(self, num, path, ctype, data, sens, out, business, df):
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

    #
    # Analyze the cluster's contingency table
    #
    # @args correction   If true, Yate's correction for continuity is applied
    #
    def analyse_ct(self, correction=False, exact=False):
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
                self.mi += p_ct*metrics.mutual_info_score(None, None, contingency=ct)

                G, _, _, _ = chi2_contingency(ct, correction, 
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
         ret += 'G-Test = {}, p-value = {}, p-value adj = {}, MI = {}\n'.format(self.G,self.p,self.adj_p,self.mi)
         ret += '\n'.join([str(ct_) for ct_ in self.ct])
         return ret
#
# Traverse the tree and output clusters for each node
#
# @targs ree             the decision tree
# @args data             the dataset
# @args feature_names    the names of the dataset features
# @args sens             the name of the sensitive feature
# @args out              the name of the algorithmic output feature
# @args business         the name of the business feature
#        
def find_clusters(tree, data, feature_names, sens, out, business=None):
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

        #
        # Simple BFS to traverse the tree
        #
        # @node         the current node
        # @data_temp    the sub-dataset rooted at this node
        # @feature_path The predicate path from the root to this node
        #
        def bfs(node, data_temp, feature_path):
            print feature_path
            #current node
            feature = features_array[node]
            threshold = threshold_array[node]

            #check node type
            ctype = C_ROOT if (node == 0)\
                else C_LEAF if (child_left_array[node] == TREE_LEAF)\
                else C_INTERNAL

            # build a cluster class and store it in the list
            clstr = Cluster(node, feature_path, ctype, data_temp, sens, out, business, df)
            clusters[node] = clstr

            if ctype is not C_LEAF:
                # recurse left, split data based on predicate and add predicate
                # to path
                data_left = data_temp[data_temp[feature] <= threshold]
                bfs(child_left_array[node], data_left, 
                    feature_path + [feature + ' <= ' + str(threshold)])

                # recurse left, split data based on predicate and add predicate
                # to path
                data_right = data_temp[data_temp[feature] > threshold]
                bfs(child_right_array[node], data_right, 
                    feature_path + [feature + ' > ' + str(threshold)])    

        # start bfs from the root with the full dataset and an empty path           
        bfs(0, data, [])
        return clusters




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
    
    SENS = "Gender"
    SENS_VALS = ["Gender_Female", "Gender_Male"]
    OUT  = "Target"
    OUT_VALS = ["Target_<=50K", "Target_>50K"]
    '''
    
    '''
    original_data = pd.read_csv(
        data,
        names=[
            "Gender", "Department", "Admitted"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")
    
    SENS = "Gender"
    SENS_VALS = ["Gender_Female", "Gender_Male"]
    OUT  = "Admitted"
    OUT_VALS = ["Admitted_No", "Admitted_Yes"]
    '''
    
    '''
    original_data = pd.read_csv(
        data,
        names=[
            "Gender", "Purpose", "Credit"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")
    
    SENS = "Gender"
    SENS_VALS = ["Gender_Female", "Gender_Male"]
    OUT  = "Credit"
    OUT_VALS = ["Credit_No", "Credit_Yes"]
    '''
    
    '''
    original_data = pd.read_csv(
        data,
        names=[
            "Distance", "Gender", "Race", "Income", "Price"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")
    
    SENS = "Gender"
    SENS_VALS = ["Gender_F", "Gender_M"]
    OUT  = "Target"
    OUT  = "Price"
    OUT_VALS = ["Price_low", "Price_high"]
    '''
    
    original_data = pd.read_csv(
        data,
        names=[
            "Distance", "Gender", "Race", "Income", "Price"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")
    
    SENS = "Gender"
    SENS_VALS = ["Gender_F", "Gender_M"]
    OUT  = "Target"
    OUT  = "Price"
    OUT_VALS = ["Price_low", "Price_high"]
    
    target = OUT_VALS[1].split('_')[1]
    for sens in SENS_VALS:
        sens = sens.split('_')[1]
        print '{}:'.format(sens)
        tot = len(original_data[original_data[SENS]==sens])
        tot_target = len(original_data[(original_data[SENS]==sens) & (original_data[OUT]==target)])
        print '{}, {:.2f}%'.format(tot, (100.0*tot_target)/tot)
    
    # <headingcell level=2>
    
    # Compute bias for married vs unmarried users
    
    # <markdowncell>
    
    # For Adult dataset only
    
    # <codecell>
    
    binary_data = pd.get_dummies(original_data)
    
    # Use a single target variable
    binary_data[OUT] = binary_data[OUT_VALS[1]]
    del binary_data[OUT_VALS[0]]
    del binary_data[OUT_VALS[1]]
    
    # <headingcell level=2>
    # Train-Test set split, deletion of sensitive attributes, normalization
    # <codecell>
    
    random.seed(1)
    
    # train-test set split
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            binary_data[binary_data.columns.difference([OUT])],
            binary_data[OUT],
            train_size=0.25
            )
    
    # create a test dataset to analyze clusters
    data = X_test.copy()
    data[OUT] = y_test
    data[SENS] = data[SENS_VALS[1]]
    del data[SENS_VALS[0]]
    del data[SENS_VALS[1]]
    
    # remove sensitive attributes for the decision-tree learning phase
    del X_train[SENS_VALS[0]]
    del X_train[SENS_VALS[1]]
    del X_test[SENS_VALS[0]]
    del X_test[SENS_VALS[1]]
    
    if BUSINESS != None:
        del X_train[BUSINESS]
        del X_test[BUSINESS]
        
    # build a normalized version of the data for logistic regression
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = scaler.transform(X_test)
        
    print '{} Training Samples, {} Test Samples'.format(len(y_train), len(y_test))
    
    # <headingcell level=1>
    
    # Training of Decision-Tree Classifier
    
    # <codecell>
    
    # decision tree attributes
    MAX_DEPTH   = 4
    MAX_LEAVES  = 12
    MIN_SAMPLES = 100
    
    cls = tree.DecisionTreeClassifier(criterion='entropy', max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES, random_state=0)
    
    # baseline logistic regression
    cls_base = linear_model.LogisticRegression(random_state=0)
    
    cls.fit(X_train, y_train)
    cls_base.fit(X_train_scaled, y_train)
    y_pred = cls.predict(X_test)
    y_pred_base = cls_base.predict(X_test_scaled)
    
    # compare accuracy of decision tree and logistic regression
    print "Tree Classifier:"
    print "Accuracy: {:.2f}%".format(100*metrics.accuracy_score(y_test, y_pred))
    print "Baseline:"
    print "Accuracy: {:.2f}%".format(100*metrics.accuracy_score(y_test, y_pred_base))
    
    # print the tree to a file
    
    # <headingcell level=1>
    
    # Cluster Analysis and Hypothesis Testing
    
    # <codecell>
    
    
    # find clusters corresponding to each tree node
    clusters = find_clusters(cls.tree_, data, X_train.columns, SENS, OUT, BUSINESS)
    
    # <codecell>
    
    # print cluster info
    for cluster in clusters:
        cluster.analyse_ct(exact=False)
        print cluster
        print '-'*80
    
    # <headingcell level=2>
    
    # Show p-values
    
    # <codecell>
    
    LEAVES_ONLY = True
    
    if LEAVES_ONLY:
        clusters = filter(lambda c: c.ctype is not C_INTERNAL, clusters)
    p_vals = map(lambda c: c.p, clusters)
    p_vals
    
    # <headingcell level=2>
    
    # Correct for multiple hypothesis testing
    
    # <codecell>
    
    (reject, adj_pvals, _, _) = multipletests(p_vals, alpha=0.05, method='holm')
    for idx, cluster in enumerate(clusters):
        if not LEAVES_ONLY or cluster.ctype is not C_INTERNAL:
            adj_p = adj_pvals[idx]
        else:
            adj_p = nan
    list(adj_pvals)
    
    # <headingcell level=2>
    
    # Print significant clusters
    
    # <codecell>
    
    for idx, cluster in enumerate(clusters):
        if reject[idx]:
            print cluster
            print '-'*80
    
def usage(argv):
    print ("Usage:<%s> <filename.data>") % argv[0]
    sys.exit(-1)


if __name__ == '__main__':
    sys.exit(main())
