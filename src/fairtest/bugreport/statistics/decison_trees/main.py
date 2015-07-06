import sys
import random
import numpy as np
import pandas as pd
import sklearn.tree as tree
import sklearn.metrics as metrics
import sklearn.linear_model as linear_model
import sklearn.preprocessing as preprocessing
import sklearn.cross_validation as cross_validation
from statsmodels.sandbox.stats.multicomp import multipletests

from clusters import find_clusters

# Cluster types
C_LEAF = 1
C_ROOT = 2
C_INTERNAL = 3

def main(argv=sys.argv):
    BUSINESS = None

    if len(argv) != 2:
        usage(argv)
    data = argv[1]

    '''
    original_data = pd.read_csv(
        data,
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num",
            "Martial Status", "Occupation", "Relationship", "Race",
            "Gender", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")

    SENS = "Gender"
    SENS_VALS = ["Gender_Female", "Gender_Male"]
    OUT = "Target"
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
    OUT = "Admitted"
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
    OUT = "Credit"
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
    OUT = "Target"
    OUT = "Price"
    OUT_VALS = ["Price_low", "Price_high"]
    '''

    original_data = pd.read_csv(
        data,
        names=["Distance", "Gender", "Race", "Income", "Price"],\
               sep=r'\s*,\s*',\
               engine='python',\
               na_values="?")

    SENS = "Gender"
    SENS_VALS = ["Gender_F", "Gender_M"]
    OUT = "Target"
    OUT = "Price"
    OUT_VALS = ["Price_low", "Price_high"]

    target = OUT_VALS[1].split('_')[1]
    for sens in SENS_VALS:
        sens = sens.split('_')[1]
        print '{}:'.format(sens)
        tot = len(original_data[original_data[SENS] == sens])
        tot_target = len(original_data[(original_data[SENS] == sens)
                         & (original_data[OUT] == target)])
        print '{}, {:.2f}%'.format(tot, (100.0*tot_target)/tot)

    binary_data = pd.get_dummies(original_data)

    # Use a single target variable
    binary_data[OUT] = binary_data[OUT_VALS[1]]
    del binary_data[OUT_VALS[0]]
    del binary_data[OUT_VALS[1]]
    random.seed(1)

    # train-test set split
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            binary_data[binary_data.columns.difference([OUT])],
            binary_data[OUT],
            train_size=0.25
            )
    # create a test dataset to analyze clusters
    data = X.copy()
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
    X_train_scaled = pd.DataFrame(scaler.transform(X_train),
                                  columns=X_train.columns)
    X_test_scaled = scaler.transform(X_test)
    print '{} Training Samples, {} Test Samples'.format(len(y_train), len(y_test))

    # decision tree attributes
    MAX_DEPTH = 4
    MAX_LEAVES = 12
    MIN_SAMPLES = 100
    cls = tree.DecisionTreeClassifier(criterion='entropy',
                                      max_depth=MAX_DEPTH,
                                      min_samples_leaf=MIN_SAMPLES,
                                      random_state=0)
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

    # find clusters corresponding to each tree node
    clusters = find_clusters(cls.tree_, data, X_train.columns, SENS, OUT, BUSINESS)

    # print cluster info
    for cluster in clusters:
        cluster.analyse_ct(exact=False)
        print cluster
        print '-'*80
    LEAVES_ONLY = True
    if LEAVES_ONLY:
        clusters = filter(lambda c: c.ctype is not C_INTERNAL, clusters)
    p_vals = map(lambda c: c.p, clusters)
    p_vals

    (reject, adj_pvals, _, _) = multipletests(p_vals, alpha=0.05, method='holm')
    for idx, cluster in enumerate(clusters):
        if not LEAVES_ONLY or cluster.ctype is not C_INTERNAL:
            adj_p = adj_pvals[idx]
        else:
            adj_p = nan
    list(adj_pvals)

    for idx, cluster in enumerate(clusters):
        if reject[idx]:
            print cluster
            print '-'*80

def usage(argv):
    print ("Usage:<%s> <filename.data>") % argv[0]
    sys.exit(-1)

if __name__ == '__main__':
    sys.exit(main())
