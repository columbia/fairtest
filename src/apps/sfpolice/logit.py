"""
Reference:
https://www.kaggle.com/papadopc/sf-crime/neural-nets-and-address-featurization/run/43507
"""
import pandas as pd
import numpy as np
import scipy as sp
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from collections import Counter
from copy import deepcopy

import sklearn.ensemble as ensemble

import sys
import gzip
import pickle


def parse_time(x):
    DD = datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
    time = DD.hour
    day = DD.day
    month = DD.month
    year = DD.year

    return time, day, month, year


def get_season(x):
    summer=0
    fall=0
    winter=0
    spring=0
    if (x in [5, 6, 7]):
        summer=1
    if (x in [8, 9, 10]):
        fall=1
    if (x in [11, 0, 1]):
        winter=1
    if (x in [2, 3, 4]):
        spring=1
    return summer, fall, winter, spring


def parse_data(df, logodds, logoddsPA):
    feature_list = df.columns.tolist()
    for feature in [
        'Descript', 'Resolution', 'Category', 'Id',
        'ZipCode', 'Housing Units', 'Median Age', 'Total Population', 'White',
        'Black or African American', 'AIAN', 'Asian', 'NHOPI', 'Other',
        'Two or more races', '12:00am to 4:59am', '8:00am to 8:29am', '8:30am to 8:59am',
        '9:00am to 9:59am', '10:00am to 10:59am', '12:00pm to 3:59pm', '4:00pm to 11:59pm',
        'Family households', 'Nonfamily households', 'Less than high school graduate',
        'Median hh income 12 months', 'Aggregate hh income 12 months',
        'PPP income 12 months', 'Total Housing Units', 'Median number of rooms',
        'Median contract rent'
    ]:
        if feature in feature_list:
            feature_list.remove(feature)

    cleanData = df[feature_list]
    # print cleanData

    cleanData.index = range(len(df))
    # print cleanData.index
    print("Creating address features")

    address_features = cleanData["Address"].apply(lambda x: logodds[x])
    address_features.columns = [
        "logodds" + str(x) for x in range(len(address_features.columns))
    ]
    # print address_features


    print("Parsing dates")
    cleanData.loc[:, "Time"] = map(lambda x: x[0], map(parse_time,
                                                       cleanData.loc[:, 'Dates']))
    cleanData.loc[:, "Day"] = map(lambda x: x[1], map(parse_time,
                                                      cleanData.loc[:, 'Dates']))
    cleanData.loc[:, "Month"] = map(lambda x: x[2], map(parse_time,
                                                        cleanData.loc[:, 'Dates']))
    cleanData.loc[:, "Year"] = map(lambda x: x[3], map(parse_time,
                                                       cleanData.loc[:, 'Dates']))

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
            'Saturday', 'Sunday']

    print("Creating one-hot variables")
    dummy_ranks_PD = pd.get_dummies(cleanData.loc[:, 'PdDistrict'], prefix='PD')
    dummy_ranks_DAY = pd.get_dummies(cleanData.loc[:, "DayOfWeek"], prefix='DAY')

    cleanData.loc[:, "IsInterection"] = map(
        lambda x: 1 if "/" in x else 0, cleanData.loc[:, "Address"]
    )
    cleanData.loc[:, "logoddsPA"]= map(lambda x: logoddsPA[x], cleanData.loc[:, "Address"])
    print("droping processed columns")

    cleanData = cleanData.drop("PdDistrict",axis=1)
    cleanData = cleanData.drop("DayOfWeek",axis=1)
    cleanData = cleanData.drop("Address",axis=1)
    cleanData = cleanData.drop("Dates",axis=1)
    feature_list = cleanData.columns.tolist()

    print("joining one-hot features")
    features = cleanData[feature_list].join(dummy_ranks_DAY.ix[:,:]).\
        join(address_features.ix[:,:])

    print("creating new features")
    features["IsDup"] = map(
        int,
        pd.Series(features.duplicated() | features.duplicated(take_last=True))
    )
    features["Awake"] = map(
        lambda x: 1 if (x==0 or (x>=8 and x<=23)) else 0,
        features["Time"]
    )
    features["Summer"] = map(lambda x: x[0], map(get_season, features["Month"]))
    features["Fall"] = map(lambda x: x[1], map(get_season, features["Month"]))
    features["winter"] = map(lambda x: x[2], map(get_season, features["Month"]))
    features["Spring"] = map(lambda x: x[3], map(get_season, features["Month"]))
    # print features

    if "Category" in df.columns:
        labels = df["Category"].astype('category')
    else:
        labels = None

    return features, labels


def calculate_logodds(originalData):
    xy_scaler = preprocessing.StandardScaler()
    xy_scaler.fit(originalData[["X","Y"]])
    originalData[["X","Y"]] = xy_scaler.transform(originalData[["X","Y"]])
    originalData = originalData[abs(originalData["Y"]) < 100]
    originalData.index=range(len(originalData))
    # print originalData

    addresses = sorted(originalData["Address"].unique())
    categories = sorted(originalData["Category"].unique())
    A_counts = originalData.groupby(["Address"]).size()
    C_counts = originalData.groupby(["Category"]).size()
    A_C_counts = originalData.groupby(["Address","Category"]).size()

    logodds = {}
    logoddsPA = {}
    MIN_CAT_COUNTS = 2
    default_logodds = np.log(C_counts / len(originalData)) -\
        np.log(1.0 - C_counts / float(len(originalData)))

    for addr in addresses:
        PA = A_counts[addr] / float(len(originalData))
        logoddsPA[addr] = np.log(PA)-np.log(1.-PA)
        logodds[addr] = deepcopy(default_logodds)

        for cat in A_C_counts[addr].keys():
            if A_C_counts[addr][cat] > MIN_CAT_COUNTS\
            and A_C_counts[addr][cat] < A_counts[addr]:
                PA=A_C_counts[addr][cat] / float(A_counts[addr])
                logodds[addr][categories.index(cat)] = np.log(PA) - np.log(1.0 - PA)
        logodds[addr] = pd.Series(logodds[addr])
        logodds[addr].index = range(len(categories))
    return logodds, logoddsPA


def train_and_test_model1(features_train, labels_train, features_test,
                          labels_test, features_test_original):
    model = LogisticRegression()
    model.fit(features_train,labels_train)
    print("train", log_loss(labels_train, model.predict_proba(features_train.as_matrix())))

    cat_indexes = labels_test.cat.codes
    predictions = model.predict_proba(features_test.as_matrix())

    sumloss = .0
    losses = []
    for i in range(predictions.shape[0]):
      loss = (-1) * sp.log(max(min(predictions[i][cat_indexes[i]], 1 - 10**(-5)), 10**(-5)))
      sumloss += loss
      losses.append(loss)

    feature_list = features_test_original.columns.tolist()
    for feature in ["X","Y", "ZipCode", "Address", "Resolution", "Description",
                    "Dates", "Time", "Category", "Descript"]:
        if feature in feature_list:
            feature_list.remove(feature)
    feature_list_original  = ["X","Y", "ZipCode", "Address", "Resolution",
                              "Description", "Dates", "Time", "Category",
                              "Descript"]
    features_test_original = features_test_original[feature_list]
    print("Test Loss: %.5f" % (sumloss / predictions.shape[0]))
    print("test: %.5f" % log_loss(labels_test, model.predict_proba(features_test.as_matrix())))


def train_and_test_model2(features_train, labels_train, features_test,
                          labels_test, features_test_original, model_name):
    """
    The gradient boosting classifier with the following parameters achieves
    2.13395 test loss on augmented_train.csv with 429885 entries.

    GBM_DEPTH = 4
    GBM_MINOBS = 50
    GBM_NTREES = 500
    GBM_SHRINKAGE = 0.05

    """
    GBM_DEPTH = 4
    GBM_MINOBS = 50
    GBM_NTREES = 500
    GBM_SHRINKAGE = 0.05

    # model = ensemble.RandomForestClassifier(n_estimators=500, verbose=True)
    model = ensemble.GradientBoostingClassifier(
        n_estimators=GBM_NTREES,
        learning_rate=GBM_SHRINKAGE,
        max_depth=GBM_DEPTH,
        min_samples_leaf=GBM_MINOBS,
        verbose=1
    )
    model.fit(features_train,labels_train)

    with gzip.open(model_name, "wb") as f:
        pickle.dump(model, f)

    print("Train", log_loss(labels_train, model.predict_proba(features_train.as_matrix())))

    cat_indexes = labels_test.cat.codes
    predictions = model.predict_proba(features_test.as_matrix())
    feature_list = features_test.columns.tolist()
    features_test = features_test[feature_list]

    #_sum = .0
    #INF = 10**(-15)
    #for i in range(predictions.shape[0]):
    #    _sum -= sp.log(max(min(predictions[i][cat_indexes[i]], 1 - INF), INF))
    #print "=--->", _sum / float(len(predictions))

    print("Test: %.5f" % log_loss(labels_test, model.predict_proba(features_test.as_matrix())))


def pred_ints(model, X, Y, percentile=90, batch_size=1000):

    cat_indexes = Y.cat.codes
    std = np.zeros((len(X), Y.cat.categories.shape[0]))
    interval_size = np.zeros((len(X), Y.cat.categories.shape[0]))
    n_batches = (len(X) + batch_size - 1)/batch_size

    print "Batch Size: %d, Classifers: %d, Batches to Process: %d" %\
        (batch_size , len(model.estimators_), n_batches)

    for n_batch in range(n_batches):
        print "Batch:", n_batch
        idx = (n_batch * batch_size, min((n_batch + 1) * batch_size, len(X)))
        x = X[idx[0]: idx[1]]
        preds = np.zeros((idx[1] - idx[0], len(model.estimators_), Y.cat.categories.shape[0]))

        # for each point of the batch iterate over all classifiers of the
        # forest, and keep the size of interval and the standard deviation for
        # all of the regressors each classifier is made of. Note here that each
        # classifier (estimator) is actually composed of multiple TreeRegressors
        # -- one TreeRegressor for each classification category -- and this
        # leads to an additional dimension. Therefore, we first find the
        # confidence of the classifiers, and then the confidence for each
        # category.
        for (ii, classifier) in enumerate(model.estimators_):
            proba = np.array(map(lambda c: c.predict(x), classifier)).transpose()
            preds[:, ii, :] = proba

        err_down = np.percentile(preds, (100 - percentile) / 2. , axis=1)
        err_up = np.percentile(preds, 100 - (100 - percentile) / 2., axis=1)
        interval_size[idx[0]: idx[1]] = err_up -  err_down
        std[idx[0]: idx[1]] = np.std(preds, axis=1)

    err_down = np.percentile(interval_size, (100 - percentile) / 2. , axis=0)
    err_up  = np.percentile(interval_size, 100 - (100 - percentile) / 2., axis=0)
    interval_size = err_up -  err_down

    std = np.std(std, axis=0)

    return interval_size, std


def main(argv):
    if len(argv) != 3 and len(argv) != 4:
        print "%s <input_file> <model_name> [<train_flag>]"
        return

    INF = 10**(-15)
    train = False
    input_name = argv[1]
    model_name = argv[2]

    if len(argv) == 4:
        train = True

    originalData = pd.read_csv(input_name)
    logodds, logoddsPA = calculate_logodds(originalData)
    features, labels = parse_data(originalData, logodds, logoddsPA)
    collist = features.columns.tolist()
    scaler = preprocessing.StandardScaler()
    scaler.fit(features)

    features_original = features.copy()
    labels_original = labels.copy()

    features[collist] = scaler.transform(features)
    sss = StratifiedShuffleSplit(labels, train_size=0.5)
    for train_index, test_index in sss:
        features_train = features.iloc[train_index]
        features_test = features.iloc[test_index]
        labels_train = labels[train_index]
        labels_test = labels[test_index]
        break

    features_test_original = features_test.copy()
    labels_test_original = labels_test.copy()

    features_test.index = range(len(features_test))
    features_train.index = range(len(features_train))
    labels_train.index = range(len(labels_train))
    labels_test.index = range(len(labels_test))
    features.index = range(len(features))
    labels.index = range(len(labels))
    cat_indexes = labels_test.cat.codes

    if train:
        train_and_test_model2(features_train, labels_train, features_test, labels_test,
                              features_test_original, model_name)
    print("Loading Model...")
    model = pickle.load(gzip.open(model_name, "rb"))

    interval_sizes, stds = pred_ints(model, features_test, labels_test)
    # use std instead of percentile difference
    interval_sizes = stds
    median_interval_size = np.median(interval_sizes)
    max_interval_size = np.max(interval_sizes)
    min_interval_size = np.min(interval_sizes)

    print interval_sizes
    print "min_interval_size:", min_interval_size
    print "median_interval_size:", median_interval_size
    print "max_interval_size:", max_interval_size

    cat_conf = {}
    for i in range(stds.shape[0]):
        cat_conf[labels_test.cat.categories[i]] = interval_sizes[i]

    predictions = model.predict_proba(features_test_original.as_matrix())

    originalTestData = originalData.iloc[test_index].copy()
    originalTestData['Prediction Confidence'] =  map(
        lambda x: 'High Conf' if cat_conf[x] < median_interval_size else 'Low Conf',
        originalTestData['Category']
    )

    print(Counter(map(lambda x: 'High Conf' if x < median_interval_size
                      else 'Low Conf', interval_sizes)))

    print(Counter(map(lambda x: 'High Conf' if cat_conf[x] < median_interval_size else 'Low Conf',
                      originalTestData['Category'])))

    losses = np.zeros(originalTestData.shape[0])
    for i in range(originalTestData.shape[0]):
        losses[i] = max(min(predictions[i][cat_indexes[i]], 1 - INF), INF)
    originalTestData['logloss'] =  -1*sp.log(np.array(losses))

    for (val, g) in originalTestData.groupby('Prediction Confidence'):
        print '{}: mean logloss = {}'.format(val, (g['logloss']).mean())
    print "-"*30

    for (val, g) in originalTestData.groupby('Category'):
        print '{}: mean logloss = {}'.format(val, (g['logloss']).mean())
    print "-"*30


    for cat in sorted(cat_conf, key=cat_conf.get):
        conf = 'High Conf' if cat_conf[cat] <  median_interval_size else 'Low Conf'
        print cat_conf[cat], conf, cat
    print "-"*30

    feature_list = originalTestData.columns.tolist()
    for feature in ["X","Y", "ZipCode", "Address", "Resolution", "Description",
                    "Dates", "Time", "Descript"]:
        if feature in feature_list:
            feature_list.remove(feature)

    originalTestData = originalTestData[feature_list]
    originalTestData.to_csv("results.csv", delimiter=',', index=False)

    #  p50 =  np.percentile(originalData['Total Population'], 50)
    #  originalData = originalData.iloc[features_test_original.index]
    #  originalData = originalData[originalData['Category'] == 'LARCENY/THEFT' ]
    #  index_a = originalData[originalData['Total Population'] < p50].index
    #  index_b = originalData[originalData['Total Population'] >= p50].index
    #  index_ab = originalData.index
    #  print "<50th", features_test_original.loc[index_a, 'logloss'].mean()
    #  print ">=50th", features_test_original.loc[index_b, 'logloss'].mean()
    #  for (val, g) in features_test_original.loc[index_ab, :].groupby('Prediction Confidence'):
    #      print '{}: mean logloss = {}'.format(val, (g['logloss']).mean())
    #  print features_test_original.loc[index_ab, 'logloss'].mean()
    #  print features_test_original['logloss'].mean()

if __name__ == "__main__":
    main(sys.argv)
