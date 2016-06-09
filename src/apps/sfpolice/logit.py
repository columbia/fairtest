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
from copy import deepcopy



trainDF=pd.read_csv("augmented_train.csv")

xy_scaler=preprocessing.StandardScaler()
xy_scaler.fit(trainDF[["X","Y"]])
trainDF[["X","Y"]]=xy_scaler.transform(trainDF[["X","Y"]])
trainDF=trainDF[abs(trainDF["Y"])<100]
trainDF.index=range(len(trainDF))


def parse_time(x):
    DD=datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
    time=DD.hour#*60+DD.minute
    day=DD.day
    month=DD.month
    year=DD.year
    return time,day,month,year


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


def parse_data(df,logodds,logoddsPA,train=True):
    feature_list=df.columns.tolist()
    if "Descript" in feature_list:
        feature_list.remove("Descript")
    if "Resolution" in feature_list:
        feature_list.remove("Resolution")
    if "Category" in feature_list:
        feature_list.remove("Category")
    if "Id" in feature_list:
        feature_list.remove("Id")

    for feature in [
        'ZipCode', 'Housing Units', 'Median Age', 'Total Population', 'White',
        'Black or African American', 'AIAN', 'Asian', 'NHOPI', 'Other',
        'Two or more races', '12:00am to 4:59am', '8:00am to 8:29am', '8:30am to 8:59am',
        '9:00am to 9:59am', '10:00am to 10:59am', '12:00pm to 3:59pm', '4:00pm to 11:59pm',
        'Family households', 'Nonfamily households', 'Less than high school graduate',
        'Median hh income 12 months', 'Aggregate hh income 12 months',
        'PPP income 12 months', 'Total Housing Units', 'Median number of rooms',
        'Median contract rent']:
        try:
            feature_list.remove(feature)
        except Exception:
            print feature

    cleanData=df[feature_list]
    cleanData.index=range(len(df))
    print("Creating address features")

    address_features=cleanData["Address"].apply(lambda x: logodds[x])
    address_features.columns=["logodds"+str(x) for x in range(len(address_features.columns))]

    print("Parsing dates")
    cleanData["Time"], cleanData["Day"], cleanData["Month"], cleanData["Year"]=zip(*cleanData["Dates"].apply(parse_time))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    print("Creating one-hot variables")
    dummy_ranks_PD = pd.get_dummies(cleanData['PdDistrict'], prefix='PD')
    dummy_ranks_DAY = pd.get_dummies(cleanData["DayOfWeek"], prefix='DAY')
    cleanData["IsInterection"]=cleanData["Address"].apply(lambda x: 1 if "/" in x else 0)
    cleanData["logoddsPA"]=cleanData["Address"].apply(lambda x: logoddsPA[x])
    print("droping processed columns")

    cleanData=cleanData.drop("PdDistrict",axis=1)
    cleanData=cleanData.drop("DayOfWeek",axis=1)
    cleanData=cleanData.drop("Address",axis=1)
    cleanData=cleanData.drop("Dates",axis=1)
    feature_list=cleanData.columns.tolist()

    print("joining one-hot features")
    if train:
        features = cleanData[feature_list].\
            join(dummy_ranks_DAY.ix[:,:]).\
            join(address_features.ix[:,:])
    else:
        features = cleanData[feature_list].\
            join(dummy_ranks_DAY.ix[:,:]).\
            join(address_features.ix[:,:])

    print("creating new features")
    features["IsDup"]=pd.Series(features.duplicated()|features.duplicated(take_last=True)).apply(int)
    features["Awake"]=features["Time"].apply(lambda x: 1 if (x==0 or (x>=8 and x<=23)) else 0)
    features["Summer"], features["Fall"], features["Winter"], features["Spring"]=zip(*features["Month"].apply(get_season))
    if "Category" in df.columns:
        labels = df["Category"].astype('category')
    else:
        labels=None
    return features,labels


addresses=sorted(trainDF["Address"].unique())
categories=sorted(trainDF["Category"].unique())
C_counts=trainDF.groupby(["Category"]).size()
A_C_counts=trainDF.groupby(["Address","Category"]).size()
A_counts=trainDF.groupby(["Address"]).size()
logodds={}
logoddsPA={}
MIN_CAT_COUNTS=2
default_logodds=np.log(C_counts/len(trainDF))-np.log(1.0-C_counts/float(len(trainDF)))
for addr in addresses:
    PA=A_counts[addr]/float(len(trainDF))
    logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
    logodds[addr]=deepcopy(default_logodds)
    for cat in A_C_counts[addr].keys():
        if (A_C_counts[addr][cat]>MIN_CAT_COUNTS) and A_C_counts[addr][cat]<A_counts[addr]:
            PA=A_C_counts[addr][cat]/float(A_counts[addr])
            logodds[addr][categories.index(cat)]=np.log(PA)-np.log(1.0-PA)
    logodds[addr]=pd.Series(logodds[addr])
    logodds[addr].index=range(len(categories))


features, labels=parse_data(trainDF,logodds,logoddsPA)
print(features.columns.tolist())
print(len(features.columns))


collist=features.columns.tolist()
scaler = preprocessing.StandardScaler()
scaler.fit(features)
features[collist]=scaler.transform(features)



#new_PCA=PCA(n_components=60)
#new_PCA.fit(features)
# print new_PCA.explained_variance_ratio_


sss = StratifiedShuffleSplit(labels, train_size=0.4)
for train_index, test_index in sss:
    features_train, features_test, features_test_original = features.iloc[train_index], features.iloc[test_index], trainDF.iloc[test_index]
    labels_train, labels_test = labels[train_index],labels[test_index]
features_test.index=range(len(features_test))
features_train.index=range(len(features_train))
labels_train.index=range(len(labels_train))
labels_test.index=range(len(labels_test))
features.index=range(len(features))
labels.index=range(len(labels))


features_train = features_train
labels_train = labels_train
features_test = features_test
labels_test = labels_test

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
if "X" in feature_list:
  feature_list.remove("X")
if "Y" in feature_list:
  feature_list.remove("Y")
if "ZipCode" in feature_list:
  feature_list.remove("ZipCode")
if "Address" in feature_list:
  feature_list.remove("Address")
if "Resolution" in feature_list:
  feature_list.remove("Resolution")
if "Description" in feature_list:
  feature_list.remove("Description")
if "Dates" in feature_list:
  feature_list.remove("Dates")
if "Time" in feature_list:
  feature_list.remove("Time")
if "Category" in feature_list:
  feature_list.remove("Category")
if "Descript" in feature_list:
  feature_list.remove("Descript")
features_test_original = features_test_original[feature_list]

with open("crime_pred_logloss.csv", "w") as f:
  print >> f, "%s" % (",".join(feature_list + ["logloss"]))
  for i, ind in enumerate(test_index):
    print >> f, "%s,%.10f" % (",".join(map(str,features_test_original.ix[ind].values.tolist())), losses[i])
  f.flush()

print("Test Loss: %.5f" % (sumloss / predictions.shape[0]))
print("test: %.5f" % log_loss(labels_test, model.predict_proba(features_test.as_matrix())))
