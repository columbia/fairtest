"""
Module for representing a dataset
"""
import pandas as pd
import numpy as np
import sklearn.cross_validation as cross_validation
import sklearn.preprocessing as preprocessing
import ast


class Dataset:
    """
    Class to represent a dataset and do some pre-processing
    """
    def __init__(self):
        self.original_data = None
        self.filepath = None
        self.features = None
        self.sens = None
        self.sens_type = None
        self.out = None
        self.labels = None
        self.out_type = None
        self.expl = None
        self.encoders = None
        self.encoded_data = None
        self.data_train = None
        self.data_test = None

    def load_data_csv(self, filepath, separator=','):
        """
        Load data in csv format, with a header indicating the column names

        Parameters
        ----------
        filepath :
            the path to the csv file

        separator :
            values separator
        """
        self.original_data = pd.read_csv(
            filepath,
            header=0,
            sep=r'\s*{}\s*'.format(separator),
            engine='python',
            na_values="?")

        self.filepath = filepath
        self.features = self.original_data.columns

    def drop_feature(self, feature):
        """
        Drop a feature (column) from the dataset

        Parameters
        ----------
        feature :
            the feature to drop
        """
        assert self.original_data is not None
        assert feature in self.original_data.columns

        self.original_data = self.original_data.drop(feature, axis=1)
        self.features = self.original_data.columns

    def set_sens_feature(self, feature, feature_type='cat'):
        """
        Set the sensitive feature

        Parameters
        ----------
        feature :
            the sensitive feature

        feature_type :
            the type of feature
        """
        assert feature_type in ['cat', 'cont']
        assert self.original_data is not None
        assert feature in self.original_data.columns

        self.sens = feature
        self.sens_type = feature_type

    def set_output_feature(self, feature, feature_type='cat'):
        """
        Set the output feature

        Parameters
        ----------
        feature :
            the output feature

        feature_type :
            the type of feature
        """
        assert feature_type in ['cat', 'cont', 'labeled']
        assert self.original_data is not None
        assert feature in self.original_data.columns

        # single feature or list of features for multi-labeled output
        self.out = feature
        self.out_type = feature_type

    def set_explanatory_feature(self, feature):
        assert self.original_data is not None
        assert feature in self.original_data.columns

        # single feature or list of features for multi-labeled output
        self.expl = feature

    def encode_data(self, binary=False):
        """
        Pre-process and encode the data

        Parameters
        ----------
        binary :
            If true, categorical features are binarized (one-hot-encoding)
        """
        assert self.sens and self.out
        assert self.original_data is not None

        # keep encoding information to decode feature before printing
        self.encoders = {}

        # keep all non-sensitive features
        self.encoded_data = self.original_data.drop([self.out, self.sens],
                                                    axis=1)
        if self.expl:
            self.encoded_data = self.encoded_data.drop(self.expl, axis=1)

        if binary:
            # one-hot-encoding
            self.encoded_data = pd.get_dummies(self.encoded_data)
        else:
            # encode non-sensitive categorical features as numbers
            for column in self.encoded_data.columns:
                if self.original_data.dtypes[column] == np.object:
                    self.encoders[column] = preprocessing.LabelEncoder()
                    self.encoded_data[column] = self.encoders[column].\
                            fit_transform(self.original_data[column])

        if self.sens_type == 'cat':
            # encode sensitive feature as numbers
            self.encoders[self.sens] = preprocessing.LabelEncoder()
            self.encoded_data[self.sens] = self.encoders[self.sens].\
                    fit_transform(self.original_data[self.sens])
        else:
            self.encoded_data[self.sens] = self.original_data[self.sens]

        if self.out_type == 'cat':
            # encode output feature as numbers 
            self.encoders[self.out] = preprocessing.LabelEncoder()
            self.encoded_data[self.out] = self.encoders[self.out].\
                    fit_transform(self.original_data[self.out])
        elif self.out_type == 'labeled':

            # evaluate labels as a list
            labeled_data = map(lambda s: ast.literal_eval(s),
                               self.original_data[self.out])

            # encode labels as a binary matrix (can we support a sparse matrix?)
            self.encoders[self.out] = preprocessing.MultiLabelBinarizer()
            labeled_data = self.encoders[self.out].fit_transform(labeled_data)
            labels = self.encoders[self.out].classes_
            df_labels = pd.DataFrame(labeled_data, columns=labels)
            self.encoded_data = pd.concat([self.encoded_data, df_labels],
                                          axis=1)
            # list of labels
            self.labels = labels
        else:
            self.encoded_data[self.out] = self.original_data[self.out]

        if self.expl:
            # encode explanatory feature
            self.encoders[self.expl] = preprocessing.LabelEncoder()
            self.encoded_data[self.expl] = self.encoders[self.expl].\
                    fit_transform(self.original_data[self.expl])

    def train_test_split(self, train_size, seed):
        """
        Split the dataset into a training and a testing set

        Parameters
        ----------
        train_size :
            the size of the training set (absolute or fraction of total)

        seed :
            random seed for the split
        """
        self.data_train, self.data_test = \
            cross_validation.train_test_split(self.encoded_data,
                                              train_size=train_size,
                                              random_state=seed)
