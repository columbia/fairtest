import pandas as pd
import numpy as np
import sklearn.cross_validation as cross_validation
import sklearn.preprocessing as preprocessing
import ast

#
# Class to represent a dataset and do some pre-processing
#
class Dataset:

    def __init__(self):
        self.original_data = None
        self.filepath = None
        self.features = None
        self.SENS = None
        self.SENS_TYPE = None
        self.OUT = None
        self.LABELS = None
        self.OUT_TYPE = None
        self.encoders = None
        self.encoded_data = None
        self.data_train = None
        self.data_test = None

    #
    # Load data in csv format, with a header indicating the column names
    #
    # @args filepath    the path to the csv file
    # @args separator   values separator
    #
    def load_data_csv(self, filepath, separator=','):
        self.original_data = pd.read_csv(
            filepath,
            header=0,
            sep=r'\s*{}\s*'.format(separator),
            engine='python',
            na_values="?")
        
        self.filepath = filepath
        self.features = self.original_data.columns
    
    #
    # Drop a feature (column) from the dataset
    #
    # @args feature the feature to drop
    #
    def drop_feature(self, feature):
        assert self.original_data is not None
        assert feature in self.original_data.columns
        
        self.original_data = self.original_data.drop(feature, axis=1)
    
    #
    # Set the sensitive feature
    #
    # @args feature         the sensitive feature
    # @args feature_type    the type of feature
    #    
    def set_sens_feature(self, feature, feature_type='cat'):
        assert feature_type in ['cat', 'cont']
        assert self.original_data is not None
        assert feature in self.original_data.columns
        
        self.SENS = feature
        self.SENS_TYPE = feature_type
    
    #
    # Set the output feature
    #
    # @args feature         the output feature
    # @args feature_type    the type of feature
    #   
    def set_output_feature(self, feature, feature_type='cat'):
        assert feature_type in ['cat', 'cont', 'labeled']
        assert self.original_data is not None
        assert feature in self.original_data.columns

        # single feature or list of features for multi-labeled output
        self.OUT = feature
        self.OUT_TYPE = feature_type
    
    #
    # Pre-process and encode the data
    #
    # @args binary  If true, categorical features are binarized (one-hot-encoding)
    #       
    def encode_data(self, binary=False):
        assert (self.SENS and self.OUT)
        assert self.original_data is not None
    
        # keep encoding information to decode feature before printing
        self.encoders = {}
        
        # keep all non-sensitive features
        self.encoded_data = self.original_data.drop([self.OUT, self.SENS], axis=1)

        if binary:
            # one-hot-encoding
            self.encoded_data = pd.get_dummies(self.encoded_data)
        else:
            # encode non-sensitive categorical features as numbers
            for column in self.encoded_data.columns:
                if self.original_data.dtypes[column] == np.object:
                    self.encoders[column] = preprocessing.LabelEncoder()
                    self.encoded_data[column] = self.encoders[column].fit_transform(self.original_data[column])
        
        if self.SENS_TYPE == 'cat':
            # encode sensitive feature as numbers
            self.encoders[self.SENS] = preprocessing.LabelEncoder()
            self.encoded_data[self.SENS] = self.encoders[self.SENS].fit_transform(self.original_data[self.SENS])
        else:
            self.encoded_data[self.SENS] = self.original_data[self.SENS]
        
        if self.OUT_TYPE == 'cat':
            # encode output feature as numbers 
            self.encoders[self.OUT] = preprocessing.LabelEncoder()
            self.encoded_data[self.OUT] = self.encoders[self.OUT].fit_transform(self.original_data[self.OUT])
        elif self.OUT_TYPE == 'labeled':

            # evaluate labels as a list (there is probably a better way to do this)
            labeled_data = map(lambda s: ast.literal_eval(s), self.original_data[self.OUT])

            # encode labels as a binary matrix (can we support a sparse matrix???)
            self.encoders[self.OUT] = preprocessing.MultiLabelBinarizer()
            labeled_data = self.encoders[self.OUT].fit_transform(labeled_data)
            labels = self.encoders[self.OUT].classes_
            df_labels = pd.DataFrame(labeled_data, columns=labels)
            self.encoded_data = pd.concat([self.encoded_data, df_labels], axis=1)

            # list of labels
            self.LABELS = labels
        else:
            self.encoded_data[self.OUT] = self.original_data[self.OUT]
        
    #
    # Split the dataset into a training and a testing set
    #
    # @args train_size  the size of the training set (absolute or fraction of total)
    # @args seed        random seed for the split
    #
    def train_test_split(self, train_size, seed):
        self.data_train, self.data_test = \
            cross_validation.train_test_split(self.encoded_data, train_size=train_size, random_state=seed)
