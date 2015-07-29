import pandas as pd
import numpy as np
import sklearn.cross_validation as cross_validation
import sklearn.preprocessing as preprocessing

#
# Class to represent a dataset and do some pre-processing
#
class Dataset:
    
    #
    # Load data in csv format, with a header indicating the column names
    #
    # @args filepath    the path to the csv file
    #
    def load_data_csv(self, filepath):
        self.original_data = pd.read_csv(
            filepath,
            header=0,
            sep=r'\s*,\s*',
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
        self.original_data = self.original_data.drop(feature, axis=1)
    
    #
    # Set the sensitive feature
    #
    # @args feature the sensitive feature
    #    
    def set_sens_feature(self, feature):
        self.SENS = feature
    
    #
    # Set the output feature
    #
    # @args feature the output feature
    #   
    def set_output_feature(self, feature):
        self.OUT = feature
    
    #
    # Pre-process and encode the data
    #
    # @args binary  If true, categorical features are binarized (one-hot-encoding)
    #       
    def encode_data(self, binary=False):
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
        
        # encode sensitive feature as numbers
        self.encoders[self.SENS] = preprocessing.LabelEncoder()
        self.encoded_data[self.SENS] = self.encoders[self.SENS].fit_transform(self.original_data[self.SENS])
        
        # encode output feature as numbers 
        self.encoders[self.OUT] = preprocessing.LabelEncoder()
        self.encoded_data[self.OUT] = self.encoders[self.OUT].fit_transform(self.original_data[self.OUT])
        
    #
    # Split the dataset into a training and a testing set
    #
    # @args train_size  the size of the training set (absolute or fraction of total)
    # @args seed        random seed for the split
    #
    def train_test_split(self, train_size, seed):
        self.data_train, self.data_test = cross_validation.train_test_split(self.encoded_data, train_size=train_size, random_state=seed)