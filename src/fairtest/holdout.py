"""
Holdout For Adaptive Data Analysis
"""

from sklearn.cross_validation import train_test_split as cv_split
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from copy import copy


class Holdout(object):
    """
    A holdout set for validating statistical hypotheses.
    Splits a testing set into multiple independent sets that can be
    used to validate successive adaptive investigations.
    """
    def __init__(self, data, budget, conf):
        """
        Initializes a Data Holdout.

        Parameters
        ----------
        data :
            the test dataset
        budget :
            the maximal number of adaptive investigations that will be performed
        conf :
            overall family-wide confidence
        """
        self._adaptive_budget = budget

        # set a confidence per adaptive investigation
        self.test_set_conf = conf ** (1.0 / budget)

        # split the test set into multiple independent holdout sets
        test_set_size = len(data)/budget
        self._test_sets = []
        for i in range(budget):
            self._test_sets.append(
                data.iloc[i * test_set_size:(i + 1) * test_set_size])

        logging.info('Testing Sizes %s' % [len(x) for x in self._test_sets])

    def get_test_set(self):
        """
        Obtain a new independent testing set
        """
        if len(self._test_sets) == 0:
            raise RuntimeError('Maximum number of %d adaptive investigations '
                               'has been reached. You need to create a new '
                               'hold out set!' % self._adaptive_budget)

        return self._test_sets.pop()


class DataSource(object):
    """
    A place holder for a training set and a holdout set
    """
    def __init__(self, data, budget=1, conf=0.95, train_size=0.5,
                 random_state=0):
        """
        Prepares a dataset for FairTest investigations. Encodes categorical
        features as numbers and separates the data into a training set and a
        holdout set.

        Parameters
        ----------
        data :
            the dataset to use
        budget :
            the maximal number of adaptive investigations that will be performed
        conf :
            overall family-wide confidence
        train_size :
            the number (or fraction) of data samples to use as a training set
        random_state :
            a random seed to be used for the random train-test split
        """
        if data is not None:
            if not isinstance(data, pd.DataFrame):
                raise ValueError('data should be a Pandas DataFrame')

            data = data.copy()

            if budget < 1:
                raise ValueError("budget parameter should be a positive integer")

            if not 0 < conf < 1:
                raise ValueError('conf should be in (0,1), Got %s' % conf)

            # encode categorical features
            encoders = {}
            for column in data.columns:
                if data.dtypes[column] == np.object:
                    encoders[column] = LabelEncoder()
                    data[column] = encoders[column].fit_transform(data[column])
                    logging.info('Encoding Feature %s' % column)

            train_data, test_data = cv_split(data, train_size=train_size,
                                             random_state=random_state)

            logging.info('Training Size %d' % len(train_data))

            holdout = Holdout(test_data, budget, conf)

            self.train_data = train_data
            self.holdout = holdout
            self.encoders = encoders

    def duplicate(self):
        """
        Duplicates this Data Source and makes a copy of the training set.
        """
        new_source = DataSource(data=None)

        new_source.train_data = self.train_data.copy()
        new_source.holdout = self.holdout
        new_source.encoders = copy(self.encoders)
        return new_source

