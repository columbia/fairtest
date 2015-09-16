"""
Helper module to prepare data for a Fairtest Experiment
"""
import pandas as pd
import numpy as np


def data_from_csv(filename, sep=None, header=0, to_drop=[]):
    """
    Load data from csv into a FairTest friendly format

    Parameters:
    -----------
    filename :
        abs path odf the csv

    to_drop : opt
        A list with attributes (columns) to drop
        during loading the data
    """
    try:
        if sep:
            _sep = sep
        else:
            _sep = r'\s*,\s*'

        data = pd.read_csv(filename, header=header, sep=_sep, engine='python',
                           na_values="?")

        for attribute in to_drop:
            data = data.drop(attribute, axis=1)

    except IOError:
        print "Error: Cannot open file \"%s\"" % filename
        raise
    except Exception, error:
        print "Error: %s loading data from file \"%s\"" % (error, filename)
        raise

    return data

def data_from_mem(X, y, to_drop=[]):
    """
    Convert data from mem into a FairTest friendly format

    Parameters:
    -----------
    X :
        Input array

    y :
        Output array

    to_drop : opt
        A list with attributes (columns) to drop
        during loading the data
    """

    # TODO: Check input types - maybe imposible (or unnecessary) to convert
    X = np.array(X).astype(str)
    Y = np.array(y).astype(str)

    try:
        data = np.concatenate((X, np.array([Y]).T), axis=1)
    except ValueError:
        raise

    return data
