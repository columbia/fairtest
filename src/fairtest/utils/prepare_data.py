"""
Helper module to prepare data for a FairTest Experiment
"""
import pandas as pd


def data_from_csv(filename, sep=None, header=0, to_drop=None):
    """
    Load data from csv into a FairTest friendly format

    Parameters:
    -----------
    filename :
        abs path of the csv

    sep :
        data separator (regex or string)

    header :
        number of header lines to drop

    to_drop : opt
        A list with attributes (columns) to drop from the loaded data

    Returns
    -------
    data :
        the loaded dataset
    """
    try:
        if sep:
            _sep = sep
        else:
            _sep = r'\s*,\s*'

        data = pd.read_csv(filename, header=header, sep=_sep, engine='python',
                           na_values="?")

        if to_drop:
            for attribute in to_drop:
                data = data.drop(attribute, axis=1)

    except IOError:
        print "Error: Cannot open file \"%s\"" % filename
        raise
    except Exception, error:
        print "Error: %s loading data from file \"%s\"" % (error, filename)
        raise

    return data
