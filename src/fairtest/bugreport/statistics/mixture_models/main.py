import sys
import random
import numpy as np
import pandas as pd
from sklearn import mixture
import sklearn.tree as tree
import sklearn.metrics as metrics
import sklearn.linear_model as linear_model
import sklearn.preprocessing as preprocessing
import sklearn.cross_validation as cross_validation
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import chi2_contingency, chisqprob, fisher_exact

def main(argv=sys.argv):

    if len(argv) != 2:
        usage(argv)
    data = argv[1]
    BUSINESS = None
    '''
    original_data = pd.read_csv(
        data,
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num",
            "Martial Status", "Occupation", "Relationship", "Race",
            "Gender", "Capital Gain", "Capital Loss", "Hours per week",
            "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")

    sensitive_attr = "Gender"
    sensitive_attr_vals = {}
    sensitive_attr_vals['Gender'] = ["Female", "Male"]
    output  = "Target"
    output_vals = ["<=50K", ">50K"]
    '''

    '''
    original_data = pd.read_csv(
        data,
        names=[
            "Gender", "Department", "Admitted"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")

    sensitive_attr = "Gender"
    sensitive_attr_vals = {}
    sensitive_attr_vals['Gender'] = ["Female", "Male"]
    output  = "Admitted"
    output_vals = ["No", "Yes"]
    '''

    original_data = pd.read_csv(
        data,
        names=[
            "Distance", "Gender", "Race", "Income", "Price"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")

    output = "Price"

    sensitive_attr = 'Race'
    sensitive_attr_vals = {}
    sensitive_attr_vals['Race'] = ["Some Other Race",\
            "Native Hawaiian and Other Pacific Islander",\
            "Hispanic or Latino", "Black or African American",\
            "Asian", "American Indian and Alaska Native",\
            "White Not Hispanic or Latino",\
            "Two or More Races"]

    sensitive_attr = 'Income'
    sensitive_attr_vals['Income'] = ["10000<=income<20000",\
            "160000<=income<320000", "20000<=income<40000",\
            "320000<=income", "40000<=income<80000",\
            "5000<=income<10000", "80000<=income<160000",\
            "income<5000"]

    output_vals = ["low", "high"]

    '''
    original_data = pd.read_csv(
        data,
        names=[
            "Gender", "Purpose", "Credit"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")

    output = 'Credit'
    sensitive_attr = 'Gender'
    sensitive_attr_vals = {}
    sensitive_attr_vals['Gender'] = ["Female", "Male"]
    output_vals = ["Yes", "No"]
    '''

    BUSINESS = None
    original_data.tail()
    original_data = original_data.reindex(
            np.random.permutation(original_data.index)
            )
    binary_data = pd.get_dummies(original_data)
    binary_data_copy =  binary_data.copy()

    #remove sensitive attributes
    for attr in sensitive_attr_vals:
        for attr_val in sensitive_attr_vals[attr]:
           del binary_data_copy[attr + "_" + attr_val]

    binary_data_copy = 10 * binary_data_copy
    components = 2
    covariance_type = 'diag'
    dpgmm = mixture.DPGMM(n_components=components,covariance_type=covariance_type)
    dpgmm.fit(binary_data_copy)

    for i in range(0, components):
        freq_dict = {}
        for attr_val in sensitive_attr_vals[sensitive_attr]:
            freq_dict[attr_val] = {}
            for output_val in output_vals:
                freq_dict[attr_val][output_val] = 0

        current = dpgmm.predict(binary_data_copy)

        for (a, b) in zip(original_data[current == i][output],\
                          original_data[current == i][sensitive_attr]):
            freq_dict[b][a] += 1

        # convert dictionary into a table
        freq_table = [ [freq_dict[attr_val][val] for val in freq_dict[attr_val]]\
                        for attr_val in freq_dict]
        print "\n"
        print "Cluster:", i, "\nlength:", len(original_data[current == i])
        #print freq_table
        columns = list(binary_data_copy.columns.values)
        for j, mean in enumerate(dpgmm.means_[i]):
            if mean > 5:
                print columns[j],':', mean
        print "----"
        for attr_val in freq_dict:
            for output_val in freq_dict[attr_val]:
                print output_val, "\t",
            print "----",
            print ""
            break

        for attr_val in freq_dict:
            for output_val in freq_dict[attr_val]:
                print freq_dict[attr_val][output_val], "\t", 
            print attr_val,
            print ""

        try:
            (_, p_value, _, _) = chi2_contingency(freq_table,
                                                  lambda_="log-likelihood")
            print "p-value:", p_value
        except Exception, error:
            print error


def usage(argv):
    print ("Usage:<%s> <filename.data>") % argv[0]
    sys.exit(-1)


if __name__ == '__main__':
    sys.exit(main())
