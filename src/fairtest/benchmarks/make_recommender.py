"""
Run FairTest Investigations on Movie Recommender Dataset

Usage: ./make_recommender.py fairtest/data/recommender/recommendations.txt \
       results/recommender
"""

import fairtest.utils.prepare_data as prepare
from fairtest import Discovery, Testing, train, test, report, DataSource
import ast
import pandas as pd
import numpy as np
from sklearn import preprocessing

from time import time
import sys


def main(argv=sys.argv):
    if len(argv) != 3:
        usage(argv)

    '''
    1. Testing (average movie rating across age)
    '''
    # Prepare data into FairTest friendly format
    FILENAME = argv[1]
    data = prepare.data_from_csv(FILENAME, sep='\t')
    OUTPUT_DIR = argv[2]

    # prepare age
    data['Age'] = map(lambda a: 10 if a == 1
                           else 20 if a == 18
                           else 30 if a == 25
                           else 40 if a == 35
                           else 50 if a == 45 or a == 50
                           else 60 if a == 56 else None, data['Age'])

    data['Avg Seen Rating'] = ['low' if x < np.mean(data['Avg Seen Rating'])
                                   else 'high' for x in data['Avg Seen Rating']]

    data_source = DataSource(data)

    # Instantiate the experiments
    t1 = time()

    #
    # Test of associations on movie popularity
    #
    SENS = ['Gender', 'Age']
    TARGET = 'Avg Recommended Rating'
    EXPL = []

    test_ratings = Testing(data_source, SENS, TARGET, EXPL, random_state=0,
                           to_drop=['RMSE', 'Avg Movie Age',
                                    'Types', 'Avg Seen Rating'])

    #
    # Test of associations on movie popularity conditioned on error
    #
    SENS = ['Gender', 'Age']
    TARGET = 'Avg Recommended Rating'
    EXPL = ['Avg Seen Rating']

    test_ratings_expl = Testing(data_source, SENS, TARGET, EXPL, random_state=0,
                                to_drop=['RMSE', 'Avg Movie Age', 'Types'])

    inv = [test_ratings, test_ratings_expl]

    # Train the classifier
    t2 = time()
    train(inv)

    # Evaluate on the testing set
    t3 = time()
    test(inv)

    # Create the report
    t4 = time()
    report(inv, "recommender", OUTPUT_DIR)

    t5 = time()
    print "Testing:Recommender:Instantiation: %.2f, Train: %.2f, Test: %.2f, " \
          "Report: %.2f" % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))
    print "-" * 80
    print


def usage(argv):
    print "Usage:%s <filename> <output_dir>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
