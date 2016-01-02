"""
Run FairTest Investigations on Movie Recommender Dataset

Usage: ./make_recommender.py fairtest/data/recommender/recommendations.txt \
       results/recommender
"""

import fairtest.utils.prepare_data as prepare
from fairtest import Discovery, Testing, train, test, report, DataSource
import ast
import pandas as pd
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

    label_col = 'Types'
    labeled_data = [ast.literal_eval(s) for s in data[label_col]]
    for labels in labeled_data:
        assert len(labels) == 5
    label_encoder = preprocessing.MultiLabelBinarizer()
    labeled_data = label_encoder.fit_transform(labeled_data)
    labels = label_encoder.classes_
    df_labels = pd.DataFrame(labeled_data, columns=labels)
    data = pd.concat([data.drop(label_col, axis=1), df_labels], axis=1)
    labels = labels.tolist()

    data_source = DataSource(data, budget=3, train_size=0.25)

    # Initializing parameters for experiment
    SENS = ['Gender']
    TARGET = 'Avg Movie Rating'
    EXPL = []

    # Instantiate the experiment
    t1 = time()
    inv = Testing(data_source, SENS, TARGET, EXPL, random_state=0,
                  to_drop=['RMSE', 'Avg Movie Age'] + labels)
    # Train the classifier
    t2 = time()
    train([inv])

    # Evaluate on the testing set
    t3 = time()
    test([inv])

    # Create the report
    t4 = time()
    report([inv], "recommender_test", OUTPUT_DIR+"/test")

    t5 = time()
    print "Testing:Recommender:Instantiation: %.2f, Train: %.2f, Test: %.2f, " \
          "Report: %.2f" % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))
    print "-" * 80
    print

    '''
    2. Error Profiling
    '''
    SENS = ['Gender']
    TARGET = 'RMSE'
    EXPL = []

    # Instantiate the experiment
    t1 = time()
    inv = Testing(data_source, SENS, TARGET, EXPL, random_state=0,
                  to_drop=['Avg Movie Age', 'Avg Movie Rating'] + labels)
    # Train the classifier
    t2 = time()
    train([inv])

    # Evaluate on the testing set
    t3 = time()
    test([inv])

    # Create the report
    t4 = time()
    report([inv], "recommender_error", OUTPUT_DIR+"/error")

    t5 = time()
    print "Error:Recommender:Instantiation: %.2f, Train: %.2f, Test: %.2f, " \
          "Report: %.2f" % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))
    print "-" * 80
    print

    '''
    3. Discovery
    '''
    SENS = ['Gender']
    TARGET = labels
    EXPL = []

    # Instantiate the experiment
    t1 = time()
    inv = Discovery(data_source, SENS, TARGET, EXPL, topk=10, random_state=0,
                    to_drop=['RMSE', 'Avg Movie Age',
                             'Avg Movie Rating', 'Occupation'])
    # Train the classifier
    t2 = time()
    train([inv])

    # Evaluate on the testing set
    t3 = time()
    test([inv])

    # Create the report
    t4 = time()
    report([inv], "recommender_discovery", OUTPUT_DIR+"/discovery")

    t5 = time()

    print "Discovery:Recommender:Instantiation: %.2f, Train: %.2f, " \
          "Test: %.2f, Report: %.2f" % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))
    print "-" * 80
    print


def usage(argv):
    print "Usage:%s <filename> <output_dir>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
