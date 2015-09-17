#!/usr/bin/env python
from fairtest.bugreport import api2 as api
from fairtest.bugreport.helpers import prepare

import pandas as pd
import sklearn.preprocessing as preprocessing

from time import time

import sys
import ast

def main(argv=sys.argv):

    if len(argv) != 2:
        usage(argv)

    FILENAME = argv[1]

    data = prepare.data_from_csv(FILENAME, sep='\\t', to_drop=['userID', 'zip',
                                            'movieType', 'mode', 'median', 'improvement'])


    TARGET = 'Labels'
    SENS = ['gender']
    EXPL = []
    labeled_data = map(lambda s: ast.literal_eval(s), data[TARGET])
    for l in labeled_data:
        assert len(l) > 0

    label_encoder = preprocessing.MultiLabelBinarizer()
    labeled_data = label_encoder.fit_transform(labeled_data)
    labels = label_encoder.classes_
    df_labels = pd.DataFrame(labeled_data, columns=labels)
    data = pd.concat([data.drop(TARGET, axis=1), df_labels], axis=1)
    TARGET = labels.tolist()

    # Instanciate the experiment
    t1 = time()
    FT1 = api.Experiment(data, SENS, TARGET, EXPL,
                         measures={'gender': 'Reg'},
                         random_state=0)
    # Train the classifier
    t2 = time()
    FT1.train()

    # Evaluate on the testing set
    t3 = time()
    FT1.test(approx_stats=False)

    # Create the report
    t4 = time()
    FT1.report("recommender1")

    t5 = time()

    print "Regression:Recommender-Gender-Movies:Instantiation: %.2f, Train: %.2f, Test: %.2f, Report: %.2f"\
            % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))
    print "-" * 80
    print


#    data = prepare.data_from_csv(FILENAME, sep='\\t', to_drop=['userID', 'zip',
#                                            'movieType', 'mode', 'median', 'improvement'])
#
#    TARGET = 'Labels'
#    SENS = ['age']
#    EXPL = []
#    labeled_data = map(lambda s: ast.literal_eval(s), data[TARGET])
#    for l in labeled_data:
#        assert len(l) > 0
#
#    label_encoder = preprocessing.MultiLabelBinarizer()
#    labeled_data = label_encoder.fit_transform(labeled_data)
#    labels = label_encoder.classes_
#    df_labels = pd.DataFrame(labeled_data, columns=labels)
#    data = pd.concat([data.drop(TARGET, axis=1), df_labels], axis=1)
#    TARGET = labels.tolist()
# 
#
#    # Instanciate the experiment
#    t1 = time()
#    FT2 = api.Experiment(data, SENS, TARGET, EXPL,
#                         measures={'age': 'Reg'},
#                         random_state=0)
#    # Train the classifier
#    t2 = time()
#    FT2.train()
#
#    # Evaluate on the testing set
#    t3 = time()
#    FT2.test(approx_stats=False)
#
#    # Create the report
#    t4 = time()
#    FT2.report("recommender2")
#
#    t5 = time()
#
#    print "Regression:Recommender-Age-Movies:Instantiation: %.2f, Train: %.2f, Test: %.2f, Report: %.2f"\
#            % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))
#    print "-" * 80
#    print


def usage(argv):
    print "Usage:%s <filename>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
