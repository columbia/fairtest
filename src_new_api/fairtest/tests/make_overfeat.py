#!/usr/bin/env python
from fairtest.bugreport import api2 as api
from fairtest.bugreport.helpers import prepare
from time import time

import pandas as pd
import sklearn.preprocessing as preprocessing

import sys
import ast

def main(argv=sys.argv):

    if len(argv) != 2:
        usage(argv)

    # Preapre data into FairTest friendly format
    FILENAME = argv[1]
    data = prepare.data_from_csv(FILENAME, sep='\\t')

    TARGET = 'Labels'
    SENS = ['Race']
    EXPL = []
    labeled_data = map(lambda s: ast.literal_eval(s), data[TARGET])
    for l in labeled_data:
        assert len(l) == 5
    label_encoder = preprocessing.MultiLabelBinarizer()
    labeled_data = label_encoder.fit_transform(labeled_data)
    labels = label_encoder.classes_
    df_labels = pd.DataFrame(labeled_data, columns=labels)
    data = pd.concat([data.drop(TARGET, axis=1), df_labels], axis=1)
    TARGET = labels.tolist()
 

    # Instanciate the experiment
    t1 = time()
    FT1 = api.Experiment(data, SENS, TARGET, EXPL,
                         measures={'Race':'Reg'},
                         random_state=0)
    # Train the classifier
    t2 = time()
    FT1.train()

    # Evaluate on the testing set
    t3 = time()
    FT1.test(approx_stats=False)

    # Create the report
    t4 = time()
    FT1.report("overfeat")

    t5 = time()

    print "Regression:Overfeat-Race-Label:Instantiation: %.2f, Train: %.2f, Test: %.2f, Report: %.2f"\
            % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))
    print "-" * 80
    print


def usage(argv):
    print "Usage:%s <filename>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
