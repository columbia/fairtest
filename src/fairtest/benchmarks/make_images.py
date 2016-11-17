"""
Run FairTest Discovery Investigation on ImageNet Dataset

Usage: ./make_images.py fairtest/data/images/caffe.csv results/images
"""

import fairtest.utils.prepare_data as prepare
from fairtest import Discovery, train, test, report, DataSource

from time import time

import pandas as pd
import sklearn.preprocessing as preprocessing

import sys
import ast


def main(argv=sys.argv):
    if len(argv) != 3:
        usage(argv)

    # Preapre data into FairTest friendly format
    FILENAME = argv[1]
    data = prepare.data_from_csv(FILENAME, sep='\\t')
    OUTPUT_DIR = argv[2]


    TARGET = 'Labels'
    SENS = ['Race']
    EXPL = []

    labeled_data = [ast.literal_eval(s) for s in data[TARGET]]
    for l in labeled_data:
        assert len(l) == 5
    label_encoder = preprocessing.MultiLabelBinarizer()
    labeled_data = label_encoder.fit_transform(labeled_data)
    labels = label_encoder.classes_
    df_labels = pd.DataFrame(labeled_data, columns=labels)
    data = pd.concat([data.drop(TARGET, axis=1), df_labels], axis=1)
    TARGET = labels.tolist()

    data_source = DataSource(data)

    # Instantiate the experiment
    t1 = time()
    inv = Discovery(data_source, SENS, TARGET, EXPL, topk=35, random_state=0)

    # Train the classifier
    t2 = time()
    train([inv])

    # Evaluate on the testing set
    t3 = time()
    test([inv])

    # Create the report
    t4 = time()
    report([inv], "overfeat", OUTPUT_DIR)

    t5 = time()

    print "Discovery:Overfeat:Instantiation: %.2f, Train: %.2f, Test: %.2f, " \
          "Report: %.2f" % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))
    print "-" * 80
    print


def usage(argv):
    print "Usage:%s <filename> <output_dir>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
