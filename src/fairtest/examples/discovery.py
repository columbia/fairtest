"""
Run FairTest Discovery Investigations on Movie Recommender Dataset

Usage: python discovery.py
"""

import fairtest.utils.prepare_data as prepare
from fairtest import Discovery, train, test, report, DataSource
import ast
import pandas as pd
from sklearn import preprocessing

import sys


def main(argv=sys.argv):
    if len(argv) != 1:
        usage(argv)

    FILENAME = "../../../data/recommender/recommendations.txt"
    OUTPUT_DIR = "."
    data = prepare.data_from_csv(FILENAME, sep='\\t',
                                 to_drop=['RMSE', 'Avg Movie Age',
                                          'Avg Recommended Rating',
                                          'Avg Seen Rating', 'Occupation'])
    TARGET = 'Types'
    SENS = ['Gender']

    EXPL = []
    labeled_data = [ast.literal_eval(s) for s in data[TARGET]]
    for labels in labeled_data:
        assert len(labels) == 5
    label_encoder = preprocessing.MultiLabelBinarizer()
    labeled_data = label_encoder.fit_transform(labeled_data)
    labels = label_encoder.classes_
    df_labels = pd.DataFrame(labeled_data, columns=labels)
    data = pd.concat([data.drop(TARGET, axis=1), df_labels], axis=1)
    TARGET = labels.tolist()

    data_source = DataSource(data)

    # Instantiate the experiment
    inv = Discovery(data_source, SENS, TARGET, EXPL, topk=10, random_state=0)

    # Train the classifier
    train([inv])

    # Evaluate on the testing set
    test([inv])

    # Create the report
    report([inv], "discovery", OUTPUT_DIR)


def usage(argv):
    print "Usage:%s" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
