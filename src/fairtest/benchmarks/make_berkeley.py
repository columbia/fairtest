"""
Run FairTest Testing Investigation on Berkeley Dataset

Usage: ./make_berkeley.py fairtest/data/berkeley/berkeley.csv results/berkeley
"""

import fairtest.utils.prepare_data as prepare
from fairtest import Testing, train, test, report, DataSource

from time import time
import sys


def main(argv=sys.argv):
    if len(argv) != 3:
        usage(argv)

    # Preapre data into FairTest friendly format
    FILENAME = argv[1]
    data = prepare.data_from_csv(FILENAME)
    OUTPUT_DIR = argv[2]

    data_source = DataSource(data, budget=2)

    """
    First Experiment Without Explanatory Features
    """

    # Initializing parameters for experiment
    EXPL = []
    SENS = ['gender']
    TARGET = 'accepted'

    # Instantiate the experiment
    t1 = time()
    inv = Testing(data_source, SENS, TARGET, EXPL, random_state=0)

    # Train the classifier
    t2 = time()
    train([inv])

    # Evaluate on the testing set
    t3 = time()
    test([inv])

    # Create the report
    t4 = time()
    report([inv], "berkeley", OUTPUT_DIR)

    t5 = time()
    print "Testing:Berkeley:Instantiation: %.2f, Train: %.2f, Test: %.2f, " \
          "Report: %.2f" % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))
    print "-" * 80
    print

    """
    Second Experiment With Explanatory Feature
    """

    # Initializing parameters for experiment
    EXPL = ['department']
    SENS = ['gender']
    TARGET = 'accepted'

    # Instantiate the experiment
    t1 = time()
    inv = Testing(data_source, SENS, TARGET, EXPL, random_state=0)

    # Train the classifier
    t2 = time()
    train([inv])

    # Evaluate on the testing set
    t3 = time()
    test([inv])

    # Create the report
    t4 = time()
    report([inv], "berkeley_expl", OUTPUT_DIR)

    t5 = time()
    print "Testing:Berkeley_Expl:Instantiation: %.2f, Train: %.2f, " \
          "Test: %.2f, Report: %.2f" % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))
    print "-" * 80
    print


def usage(argv):
    print "Usage:%s <filename> <output_dir>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
