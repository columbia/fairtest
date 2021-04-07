"""
Run FairTest Testing Investigation on Staples Dataset

Usage: python testing.py
"""

import fairtest.utils.prepare_data as prepare
from fairtest import Testing, train, test, report, DataSource

import sys


def main(argv=sys.argv):
    if len(argv) != 1:
        usage(argv)

    # Preapre data into FairTest friendly format
    FILENAME = "../../../data/staples/staples.csv"
    data = prepare.data_from_csv(FILENAME, to_drop=['zipcode', 'distance'])
    OUTPUT_DIR = "."

    # Initializing parameters for experiment
    EXPL = []
    SENS = ['income']
    TARGET = 'price'

    data_source = DataSource(data)

    # Instantiate the experiment
    inv = Testing(data_source, SENS, TARGET, EXPL, random_state=0)

    # Train the classifier
    train([inv])

    # Evaluate on the testing set
    test([inv])

    # Create the report
    report([inv], "testing", OUTPUT_DIR)


def usage(argv):
    print "Usage:%s" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
