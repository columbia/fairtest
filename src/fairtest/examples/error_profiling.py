"""
Run FairTest Error Profiling Investigations on Movie Recommender Dataset

Usage: python error_profiling.py
"""

import fairtest.utils.prepare_data as prepare
from fairtest import Testing, train, test, report, DataSource

import sys


def main(argv=sys.argv):
    if len(argv) != 1:
        usage(argv)

    FILENAME = "../../../data/recommender/recommendations.txt"
    OUTPUT_DIR = "."
    data = prepare.data_from_csv(FILENAME, sep='\t',
                                 to_drop=['Types', 'Avg Movie Age',
                                          'Avg Recommended Rating',
                                          'Avg Seen Rating'])
    SENS = ['Gender']
    TARGET = 'RMSE'
    EXPL = []

    data_source = DataSource(data)

    # Instantiate the experiment
    inv = Testing(data_source, SENS, TARGET, EXPL, random_state=0)
    # Train the classifier
    train([inv])

    # Evaluate on the testing set
    test([inv])

    # Create the report
    report([inv], "error_profiling", OUTPUT_DIR)


def usage(argv):
    print "Usage:%s" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
