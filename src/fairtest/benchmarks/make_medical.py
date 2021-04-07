"""
Run FairTest Error Profiling Investigation on Medical Dataset
Usage: ./make_medical.py fairtest/data/medical/predictions_reg_log.csv \
       results/medical
"""

import fairtest.utils.prepare_data as prepare
from fairtest import ErrorProfiling, train, test, report, DataSource

from time import time
import sys


def main(argv=sys.argv):
    if len(argv) != 3:
        usage(argv)

    # Prepare data into FairTest friendly format
    FILENAME = argv[1]
    data = prepare.data_from_csv(FILENAME)
    OUTPUT_DIR = argv[2]

    data_source = DataSource(data)

    # Initializing parameters for experiment
    TARGET = 'Prediction'
    GROUND_TRUTH = 'Ground_Truth'
    SENS = ['Age']
    EXPL = []

    # Instantiate the experiment
    t1 = time()
    inv = ErrorProfiling(data_source, SENS, TARGET, GROUND_TRUTH, EXPL,
                         random_state=0)

    # Train the classifier
    t2 = time()
    train([inv])

    # Evaluate on the testing set
    t3 = time()
    test([inv])

    # Create the report
    t4 = time()
    report([inv], "medical_reg", OUTPUT_DIR)

    t5 = time()

    print "Error:Health(Cont.):Instantiation: %.2f, Train: %.2f, Test: %.2f, " \
          "Report: %.2f" % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))
    print "-" * 80
    print


def usage(argv):
    print "Usage:%s <filename> <output_dir>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
