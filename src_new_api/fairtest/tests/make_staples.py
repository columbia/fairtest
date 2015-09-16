#!/usr/bin/env python
from fairtest.bugreport import api2 as api
from fairtest.bugreport.helpers import prepare

from time import time
import sys

def main(argv=sys.argv):

    if len(argv) != 2:
        usage(argv)

    # Preapre data into FairTest friendly format
    FILENAME = argv[1]
    data = prepare.data_from_csv(FILENAME, to_drop=['zipcode', 'distance'])

    # Initializing parameters for experiment
    EXPL = []
    SENS = ['income', 'race']
    TARGET = 'price'

    # Instanciate the experiment
    t1 = time()
    FT1 = api.Experiment(data, SENS, TARGET, EXPL)

    # Train the classifier
    t2 = time()
    FT1.train()

    # Evaluate on the testing set
    t3 = time()
    FT1.test()

    # Create the report
    t4 = time()
    FT1.report("staples1")

    t5 = time()
    print "Instantiation: %.2f, Train: %.2f, Test: %.2f, Report: %.2f"\
            % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))




    data = prepare.data_from_csv(FILENAME, to_drop=['zipcode', 'distance', 'city'])

    # Instanciate the experiment
    FT2 = api.Experiment(data, SENS, TARGET, EXPL)

    # Train the classifier
    FT2.train()

    # Evaluate on the testing set
    FT2.test()

    # Create the report
    FT2.report("staples2")



    data = prepare.data_from_csv(FILENAME, to_drop=['zipcode', 'distance', 'state'])

    # Instanciate the experiment
    FT3 = api.Experiment(data, SENS, TARGET, EXPL)

    # Train the classifier
    FT3.train()

    # Evaluate on the testing set
    FT3.test()

    # Create the report
    FT3.report("staples3")


def usage(argv):
    print "Usage:%s <filename>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
