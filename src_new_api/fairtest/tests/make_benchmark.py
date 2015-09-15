#!/usr/bin/env python
from fairtest.bugreport import api2 as api
from fairtest.bugreport.helpers import prepare

import sys

def main(argv=sys.argv):

    if len(argv) != 2:
        usage(argv)

    # Preapre data into FairTest friendly format
    FILENAME = argv[1]
    data = prepare.data_from_csv(FILENAME, ['city'])

    # Initializing parameters for experiment
    EXPL = []
    SENS = ['race', 'income']
    TARGET = 'price'

    # Instanciate the experiment
    FT1 = api.Experiment(data, SENS, TARGET, EXPL,
                         measures={'race':'NMI', 'income': 'NMI'}, random_state=0)
    # Train the classifier
    FT1.train()

    # Evaluate on the testing set
    FT1.test()

    # Create the report
    FT1.report("benchmark")


def usage(argv):
    print "Usage:%s <filename>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
