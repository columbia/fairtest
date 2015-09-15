#!/usr/bin/env python
from fairtest.bugreport import api2 as api
from fairtest.bugreport.helpers import prepare
from fairtest.bugreport.clustering import display_clusters

import sys

def main(argv=sys.argv):

    if len(argv) != 2:
        usage(argv)

    # Preapre data into FairTest friendly format
    FILENAME = argv[1]
    data = prepare.data_from_csv(FILENAME)

    # Initializing parameters for experiment
    EXPL = []
    SENS = ['gender']
    TARGET = 'accepted'

    # Instanciate the experiment
    FT1 = api.Experiment(data, SENS, TARGET, EXPL)

    # Train the classifier
    FT1.train()

    # Evaluate on the testing set
    FT1.test()

    # Create the report
    FT1.report("berkeley", filter_by=display_clusters.FILTER_ALL)


def usage(argv):
    print "Usage:%s <filename>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
