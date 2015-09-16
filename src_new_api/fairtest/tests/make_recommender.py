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
    data = prepare.data_from_csv(FILENAME, to_drop=['userID', 'city', 'zip',
                                            'movieType', 'mode', 'improvement'])

    # Initializing parameters for experiment
    EXPL = []
    SENS = ['age', 'gender']
    TARGET = 'median'

    # Instanciate the experiment
    t1 = time()
    FT1 = api.Experiment(data, SENS, TARGET, EXPL,
                         measures={'age':'Corr', 'gender': 'Corr'})
    # Train the classifier
    t2 = time()
    FT1.train()

    # Evaluate on the testing set
    t3 = time()
    FT1.test(approx_stats=False)

    # Create the report
    t4 = time()
    FT1.report("recommender1")

    t5 = time()
    print "Instantiation: %.2f, Train: %.2f, Test: %.2f, Report: %.2f"\
            % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))




    data = prepare.data_from_csv(FILENAME, to_drop=['userID', 'city', 'zip',
                                            'movieType', 'median',
                                            'improvement'])

    # Initializing parameters for experiment
    EXPL = []
    SENS = ['age', 'gender']
    TARGET = 'mode'

    # Instanciate the experiment
    FT2 = api.Experiment(data, SENS, TARGET, EXPL,
                         measures={'age':'Corr', 'gender': 'Corr'})

    # Train the classifier
    FT2.train()

    # Evaluate on the testing set
    FT2.test(approx_stats=False)

    # Create the report
    FT2.report("recommender2")



    data = prepare.data_from_csv(FILENAME, to_drop=['userID', 'city', 'zip',
                                            'movieType', 'median',
                                            'mode'])

    # Initializing parameters for experiment
    EXPL = []
    SENS = ['age', 'gender']
    TARGET = 'improvement'

    # Instanciate the experiment
    FT3 = api.Experiment(data, SENS, TARGET, EXPL,
                         measures={'age':'Corr', 'gender': 'Corr'})


    # Train the classifier
    FT3.train()

    # Evaluate on the testing set
    FT3.test(approx_stats=False)

    # Create the report
    FT3.report("recommender3")



def usage(argv):
    print "Usage:%s <filename>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
