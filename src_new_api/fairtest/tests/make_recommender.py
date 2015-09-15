#!/usr/bin/env python
from fairtest.bugreport import api2 as api
from fairtest.bugreport.helpers import prepare

import sys

def main(argv=sys.argv):

    if len(argv) != 2:
        usage(argv)

    # Preapre data into FairTest friendly format
    FILENAME = argv[1]
    data = prepare.data_from_csv(FILENAME, ['userID', 'city', 'zip',
                                            'movieType', 'mode', 'improvement'])

    # Initializing parameters for experiment
    EXPL = []
    SENS = ['age', 'gender']
    TARGET = 'median'

    # Instanciate the experiment
    FT1 = api.Experiment(data, SENS, TARGET, EXPL,
                         measures={'age':'Corr', 'gender': 'Corr'})
    # Train the classifier
    FT1.train()

    # Evaluate on the testing set
    FT1.test()

    # Create the report
    FT1.report("recommender1")



    data = prepare.data_from_csv(FILENAME, ['userID', 'city', 'zip',
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
    FT2.test()

    # Create the report
    FT2.report("recommender2")



    data = prepare.data_from_csv(FILENAME, ['userID', 'city', 'zip',
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
    FT3.test()

    # Create the report
    FT3.report("recommender3")



def usage(argv):
    print "Usage:%s <filename>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
