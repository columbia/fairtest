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
    data = prepare.data_from_csv(FILENAME, sep='\t',
    to_drop=['userID', 'RMSE', 'Labels'])

    # Initializing parameters for experiment
    dataname = 'movies.txt'
    SENS = ['age']
    TARGET = 'avgScore'
    EXPL = []

    # Instanciate the experiment
    t1 = time()
    FT1 = api.Experiment(data, SENS, TARGET, EXPL,
                         topk = 10,
                         random_state=0)
    # Train the classifier
    t2 = time()
    FT1.train()

    # Evaluate on the testing set
    t3 = time()
    FT1.test(approx_stats=False, prune_insignificant=True)

    # Create the report
    t4 = time()
    FT1.report("recommender_test")

    t5 = time()
    print "Testing:Recommender:Instantiation: %.2f, Train: %.2f, Test: %.2f, Report: %.2f"\
            % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))
    print "-" * 80
    print
    
    '''
        Error Profiling
    '''
    
    data = prepare.data_from_csv(FILENAME, sep='\t',
    to_drop=['userID', 'avgScore', 'Labels'])
    SENS = ['age']
    TARGET = 'RMSE'
    EXPL = []

    # Instanciate the experiment
    t1 = time()
    FT1 = api.Experiment(data, SENS, TARGET, EXPL,
                         topk = 10,
                         random_state=0)
    # Train the classifier
    t2 = time()
    FT1.train()

    # Evaluate on the testing set
    t3 = time()
    FT1.test(approx_stats=False, prune_insignificant=True)

    # Create the report
    t4 = time()
    FT1.report("recommender_error")

    t5 = time()
    print "Error:Recommender:Instantiation: %.2f, Train: %.2f, Test: %.2f, Report: %.2f"\
            % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))
    print "-" * 80
    print



def usage(argv):
    print "Usage:%s <filename>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
