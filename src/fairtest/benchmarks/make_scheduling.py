import fairtest.utils.prepare_data as prepare
from fairtest import Testing, ErrorProfiling, train, test, report, DataSource

from time import time
import sys
import numpy as np


def main(argv=sys.argv):
    if len(argv) != 3:
        usage(argv)

    # Prepare data into FairTest friendly format
    FILENAME = argv[1]
    data = prepare.data_from_csv(FILENAME)
    OUTPUT_DIR = argv[2]

    # Initializing parameters for experiment
    EXPL = []
    SENS = ['ReportsAbs']
    TARGET = 'PoliceUnitsPerReport'
    GROUND_TRUTH = 'Mean'

    to_drop = [
#        'Shift',
#        'Zipcode',
#        'Reports',
#        'ReportsAbs',
#        'Households',
        'MedianAgeMale',
        'MedianAgeFemale',
        'PoliceUnits',
        'HouseHolder65to74',
        'HouseHolder55to59',
        'HouseHolder35to44',
        'HouseHolder45to54',
        'HouseHolder25to34',
        'HouseHolder75to84',
        'HouseHolder85over',
        'HouseHolder15to24',
        'HouseHolder60to64',
        'NonFamilyHouseholds',
        'Households7PlusPerson',
        'Households2Person',
        'Households4Person',
        'Households6Person',
        'HouseholdsWith60Plus',
        'HouseholdsWith75Plus',
    ]

    data[SENS] = np.round(data[SENS])

    data_source = DataSource(data, train_size=0.25)

    # Instantiate the experiment
    t1 = time()
#    inv = Testing(data_source, SENS, TARGET, EXPL,
#                         random_state=0, to_drop=to_drop)
    inv = ErrorProfiling(data_source, SENS, TARGET, GROUND_TRUTH, EXPL,
                         random_state=0, to_drop=to_drop)


    # Train the classifier
    t2 = time()
    train([inv])

    # Evaluate on the testing set
    t3 = time()
    test([inv], exact=False)

    # Create the report
    t4 = time()
    report([inv], "scheduling", OUTPUT_DIR)

    t5 = time()
    print "Testing:Scheduling:Instantiation: %.2f, Train: %.2f, Test: %.2f, " \
          "Report: %.2f" % ((t2-t1), (t3-t2), (t4-t3), (t5-t4))
    print "-" * 80
    print


def usage(argv):
    print "Usage:%s <filename> <output_dir>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
