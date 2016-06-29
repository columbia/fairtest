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

    to_drop = ['logloss', 'Entropy Abs', 'PdDistrict']

    EXPL = ['Entropy Bin']
    SENS = ['Category']
    TARGET = 'Error Str'

    data_source = DataSource(data, train_size=0.5)

    inv = Testing(data_source, SENS, TARGET, EXPL, random_state=0,
                  to_drop=to_drop)
    train([inv])
    test([inv], exact=False)
    if EXPL:
        report(
            [inv],
            "sf" + SENS[0].replace(' ', '') + TARGET.replace(' ','') + EXPL[0].replace(' ', ''),
            OUTPUT_DIR
        )
    else:
        report(
            [inv],
            "sf" + SENS[0].replace(' ', '') + TARGET.replace(' ',''),
            OUTPUT_DIR
        )




def usage(argv):
    print "Usage:%s <filename> <output_dir>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
