"""
Run FairTest Testing Investigation on Staples Dataset

Usage: ./make_staples.py fairtest/data/staples/staples.csv results/staples
"""

import fairtest.utils.prepare_data as prepare
from fairtest import Testing, train, test, report, DataSource
from fairtest.utils import log
import logging

from time import time
import sys


def main(argv=sys.argv):
    if len(argv) != 3:
        usage(argv)

    log.set_params(level=logging.DEBUG)

    EXPL = ['less_than_median']
    # EXPL = ['juv_fel_count_bin']
    # EXPL = ['score_very_high']
    EXPL = ['']
    # SENS = ['age_cat', 'sex', 'race']
    SENS = ['race']
    TARGET = 'FP'
    to_drop = [
#        'FP',
        'FN',
#        'age',
        'priors_count.1',
        'priors_count',
        'decile_score.1',
        'id',
        'name',
        'first',
        'last',
        'start',
        'end',
        'event',
        'is_recid',
        'two_year_recid',
        'compas_screening_date',
        'dob',
        'days_b_screening_arrest',
        'c_jail_in',
        'c_jail_out',
        'c_case_number',
        'c_offense_date',
        'c_charge_desc',
        "r_case_number",
        "c_arrest_date",
        "c_days_from_compas",
        "c_charge_degree",
        "r_charge_degree",
        "r_days_from_arrest",
        "r_offense_date",
        "r_charge_desc",
        "r_jail_in",
        "r_jail_out",
        "violent_recid",
        "is_violent_recid",
        "vr_case_number",
        "vr_charge_degree",
        "vr_offense_date",
        "vr_charge_desc",
        "screening_date",
        "v_screening_date",
        "in_custody",
        "out_custody"
    ]

    # Preapre data into FairTest friendly format
    FILENAME = argv[1]
    data = prepare.data_from_csv(FILENAME, to_drop=to_drop)
    OUTPUT_DIR = argv[2]

    data_source = DataSource(data, train_size=0.5)

    # Initializing parameters for experiment
    inv = Testing(data_source, SENS, TARGET, EXPL,
                  metrics={'race':'NMI'}, random_state=10)
    train([inv], max_bins=5)
    test([inv])
    report([inv], "compas" + "_" + "_".join(SENS + [TARGET] + EXPL), OUTPUT_DIR)

    print


def usage(argv):
    print "Usage:%s <filename> <output_dir>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
