#!/usr/bin/env python
"""
Performance Benchmark.

Usage: ./make_benchmark_performance.py fairtest/data/staples/staples.csv 100


Logic for now is:
    - Load staples file (excluding zipcode, city from BASE_FEATURES)
    - For ADDITIONAL_FEATURES in {5, 10, 15, ..., 50}
        - pick ADDITIONAL_FEATURES features from BASE_FEATURES (relpicates the same features)
        - shuffle file entries
        - For SIZE in {1000, 2000, 5000, 10000, 20000}
             - Pick the first SIZE entries
             - Apply fairtest for (race, output_
             - Report train_time and test_time
"""
from fairtest.bugreport import api2 as api
from fairtest.bugreport.helpers import prepare

from time import time
from copy import deepcopy
from random import shuffle, randint, seed

import os
import sys
import pandas as pd
import multiprocessing
from datetime import datetime


def parse_line(line):
    """
    line is in the format distance,zipcode,city,state,gender,race,income,price
    """
    line = line.strip().split(",")[1:]
    zipcode = str(line[0])
    city = str(line[1])
    state = str(line[2])
    gender = str(line[3])
    race = str(line[4])
    income = str(line[5])
    price = str(line[6])

    return [state, gender, race, income, price]


def load_file(file_name):
    """
    Helper loading file in mem. and mapping to dictionaries
    """
    f = open(file_name, "r")
    # drop header
    f.next()

    contents = []
    for line in f:
        contents.append(parse_line(line)[:])
    return contents


def get_avg_no_of_feat_values(contents):
    """
    Helper to calcultate numbers of differect values
    of categorical features, averaged for all featurs
    """
    total = 0
    for i in range(0, len(contents[0])):
        total += len(set(map(lambda x: x[i], contents)))

    return float(total) / float(len(contents[0]))


def magnify_contents(contents, features):
    """
    Create additional features in each entry by replicating some column
    of the original data. In order for the colums to differ from the
    original data append a suffix different for each new additional
    artificial column.
    """
    magnified_contents = []

    for entry in contents:
        magnified_entry = entry[:-1] +\
                          [entry[feature]+str(i) for i, feature in enumerate(features)] +\
                          [entry[-1]]
        magnified_contents.append(magnified_entry)

    return magnified_contents


def shuffle_column_contents(data, base_features):
    """
    After having create additional features by replicating some feature columns,
    shuffle the contencts of the additional columns in order to randomize the
    entropy of the data and cause more frustration to FairTest.
    """
    data_dict = dict(data)
    keys = list(data.columns.values)

    for key in keys:
        if key not in base_features:
            _values = dict(data_dict[key]).values()
            _keys = dict(data_dict[key]).keys()
            shuffle(_values)
            shuffled = dict(zip(_keys, _values))
            data_dict[key] = shuffled

    data = pd.DataFrame(data_dict)
    return data.reindex(columns=keys)


def do_benchmark((contents, n_features_max, inc, size_range)):
    """
    main method doing the benchmark
    """
    BASE_FILENAME = "/tmp/temp_fairtest"

    MICROSECONDS = int((datetime.now() - datetime(1970, 1, 1)).total_seconds()*10**6)
    RANDOM_SEED = MICROSECONDS % 10**8
    seed(RANDOM_SEED)

    BASE_FEATURES = ['state', 'gender', 'race', 'income', 'price']
    N_BASE = len(BASE_FEATURES)

    _contents = deepcopy(contents)

    # create more features without including the last two that will
    # be used as sensitiv and output. For each size, create the map
    # once for the maximum feature size.
    range_min = 0
    range_max = N_BASE - 3
    features = [randint(range_min, range_max) for _ \
                in range(0, n_features_max - N_BASE)]

    # create header
    features_header = \
            ','.join(BASE_FEATURES[:-1]) + ',' +\
            ','.join([BASE_FEATURES[feature] for feature in features]) +\
            ',price'

    # shuffle entries of the file loaded in memory
    # and copied within this function
    shuffle(_contents)
    _contents = _contents[:size_range[-1]]

    random_suffix = str(randint(1, 99999999))
    current_filename = BASE_FILENAME + random_suffix
    f_temp = open(current_filename, "w+")

    print >> f_temp, features_header
    for content in magnify_contents(_contents, features):
        print >> f_temp, ','.join(content)

    f_temp.close()

    # Prepare data into FairTest friendly format
    data = prepare.data_from_csv(current_filename)

    # shuffle around additional feature vales
    data = shuffle_column_contents(data, BASE_FEATURES)
    os.remove(current_filename)

    # Initializing parameters for experiment
    EXPL = []
    SENS = ['income']
    TARGET = 'price'

    # initialize the dictionary
    results = {}
    for n_features in range(N_BASE+1, n_features_max, inc):
        results[n_features] = {}
        for size in size_range:
            results[n_features][size] = {}

    for additional_features in range(1, n_features_max-N_BASE, inc):
        n_features = additional_features + N_BASE
        results[n_features] = {}

        for size in size_range:
            # Instanciate the experiment
            # print data.drop(data.columns[range(N_BASE-1, n_features_max-1-additional_features)], axis=1).head(size)
            _data = data.drop(data.columns[range(N_BASE-1, n_features_max-1-additional_features)], axis=1).head(size)
            FT1 = api.Experiment(_data, SENS, TARGET, EXPL,
                                 random_state=int(random_suffix))

            # Train the classifier
            t1 = time()
            FT1.train(min_leaf_size=50, score_aggregation="avg")
            t2 = time()

            # Evaluate on the testing set
            FT1.test(approx_stats=False, prune_insignificant=True)
            t3 = time()

            # Create the report
            random_suffix = str(randint(1, 99999999))
            FT1.report("benchmark_performance" + random_suffix, "/tmp", filter_by='all')

            train = t2 - t1
            test = t3 - t2

            avg_no_of_feat_values = get_avg_no_of_feat_values(_contents[:size])
            results[n_features][size] = [train, test, avg_no_of_feat_values]
            print n_features, size, results[n_features][size]
        # for all sizes
    # for all feature numbers
    return  results


def parse_results(results, iterations):
    """
    Helper parsing the results and converting into csv format
    """
    merged = {}

    print "#n_features,size,t_train,t_test,t_train_perc,t_test_perc,t_total,avg_feat_vals"

    for n_features in sorted(results[0]):
        merged[n_features] = {}
        for result in results:
            for size in result[n_features]:
                if size not in merged[n_features]:
                    merged[n_features][size] = [0, 0, 0]
                t_train = result[n_features][size][0]
                t_test = result[n_features][size][1]
                avg_no_of_feat_values = result[n_features][size][2]
                merged[n_features][size][0] += t_train
                merged[n_features][size][1] += t_test
                merged[n_features][size][2] += avg_no_of_feat_values

    for n_features in sorted(merged):
        for size in sorted(merged[n_features]):
            result =  merged[n_features][size]
            t_train = result[0]/iterations
            t_test = result[1]/iterations
            avg_no_of_feat_values = result[2]/iterations
            print "%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f" % \
                    (n_features, size, t_train, t_test,
                     100*t_train/(t_train + t_test),
                     100*t_test/(t_train + t_test),
                     t_train + t_test,
                     avg_no_of_feat_values)


def main(argv=sys.argv):
    """
    Entry point -- will try to parallelize
    """
    if len(argv) != 3:
        usage(argv)

    FILENAME = argv[1]
    ITERATIONS = int(argv[2])

    contents = load_file(FILENAME)

    INC = 5
    MAX_FEATURES = 70
    SIZE_RANGE = [10000, 20000, 30000, 40000]

    P = multiprocessing.Pool(multiprocessing.cpu_count()+2)
    results = P.map_async(do_benchmark,
                          [(contents, MAX_FEATURES, INC, SIZE_RANGE)]*ITERATIONS)
    results = results.get()
    P.close()
    P.join()

    parse_results(results, ITERATIONS)


def usage(argv):
    print "Usage:%s <filename> <iterations>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())