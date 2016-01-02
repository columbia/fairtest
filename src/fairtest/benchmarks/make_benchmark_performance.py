"""
Performance Benchmark.

Usage: ./make_benchmark_performance.py fairtest/data/staples/staples.csv 100


Logic for now is:
    - Load staples file (excluding zipcode, city from BASE_FEATURES)
    - For ADDITIONAL_FEATURES in {5, 10, 15, ..., 50}
        - pick ADDITIONAL_FEATURES features from BASE_FEATURES
          (replicates the same features)
        - shuffle file entries
        - For SIZE in {1000, 2000, 5000, 10000, 20000}
             - Pick the first SIZE entries
             - Apply FairTest for (race, output)
             - Report train_time and test_time
"""

import fairtest.utils.prepare_data as prepare
from fairtest import Testing, train, test, report, DataSource

from time import time
from copy import deepcopy
from random import shuffle, randint, seed

import os
import sys
import pandas as pd
from datetime import datetime


def parse_line(line):
    """
    line is in the format distance,zipcode,city,state,gender,race,income,price
    """
    line = line.strip().split(",")[1:]
    # zipcode = str(line[0])
    # city = str(line[1])
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
    Helper to calculate numbers of different values
    of categorical features, averaged for all features
    """
    total = 0
    for i in range(0, len(contents[0])):
        total += len(set([x[i] for x in contents]))

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
        magnified_entry = entry[:-1] + [entry[feature]+str(i) for i, feature
                                        in enumerate(features)] + [entry[-1]]
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


def do_benchmark((contents, feature_range, size_range)):
    """
    main method doing the benchmark
    """
    BASE_FILENAME = "/tmp/temp_fairtest"

    MICROSECONDS = int((datetime.now() - datetime(1970, 1, 1)).
                       total_seconds()*10**6)
    RANDOM_SEED = MICROSECONDS % 10**8
    seed(RANDOM_SEED)

    BASE_FEATURES = ['state', 'gender', 'race', 'income', 'price']
    N_BASE = len(BASE_FEATURES)

    _contents = deepcopy(contents)

    # create more features without including the last two that will
    # be used as sensitive and output. For each size, create the map
    # once for the maximum feature size.
    range_min = 0
    range_max = N_BASE - 3
    features = [randint(range_min, range_max) for _
                in range(0, feature_range[-1] - N_BASE)]

    # create header
    features_header = ','.join(BASE_FEATURES[:-1]) + ',' + \
                      ','.join([BASE_FEATURES[feature]
                                for feature in features]) + ',price'

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
    for n_features in feature_range:
        results[n_features] = {}
        for size in size_range:
            results[n_features][size] = {}

    additional_features_range = [x-N_BASE for x in feature_range]
    for additional_features in additional_features_range:
        n_features = additional_features + N_BASE
        results[n_features] = {}

        for size in size_range:
            # Instantiate the experiment
            _data = data.drop(
                data.columns[range(N_BASE-1,
                                   additional_features_range[-1] - 1 -
                                   additional_features)], axis=1).head(size)

            data_source = DataSource(_data)
            inv = Testing(data_source, SENS, TARGET, EXPL,
                          random_state=int(random_suffix))

            # Train the classifier
            t1 = time()
            train([inv])
            t2 = time()

            # Evaluate on the testing set
            test([inv])
            t3 = time()

            # Create the report
            _random_suffix = str(randint(1, 99999999))
            report([inv], "nop_benchmark_performance" + _random_suffix,
                   output_dir="/tmp")

            train_time = t2 - t1
            test_time = t3 - t2

            avg_no_of_feat_values = get_avg_no_of_feat_values(_contents[:size])
            results[n_features][size] = [train_time, test_time,
                                         avg_no_of_feat_values]
            del _data
            # print n_features, size, results[n_features][size]
        # for all sizes
    # for all feature numbers
    return results


def parse_results(results, iterations):
    """
    Helper parsing the results and converting into csv format
    """
    merged = {}

    print "#n_features,size,t_train,t_test,t_train_perc,t_test_perc,t_total," \
          "avg_feat_vals"

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
            result = merged[n_features][size]
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
    Entry point
    """
    if len(argv) != 3:
        usage(argv)

    FILENAME = argv[1]
    ITERATIONS = int(argv[2])

    contents = load_file(FILENAME)

    SIZE_RANGE = [10000, 20000, 40000, 60000]
    FEATURES_RANGE = [10, 20, 40, 60]

#    P = multiprocessing.Pool(multiprocessing.cpu_count()-2)
#    results = P.map_async(do_benchmark,
#                          [(contents, FEATURES_RANGE, SIZE_RANGE)]*ITERATIONS)
#    results = results.get()
#    P.close()
#    P.join()
#
    results = []
    for i in range(0, ITERATIONS):
        results.append(do_benchmark((contents, FEATURES_RANGE, SIZE_RANGE)))
    parse_results(results, ITERATIONS)


def usage(argv):
    print "Usage:%s <filename> <iterations>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
