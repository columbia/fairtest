"""
Run FairTest Testing Investigation on Staples Dataset

Usage: ./make_staples.py fairtest/data/staples/staples.csv results/staples
"""

import fairtest.utils.prepare_data as prepare
from fairtest.utils import log
from fairtest import Testing, train, test, report, DataSource

import matplotlib.pyplot as plt
from time import time
import logging
import os
import sys

from copy import deepcopy
from random import shuffle, randint, seed
from math import ceil
from datetime import datetime


EXPL = []
SENS = ['income']
TARGET = 'price'

EFFECT = 15
CLASS = '1000'
FIND_CONTEXTS_STRICT = False

MAX_BUDGET_REPS = 1#0
BUDGETS = [8000, 8500, 9000, 95000, 10000, 11000, 12000, 14000, 16000, 32000]
BUDGETS = [9000]



def parse_line(line):
    """
    line is in the format distance,zipcode,city,state,gender,race,income,price
    """
    line = line.strip().split(",")[3:]
    state = str(line[0])
    gender = str(line[1])
    race = str(line[2])
    income = str(line[3])
    price = str(line[4])

    return [state, gender, race, income, price]


def round(key):
    """
    Split the classes and allow a +-20% error on the sizes
    """
    DELTA = 0.2
    DELTA_HIGH = 0.25
    key = int(key)

    SIZE = 100
    if key in range(int(SIZE*(1 - DELTA)), int(SIZE*(1 + DELTA))+1):
        return str(SIZE)

    SIZE = 500
    if key in range(int(SIZE*(1 - DELTA)), int(SIZE*(1 + DELTA))+1):
        return str(SIZE)

    SIZE = 1000
    if key in range(int(SIZE*(1 - DELTA)), int(SIZE*(1 + DELTA))+1):
        return str(SIZE)

    SIZE = 2000
    if key in range(int(SIZE*(1 - DELTA_HIGH)), int(SIZE*(1 + DELTA_HIGH))+1):
        return str(SIZE)

    SIZE = 5000
    if key in range(int(SIZE*(1 - DELTA_HIGH)), int(SIZE*(1 + DELTA_HIGH))+1):
        return str(SIZE)

    SIZE = 10000
    if key in range(int(SIZE*(1 - DELTA_HIGH)), int(SIZE*(1 + DELTA_HIGH))+1):
        return str(SIZE)

    return None


def make_price_pools(population, effect):
    """
    Helper to create a pool with exactly (50-effect)% discount for
    people with low income and (50+effect)% discount for high income
    """
    income_map = [l.split(',')[3] for l in population]
    n_poor = len([l for l in income_map if l == 'income < 50K'])
    n_rich = len([l for l in income_map if l != 'income < 50K'])

    poor_price_pool = [True]*int(ceil(n_poor*(50.0-effect)/100)) +\
                      [False]*int(ceil(n_poor*(50.0+effect)/100))
    shuffle(poor_price_pool)

    rich_price_pool = [True]*int(ceil(n_rich*(50.0+effect)/100)) +\
                      [False]*int(ceil(n_rich*(50.0-effect)/100))
    shuffle(rich_price_pool)

    return poor_price_pool, rich_price_pool


def load_file(file_name):
    """
    Helper loading file in mem. and mapping to dictionaries
    """
    f = open(file_name, "r")

    # create a state_race dictionary with the sizes of each combination
    _sizes_dict = {}
    states = set()
    races = set()
    for line in f:
        state = parse_line(line)[0]
        gender = parse_line(line)[1]
        race = parse_line(line)[2]
        income = parse_line(line)[3]
        price = parse_line(line)[4]
        state_race = state + "_" + race

        states.add(state)
        races.add(race)

        if state_race not in _sizes_dict:
            _sizes_dict[state_race] = 0
        _sizes_dict[state_race] += 1

    # filter states with a single race
    for state in states:
        all_races = [state+"_"+race for race in races]
        all_sizes = [_sizes_dict.get(state_race, 0) for state_race in all_races]

        # check that the second most frequent race is large enough
        if len(all_sizes) < 2 or sorted(all_sizes, reverse=True)[1] < 300:
            for state_race in all_races:
                _sizes_dict.pop(state_race, None)

    # swap the dictionary so that sizes are the keys
    sizes_dict = dict(zip(_sizes_dict.values(), _sizes_dict.keys()))

    # round the sizes and split in five classes
    # {'100': 30, '5000': 19, '2000': 19, '500': 23, '1000': 24}
    classes = {}
    for key in sorted(sizes_dict):
        if round(key):
            if round(key) not in classes:
                classes[round(key)] = []
            classes[round(key)].append(sizes_dict[key])

    # reload the file and keep a dictionary with state_gender combinations
    # being the keys, and all the user attributes as the dictionary items
    # so that we don't read disk blocks in the main loop
    pool = {}
    f = open(file_name, "r")
    f.next()

    lines = 0
    for line in f:
        state = parse_line(line)[0]
        gender = parse_line(line)[1]
        race = parse_line(line)[2]
        income = parse_line(line)[3]
        price = parse_line(line)[4]

        state_race = state + "_" + race
        if state_race not in pool:
            pool[state_race] = []
        pool[state_race].append(state + "," + gender + "," + race + "," +
                                income + "," + price)
        lines += 1

    return classes, pool, lines


def do_benchmark(classes, pool, guard_lines,
                 expl, sens, target, output_dir, budgets, noise=False):
    """
    main method doing the benchmark
    """
    stats = []
    BASE_FILENAME = "/tmp/temp_fairtest"

    MICROSECONDS = int((datetime.now() - datetime(1970, 1, 1)).total_seconds()*10**6)

    # keep last digits of this very large number
    RANDOM_SEED = MICROSECONDS % 10**8
    seed(RANDOM_SEED)

    selected = classes[CLASS]

    random_suffix = str(randint(1, 999999))
    current_filename = BASE_FILENAME + random_suffix
    f_temp = open(current_filename, "w+")
    print >> f_temp, "state,gender,race,income,price"

    lines = 0
    for state_race in selected:

        # create a pool with exactly the (50-effect)% discounts for poor
        # and (50+effect)% discounts for rich, so that this are a bit
        # more deterministic.
        poor_price_pool, rich_price_pool =\
                make_price_pools(pool[state_race], EFFECT)

        for entry in pool[state_race]:
            state = entry.split(",")[0]
            gender = entry.split(",")[1]
            race = entry.split(",")[2]
            income = entry.split(",")[3]
            # TODO: randomize this also
            if income == 'income >= 50K':
                price = "low" if rich_price_pool.pop() else "high"
            else:
                price = "low" if poor_price_pool.pop() else "high"

            print >> f_temp, "%s,%s,%s,%s,%s" % (state, gender, race,
                                                 income, price)
            lines += 1
        del pool[state_race]

    # print 'bias in populations {} of size {}'.format(_selected,_class)
    # This will be printing the remaining populations
    for state_race in pool:
        # create exactly 50-50 split of discounts for the rest
        price_pool = [True]*(len(pool[state_race])/2 + 1) +\
                     [False]*(len(pool[state_race])/2 + 1)

        for entry in pool[state_race]:
            price = "low" if price_pool.pop() else "high"
            print >> f_temp, "%s,%s,%s,%s,%s" % (entry.split(",")[0],
                                                 entry.split(",")[1],
                                                 entry.split(",")[2],
                                                 entry.split(",")[3],
                                                 price)
            lines += 1

    f_temp.close()
    assert guard_lines == lines

    print "Finished file creation"

    # Prepare data into FairTest friendly format
    data = prepare.data_from_csv(current_filename)
    os.remove(current_filename)

    print "Finished data loading"
    print "Max number of contexs: %s" % (len(selected))

    k = 0
    if noise:
        k = len(selected)

    for budget in budgets:
    #for budget in range(minBudget, maxBudget + 1, step):

        found = 0.0
        data_source = DataSource(data, budget=budget, k=k)
        for nRep in xrange(1, budget + 1):
            inv = Testing(data_source, sens, target, expl, random_state=0)
            train([inv])


            exact_stats = False
            if int(CLASS) < 1000:
                exact_stats = True

            test([inv], exact=exact_stats)

            # Create the report (apply no filtering)
            context_list = report([inv], "benchmark_" + random_suffix,
                                  output_dir="/tmp", node_filter='all',
                                  filter_conf=0)
            # count success
            _selected = list(selected)
            for context in context_list:
                if ('state' in context and 'race' in context) and\
                        (len(context) == 2 or not FIND_CONTEXTS_STRICT):
                    state_race = str(context['state']) + "_" + \
                              str(context['race'])
                    if state_race in _selected:
                        # remove it so that we don't count multiple
                        # times sub-sub-populations of a population
                        _selected.remove(state_race)
                        found += 1
            del _selected

            print "Iteration-%d,%d,%d,%.2f" % (nRep, budget, found, found / nRep)

            # uncomment to cap the number of reps averaged
            if nRep >= MAX_BUDGET_REPS:
                 break

        stats.append((budget, found / nRep))
        print stats
        if found == 0.0:
            break

    return stats


def make_plot(data):

    plt.gca().set_color_cycle(['red', 'green'])

    max_x = []
    for d in data:
        max_x.append(max(map(lambda x: x[0], d)))

    max_y = []
    for d in data:
        max_y.append(max(map(lambda x: x[1], d)))


    plt.xlim((0, max(max_x) + 2))
    plt.ylim((0, max(max_y) + 2))

    plt.ylabel("Discovered Contexts")
    plt.xlabel("Budget (# of investigations)")

    for d in data:
        x = list(map(lambda x: x[0], d))
        y = list(map(lambda x: x[1], d))
        plt.plot(x, y, marker='o')

    plt.legend(['Baseline CDF', 'CDF with Laplacian Noise'], loc='upper left')
    plt.savefig("./plot.png")


def main(argv=sys.argv):

    if len(argv) != 2:
        usage(argv)

    FILENAME = argv[1]
    OUTPUT_DIR = "."

    log.set_params(level=logging.DEBUG)

#    # baseline CDF
#    classes, pool, lines = load_file(FILENAME)
#    cdf1 = do_benchmark(classes, pool, lines,
#                        EXPL, SENS, TARGET, OUTPUT_DIR, BUDGETS)

    # CDF with laplacian noise
    classes, pool, lines = load_file(FILENAME)
    cdf2 = do_benchmark(classes, pool, lines,
                        EXPL, SENS, TARGET, OUTPUT_DIR, BUDGETS, noise=True)

    print "-"*20
#    print cdf1
#    print cdf2
#
#    make_plot([cdf1, cdf2])


def usage(argv):
    print "Usage:%s <filename>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())




