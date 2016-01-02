"""
Effectiveness Benchmark for FairTest

Usage: ./make_benchmark_effectiveness.py fairtest/data/staples/staples.csv 100

The logic for now is:
    for SIZE in {100, 500, 1000, 2000, 5000}:
            - select 10 contexts {race, state} of size ~SIZE
            for DIFF in {10, 15, 20, 25, 30}:
                    - randomly select (50+DIFF)% of the rich
                      and (50-DIFF)% of he poor to get discounts
                    - run FairTest and report the number of contexts found
"""
import fairtest.utils.prepare_data as prepare
from fairtest import Testing, train, test, report, DataSource

from copy import deepcopy
from random import shuffle, randint, seed

import os
import sys
from math import ceil
from datetime import datetime

FIND_CONTEXTS_STRICT = False


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


def do_benchmark((classes, pool, guard_lines)):
    """
    main method doing the benchmark
    """
    results = {}
    BASE_FILENAME = "/tmp/temp_fairtest"

    MICROSECONDS = int((datetime.now() - datetime(1970, 1, 1)).
                       total_seconds()*10**6)
    # keep last digits of this very large number
    RANDOM_SEED = MICROSECONDS % 10**8
    seed(RANDOM_SEED)

    _classes = deepcopy(classes)
    # iterate for  various population sizes
    # sorted numericaly by population size.
    for _class in [str(x) for x in sorted([int(y) for y in _classes.keys()])]:

        selected = []
        shuffle(_classes[_class])

        for _ in range(0, 10):
            state_race = _classes[_class].pop()

            # keep the contexts selected to compare later
            # with Fairtest results
            selected.append(state_race)

        results[int(_class)] = {}
        # iterate for various effects
        for effect in [2.5, 5, 10, 15]:
            _pool = deepcopy(pool)
            _selected = deepcopy(selected)
            # TODO: Keep the same populations for a specific size
            # and iterate over different effects. In this way, the
            # graph will be readable in the y-axis since the comparison
            # will be on the same popultions -- as per Roxana's sugegstion

            random_suffix = str(randint(1, 999999))
            current_filename = BASE_FILENAME + random_suffix
            f_temp = open(current_filename, "w+")
            print >> f_temp, "state,gender,race,income,price"

            lines = 0
            for state_race in _selected:

                # create a pool with exactly the (50-effect)% discounts for poor
                # and (50+effect)% discounts for rich, so that this are a bit
                # more deterministic.
                poor_price_pool, rich_price_pool =\
                        make_price_pools(pool[state_race], effect)

                for entry in _pool[state_race]:
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
                del _pool[state_race]

            # print 'bias in populations {} of size {}'.format(_selected,_class)
            # This will be printing the remaining populations
            for state_race in _pool:
                # create exactly 50-50 split of discounts for the rest
                price_pool = [True]*(len(_pool[state_race])/2 + 1) +\
                             [False]*(len(_pool[state_race])/2 + 1)

                for entry in _pool[state_race]:
                    price = "low" if price_pool.pop() else "high"
                    print >> f_temp, "%s,%s,%s,%s,%s" % (entry.split(",")[0],
                                                         entry.split(",")[1],
                                                         entry.split(",")[2],
                                                         entry.split(",")[3],
                                                         price)
                    lines += 1

            f_temp.close()
            assert guard_lines == lines

            # Prepare data into FairTest friendly format
            data = prepare.data_from_csv(current_filename)
            data_source = DataSource(data)
            os.remove(current_filename)

            # Initializing parameters for experiment
            EXPL = []
            SENS = ['income']
            TARGET = 'price'

            # Instantiate the experiment
            inv = Testing(data_source, SENS, TARGET, EXPL,
                          random_state=RANDOM_SEED)

            # Train the classifier
            train([inv], min_leaf_size=50)

            exact_stats = False
            if int(_class) < 1000:
                exact_stats = True

            # Evaluate on the testing set
            test([inv], exact=exact_stats)

            # Create the report (apply no filtering)
            context_list = report([inv], "benchmark_" + random_suffix,
                                  output_dir="/tmp", node_filter='all',
                                  filter_conf=0)

            # count success
            found = 0
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

            results[int(_class)][effect] = found
            del _selected
        # end of iterations on effects
    # end of iterations on classes of sizes
    return results


def parse_results(results, iterations):
    """
    Helper parsing the results and converting into csv format
    """

    stats = {}

    for _class in results[0]:
        stats[_class] = {}
        for result in results:
            for effect in result[_class]:
                if effect not in stats[_class]:
                    stats[_class][effect] = 0
                stats[_class][effect] += result[_class][effect]

    print "#size",
    for effect in sorted(results[0][results[0].keys()[0]].keys()):
        print ",effect-%s" % effect,
    print

    for size in sorted(stats):
        print "%s" % size,
        for effect in sorted(stats[size]):
            print ",%.2f" % (float(stats[size][effect]) / float(iterations)),
        print


def main(argv=sys.argv):
    """
    Entry point
    """
    if len(argv) != 3:
        usage(argv)

    FILENAME = argv[1]
    ITERATIONS = int(argv[2])

    classes, pool, lines = load_file(FILENAME)

#    P = multiprocessing.Pool(multiprocessing.cpu_count() + 2)
#    results = P.map_async(do_benchmark, [(classes, pool, lines)]*ITERATIONS)
#    results = results.get()
#    P.close()
#    P.join()
    results = []
    for _ in range(0, ITERATIONS):
        results.append(do_benchmark((classes, pool, lines)))

    parse_results(results, ITERATIONS)


def usage(argv):
    print "Usage:%s <filename> <iterations>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
