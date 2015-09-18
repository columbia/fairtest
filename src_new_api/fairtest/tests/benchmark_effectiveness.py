#!/usr/bin/env python
"""
Run with: ./benchmark.py fairtest/data/staples/staples.csv

The logic for now is:
    for SIZE in {100, 500, 1000, 2000, 5000}: 
            for DIFF in {10, 15, 20, 25, 30}:
                    - select 10 contexts {race, state} of size ~SIZE
                    - randomly select (50+DIFF)% of the rich 
                      and (50-DIFF)% of he poor to get discounts
                    - run FairTest and report the number of contexts found
"""
from fairtest.bugreport import api2 as api
from fairtest.bugreport.helpers import prepare

from time import time
from copy import deepcopy
from random import shuffle, randint, seed
import sys


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
    DELTA = 0.20
    key = int(key)


    SIZE = 100
    if key in range(int(SIZE*(1 - DELTA)), int(SIZE*(1 + DELTA))):
        return str(SIZE)

    SIZE = 500
    if key in range(int(SIZE*(1 - DELTA)), int(SIZE*(1 + DELTA))):
        return str(SIZE)

    SIZE = 1000
    if key in range(int(SIZE*(1 - DELTA)), int(SIZE*(1 + DELTA))):
        return str(SIZE)

    SIZE = 2000
    if key in range(int(SIZE*(1 - DELTA)), int(SIZE*(1 + DELTA))):
        return str(SIZE)

    SIZE = 5000
    if key in range(int(SIZE*(1 - DELTA)), int(SIZE*(1 + DELTA))):
        return str(SIZE)

    return None


def main(argv=sys.argv):

    if len(argv) != 2:
        usage(argv)

    FILENAME = argv[1]
    f = open(FILENAME, "r")

    RANDOM_SEED = 0
    seed(RANDOM_SEED)

    # create a state_race dictionary with the sizes of each combination
    _sizes_dict = {}
    for line in f:
        state = parse_line(line)[0]
        gender = parse_line(line)[1]
        race = parse_line(line)[2]
        income = parse_line(line)[3]
        price = parse_line(line)[4]
        state_race = state + "_" + race

        if state_race not in _sizes_dict:
            _sizes_dict[state_race] = 0
        _sizes_dict[state_race] += 1

    # swap the dictionary so that sizes are the keys
    sizes_dict = dict (zip(_sizes_dict.values(), _sizes_dict.keys()))

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
    f = open(FILENAME, "r")
    f.next()

    lines1 = 0
    for line in f:
        state = parse_line(line)[0]
        gender = parse_line(line)[1]
        race = parse_line(line)[2]
        income = parse_line(line)[3]
        price = parse_line(line)[4]

        state_race = state + "_" + race
        if state_race not in pool:
            pool[state_race] = []
        pool[state_race].append(state + "," + gender + "," + race + "," + income + "," +  price)
        lines1 += 1

    FILENAME = "/tmp/temp_fairtest.txt"

    # iterate for various effects
    for effect in [5, 10, 15, 20]:

        _classes = deepcopy(classes)
        # iterate for  various population sizes
        # sorted numericaly by population size.
        for _class in map(str, sorted(map(int, _classes.keys()))):

            f_temp = open(FILENAME, "w+")
            print >> f_temp, "state,gender,race,income,price"

            selected = []
            _pool = deepcopy(pool)
            shuffle(_classes[_class])

            lines2 = 0
            for i in range(0, 10):
                state_race = _classes[_class].pop()

                # keep the contexts selected to compare later
                # with Fairtest results
                selected.append(state_race)

                for entry in _pool[state_race]:
                    state = entry.split(",")[0]
                    gender = entry.split(",")[1]
                    race = entry.split(",")[2]
                    income = entry.split(",")[3]
                    # TODO: randomize this also
                    if income == 'income >= 50K':
                        price = "low" if randint(1, 100) <= 50 + effect else "high"
                    else:
                        price = "low" if randint(1, 100) <= 50 - effect else "high"

                    print >> f_temp, "%s,%s,%s,%s,%s" % (state, gender, race, income, price)
                    lines2 += 1

                del _pool[state_race]

            # print 'bias in populations {} of size {}'.format(selected, _class)
            # This will be printing the remaining populations
            # TODO: Maybe we don't want to keep the whole remaining file,
            # but just a random part.
            for state_race in _pool:
                for entry in _pool[state_race]:
                    price = "low" if randint(1, 100) <= 50 else "high"
                    print >> f_temp, "%s,%s,%s,%s,%s" % (entry.split(",")[0],
                                              entry.split(",")[1],
                                              entry.split(",")[2],
                                              entry.split(",")[3],
                                              price)
                    lines2 += 1

            f_temp.close()

            # protect from random bugs that appear after 3a.m. ;-)
            assert lines1 == lines2

            # Preapre data into FairTest friendly format
            data = prepare.data_from_csv(FILENAME)

            # Initializing parameters for experiment
            EXPL = []
            SENS = ['income']
            TARGET = 'price'

            # Instanciate the experiment
            FT1 = api.Experiment(data, SENS, TARGET, EXPL,
                                 measures={'race': 'NMI'},
                                 random_state=RANDOM_SEED)
            # Train the classifier
            FT1.train(min_leaf_size=50)

            approx_stats = True
            if int(_class) < 1000:
                approx_stats = False

            # Evaluate on the testing set
            FT1.test(approx_stats=approx_stats, prune_insignificant=True)

            # Create the report
            # Create the report
            context_list = FT1.report("benchmark", "/tmp", filter_by='all')

            #print selected
            #print context_list

            # count sucess
            found = 0
            for context in context_list:
                # print context
                if len(context) == 2 and 'state' in context and 'race' in context:
                    state_race = str(context['state']) + "_" + str(context['race'])
                    if state_race in selected:
                        # remove it so that we don't count multiple
                        # times sub-sub-populations of a population
                        selected.remove(state_race)
                        found += 1

            print "effect:%s,size:%s,found:%s/10" % (str(effect), str(_class), str(found))


def usage(argv):
    print "Usage:%s <filename>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
