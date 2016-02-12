#!/usr/bin/python3
"""
Police department patrol scheduling

Scenario:

In this scenario we build a data-driven application to help Dallas police
departments efficiently distribute their police forces in order patrol Dallas.
To this end, we use past public Dallas police data to profile the relative risk
of various districts in Dallas during the three 8-hours police shifts.
In essence, based on the time of day (morning, noon, afternoon, or night),
differences in the relative risk of areas are being taken into consideration
and provide insights to better utilize a finite amount of police units.

The result is a plan describing the optimal allocation of police units into
districts in order to maximize the coverage of relatively dangerous areas. Note
that the optimal plan may be different between the three shifts, since,
intuitively, the relative risk of areas may be dependant on the time of day.

Usage: ./scheduling.py <report> <police_units>
"""
import sys
import json
from helpers import columns
from helpers import utils


def main(argv=sys.argv):
    '''
    Application entry-point
    '''
    if len(argv) != 3:
        usage(argv)

    filename = argv[1]
    total_units = int(argv[2])

    with open(filename, "r") as f:
        data = json.load(f)

    utils.print_stats(data, columns.COLUMNS)
    weights = utils.zipcode_weights_per_shift(data, columns.COLUMNS)
    schedule = make_schedule(weights, total_units)
    utils.print_schedule(schedule)


def make_schedule(weights, total_units):
    '''
    Core function constructing patrol schedule in a greedy fashion
    '''
    schedule = {1:{}, 2:{}, 3:{}}

    # dedicate units based on the percentage of reports
    for shift in weights:
        allocated = 0
        for i in range(len(weights[shift])):
            zipcode =  weights[shift][i][0]
            units = int(weights[shift][i][1] * total_units)
            if units < 1:
                continue
                units = 0
            schedule[shift][zipcode] = units
            allocated += units
        if not allocated:
            raise Exception("Too few units")
        while True:
            for i in range(len(weights[shift])):
                zipcode =  weights[shift][i][0]
                if allocated == total_units:
                    break
                if zipcode not in schedule[shift]:
                   continue
                schedule[shift][zipcode] += 1
                allocated += 1
            if allocated == total_units:
                break

    # remap final scheduling to zipcode
    for shift in weights:
        for i in range(len(weights[shift])):
            zipcode =  weights[shift][i][0]
            if zipcode not in schedule[shift]:
                break

    return schedule


def usage(argv):
    print('Usage:%s \"report\" \"police_units\"', argv[0])
    exit(-1)


if __name__ == '__main__':
    sys.exit(main())
