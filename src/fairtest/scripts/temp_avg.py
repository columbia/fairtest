#!/usr/bin/env python3
"""
avg_income_per_race.py
Copyright (C) 2014, V. Atlidakis vatlidak@cs.columbia.edu>

Parse a file of synthetic users

@argv[1]: Input file

output: A csv with parsed users
"""
import sys
import re
import os
from stat import *


stats = {}


def main(argv=sys.argv):
    """
    Entry point
    """
    global stats

    if len(argv) !=2:
        usage(argv)

    ifile = argv[1]

    try:
        f = open(ifile)
    except IOError as error:
        print ("I/O error while opening file: %s" % error, file=sys.stderr)
        return
    incomes = {}
    i = 0
    print("#race, average income, total members")
    for line in f:
        try:
            line = line.split('\n')[0]
            income = int(line.split(',')[4])
            race = int(line.split(',')[3])
            if race not in incomes:
                incomes[race] = []
            incomes[race].append(income)

        except Exception as error:
            print("Exception:%s at line:%s" % (error, line), file=sys.stderr)

    f.close()
    for i in incomes:
        print("%d,%.0f,%d" %
              (i, sum(incomes[i]) / len(incomes[i]), len(incomes[i])))

def usage(argv):
    print ("Usage:%s ifile" % argv[0])
    sys.exit(-1)


if __name__ == '__main__':
    sys.exit(main())
