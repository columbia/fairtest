#!/usr/bin/env python3
"""
mkstats.py
Copyright (C) 2014, V. Atlidakis vatlidak@cs.columbia.edu>

Parse a bugreport log-file to create csv stats

@argv[1]: Input file

output: A csv with stats
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

    i = 0
    print("#location_dep, attr_val1, attr_val2, ..., attr_valN")
    for line in f:
        try:
            line = line.split('\n')[0]
            if line.split(',')[0].split(':')[0] == "total":
                continue
            attr_val = int(line.split(',')[0].split(':')[1])
            delta = float(line.split(',')[1].split(':')[1])
            location_dependency = int(line.split(',')[6].split(':')[1])
            if location_dependency not in stats:
                stats[location_dependency] = {}
            if attr_val not in stats[location_dependency]:
                stats[location_dependency][attr_val] = []
            stats[location_dependency][attr_val].append(delta)

        except Exception as error:
            print("Exception:%s at line:%s" % (error, line), file=sys.stderr)

    for location_dependency in sorted(stats):
        print("%d" % location_dependency, end="")
        for s in sorted(stats[location_dependency]):
            print(",%.2f" % (sum(stats[location_dependency][s]) / len(stats[location_dependency][s])), end="")
        print("")
    f.close()


def usage(argv):
    print ("Usage:%s ifile" % argv[0])
    sys.exit(-1)


if __name__ == '__main__':
    sys.exit(main())
