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
    print("#dependency, min, avg, max, iter")
    for line in f:
        try:
            line = line.split('\n')[0]
            if line.split(',')[0].split(':')[0] != "total":
                continue
            discriminated = int(line.split(',')[3].split(':')[1])
            location_dependency = int(line.split(',')[4].split(':')[1])
            if location_dependency not in stats:
                stats[location_dependency] = []
            stats[location_dependency].append(discriminated)
        except Exception as error:
            print("Exception:%s at line:%s" % (error, line), file=sys.stderr)

    for s in sorted(stats):
        print("%d,%d,%.2f,%d,%d" % (s, min(stats[s]),
                                    sum(stats[s]) / len(stats[s]),
                                    max(stats[s]), len(stats[s])))
    f.close()


def usage(argv):
    print ("Usage:%s ifile" % argv[0])
    sys.exit(-1)


if __name__ == '__main__':
    sys.exit(main())
