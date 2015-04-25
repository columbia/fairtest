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
    print("#attr_val,total_visits,low_price,high_price,percent_low, percent_high")
    for line in f:
        try:
            line = line.split('\n')[0]
            if line.split(',')[0].split(':')[0] == "total":
                continue
            attr_val = str(line.split(',')[0].split(':')[1])
            #flag = line.split(',')[5].split(':')[1]
            low = int(line.split(',')[3].split(':')[1])
            high = int(line.split(',')[4].split(':')[1])
            if attr_val not in stats:
                stats[attr_val] = [0, 0]
            #if flag == "red":
            #    stats[attr_val][0] += 1
            #else:
            #    stats[attr_val][1] += 1
            stats[attr_val][0] += low
            stats[attr_val][1] += high

        except Exception as error:
            print("Exception:%s at line:%s" % (error, line), file=sys.stderr)

    for s in sorted(stats):
        perc_high = 100 * int(stats[s][1]) / ( int(stats[s][0]) + int(stats[s][1]))
        perc_low  = 100.0 - perc_high
        print("%s,%d,%d,%d,%.2f,%.2f" % (s, stats[s][0] + stats[s][1],
                                    stats[s][0], stats[s][1],
                                    perc_low, perc_high))
    f.close()


def usage(argv):
    print ("Usage:%s ifile" % argv[0])
    sys.exit(-1)


if __name__ == '__main__':
    sys.exit(main())
