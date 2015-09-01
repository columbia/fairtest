#!/usr/bin/env python3
"""
zipcode_match.py
Copyright (C) 2015 Xingzhou He <x.he@columbia.edu>

Match zip code coornidates with a list

@argv[1]: Input file
@argv[2]: zipcode.csv

output: A csv with coordinates
"""
import sys

zipcodes = {}

def main(argv=sys.argv):

    if len(argv) != 3:
        usage(argv)

    for l in open(argv[2], 'r').readlines()[1:]:
        l = l[:-1]
        item = l.split(',')
        zipcodes[item[0][1:-1]] = (item[3][1:-1], item[4][1:-1])

    for zips in open(argv[1], 'r').readlines():
        zips = zips[:-1]
        if zips in zipcodes:
            print(zips + ',' + zipcodes[zips][0] + ',' + zipcodes[zips][1])
        else:
            sys.stderr.write(zips + '\n')


def usage(argv):
    print ("Usage:%s ifile zipcode" % argv[0])
    sys.exit(-1)


if __name__ == '__main__':
    sys.exit(main())
