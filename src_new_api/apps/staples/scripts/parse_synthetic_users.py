#!/usr/bin/env python3
"""
parse_synthetic_users.py
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

    i = 0
    print("#uid, zipcode, sex, race, income")
    for line in f:
        try:
            line = next(f)
            line = line.split('\n')[0]
            income = int(line.split(': ')[1].split(',')[0])

            line = next(f)
            line = line.split('\n')[0]
            race = int(line.split(': ')[1].split(',')[0])

            line = next(f)
            line = line.split('\n')[0]
            sex = int(line.split(': ')[1].split(',')[0])

            line = next(f)
            line = line.split('\n')[0]
            zipcode = str(line.split(': ')[1].split(',')[0].replace('"',''))

            next(f)

            if not zipcode:
                print("Malformed line:%s" % line, file=sys.stderr)
                continue
            else:
                print("%d,%s,%d,%d,%d" % (i, zipcode, sex, race, income))
                i += 1
        except Exception as error:
            print("Exception:%s at line:%s" % (error, line), file=sys.stderr)

    f.close()


def usage(argv):
    print ("Usage:%s ifile" % argv[0])
    sys.exit(-1)


if __name__ == '__main__':
    sys.exit(main())
