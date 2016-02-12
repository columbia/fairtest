#!/usr/bin/python3
"""
join.py
Copyright (C) 2016 V. Atlidakis  <vatlidak@cs.columbia.edu>

join two csv file entries on the first field (being zipcode)

@argv[1]: Input file
@argv[2]: Input file

output: A csv with the join
"""
import sys

def main(argv=sys.argv):

    if len(argv) != 3:
        usage(argv)
    filename1 = argv[1]
    filename2 = argv[2]
    header = None

    demographics = {}
    with open(filename1, "r") as f:
        for line in f:
            try:
                line = line.strip()
                zipcode = line.split(",")[0]
                rest = line.split(",")[1:]
            except Exception:
                continue
            demographics[zipcode] = rest

    with open(filename1, "r") as f:
        for line in f:
            header = line.strip()
            break

    shifts = {}
    print("zipcode,police_units,shift," + header[4:])
    with open(filename2, "r") as f:
        for line in f:
            try:
                line = line.strip()
                zipcode = line.split(",")[0]
                units = line.split(",")[1]
                shift = line.split(",")[2]
                print(
                    "%s,%s,%s,%s" % (
                        zipcode, units, shift, ".".join(demographics[zipcode])
                    )
                )
            except Exception as erro:
                print(erro)
                continue
            shifts[zipcode] = [units, shift]


def usage(argv):
    print ("Usage:<%s> <demographics> <shifts>" % argv[0])
    sys.exit(-1)


if __name__ == '__main__':
    sys.exit(main())



