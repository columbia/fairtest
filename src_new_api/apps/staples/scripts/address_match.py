#!/usr/bin/env python3
"""
address_match.py
Copyright (C) 2015 Xingzhou He <x.he@columbia.edu>

Match address csv with Google Geocoding API

@argv[1]: Input file

output: A csv with coordinates
"""
import geopy, sys, time

def main(argv=sys.argv):

    if len(argv) != 2:
        usage(argv)

    google = geopy.geocoders.GoogleV3()

    print("#store_numder,zipcode,latitude,longitude")
    for line in open(argv[1], 'r').readlines()[1:]:
        items = line[:-1].split(',')
        addr = items[0] + ',' + items[3] + ',' + items[4] + ',' + items[5]
        ret = google.geocode(addr, region="US")
        print(items[2]+ ',' + items[5] + ',' + str(ret.latitude) + ',' + str(ret.longitude))
        time.sleep(.2)


def usage(argv):
    print ("Usage:%s ifile" % argv[0])
    sys.exit(-1)


if __name__ == '__main__':
    sys.exit(main())
