#!/usr/bin/env python
"""
get_users.py
Copyright (C) 2015, V. Atlidakis vatlidak@cs.columbia.edu>

Script to get json requests representing synthetic users
of the the users DB.

"""
import sys
import requests

def main(argv=sys.argv):

    if len(argv) != 1:
        usage(argv)

    response = requests.get('http://127.0.0.1:8000/users/',
                            auth=('root', 'os15')).json()
    for j in response:
        print j


def usage(argv):
    print ("Usage:%s") % argv[0]
    sys.exit(-1)


if __name__ == '__main__':
    sys.exit(main())
