#!/usr/bin/env python3
"""
post_users.py
Copyright (C) 2015, V. Atlidakis vatlidak@cs.columbia.edu>

Script to post json requests representing synthetic users to
be inserted into the users DB.

@argv[1]: A csv file containing sunthetic users. Each line should
            be formatted as follows: <uid,sex,zipcode,race,price>
"""
import sys
import requests

def main(argv=sys.argv):

    if len(argv) != 2:
        usage(argv)

    ifile = argv[1]
    try:
        f = open(ifile)
    except IOError:
        print >> sys.stderr, "IOError while opening file: %s" % ifile
        return

    for line in f:
        try:
            line = line.split('\n')[0]
            uid = int(line.split(',')[0])
            sex = int(line.split(',')[1])
            zipcode = str(line.split(',')[2])
            race = int(line.split(',')[3])
            price = float(line.split(',')[4])
        except (IndexError,ValueError) as error:
            print("IndexError at line:", line, file=sys.stderr)
            continue

        response = requests.post('http://127.0.0.1:8000/usertransactions/',
                        data={'uid':uid, 'sex':sex, 'zipcode':zipcode,
                            'race': race, 'price':price},
                        auth=('root', 'os15'))
        if response.ok:
            print("POST success for user with id: %d", uid)
        else:
            print("POST fail for user with id: %d. Got response: %s",
                  uid, response.text)

    f.close()


def usage(argv):
    print ("Usage:%s users_file", argv[0])
    sys.exit(-1)


if __name__ == '__main__':
    sys.exit(main())
