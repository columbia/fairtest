"""
post_users.py
Copyright (C) 2015, V. Atlidakis vatlidak@cs.columbia.edu>

Script to POST users from a csv file into the REST-ful API.
"""

import sys
import requests
from optparse import make_option

from django.core.management.base import BaseCommand
from django.db import transaction
from api.models import User


class Command(BaseCommand):
    """A class used to POST users from a csv file into the API"""

    help = "Command to import a set of users"
    option_list = BaseCommand.option_list + (
                        make_option(
                                "-f",
                                "--file",
                                dest = "filename",
                                help = "specify import file", metavar = "FILE"
                        ),

                    )

    def handle(self, *args, **options):
        "Register handle for filename option"
        if options['filename']:
            # Clean old updates
            print("Cleaning old DB entries...")
            User.objects.all().delete()
            print("POST users...")
            self._post_users(options['filename'])
        else:
            print("No filename given... :-(", file=sys.stderr)

    def _post_users(self, filename):
        "Open and parse a csv file, and POST users"
        try:
            f = open(filename)
        except IOError as error:
            print("I/O error while opening file: <%s>" % filename,\
                    file=sys.stderr)
            return

        for line in f:
            try:
                line = line.split('\n')[0]
                if line[0] == '#':
                    continue
                uid = int(line.split(',')[0])
                zipcode = str(line.split(',')[1])
                sex = int(line.split(',')[2])
                race = int(line.split(',')[3])
            except IndexError as error:
                print("Malformed line: <%s>" % line, file=sys.stderr)
                continue
            response = requests.post('http://127.0.0.1:8000/users/',
                    data={'uid':uid, 'zipcode':zipcode, 'sex':sex, 'race':race},
                    auth=('root', 'os15'))
            if response.ok:
                print("POST success for user with id: %d" % uid)
            else:
                print("POST fail for user with id: %d. Got response: %s" %
                      (uid, response.text))
        f.close()
