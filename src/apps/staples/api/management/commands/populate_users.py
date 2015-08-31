"""
populate_users.py
Copyright (C) 2015, V. Atlidakis vatlidak@cs.columbia.edu>

Script to populate users from a csv file into the DB,
shortcuting the REST-ful API.
"""

import sys
from optparse import make_option

from django.core.management.base import BaseCommand
from django.db import transaction

from api.models import User


class Command(BaseCommand):
    """A class used to register users from a csv file"""

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
            print("Registering users...")
            self._populate_users(options['filename'])
        else:
            print("No filename given... :-(", file=sys.stderr)

    def _normalized_income(self, income):
        if income < 5000:
            return 1
        elif 5000   <= income < 10000:
            return 2
        elif 10000  <= income < 20000:
            return 3
        elif 20000  <= income < 50000:
            return 4
        elif 50000  <= income < 80000:
            return 5
        elif 80000  <= income < 160000:
            return 6
        elif 160000 <= income < 320000:
            return 7
        elif 320000 <= income:
            return 8

    @transaction.commit_manually
    def _populate_users(self, filename):
        "Open and parse a csv file, and register users"
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
                city = str(line.split(',')[2])
                state = str(line.split(',')[3])
                sex = int(line.split(',')[4])
                race = int(line.split(',')[5])
                income = self._normalized_income(int(line.split(',')[6]))
            except IndexError as error:
                print("Malformed line: <%s>" % line, file=sys.stderr)
                continue
            u = User(uid, zipcode, city, state, sex, race, income)
            u.save()
            print("User:%d saved"% (uid))
        transaction.commit()
        f.close()
