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
                sex = int(line.split(',')[2])
                race = int(line.split(',')[3])
                income = int(line.split(',')[4])
            except IndexError as error:
                print("Malformed line: <%s>" % line, file=sys.stderr)
                continue
            u = User(uid, zipcode, sex, race, income)
            u.save()
            print("User:%d saved"% (uid))
        transaction.commit()
        f.close()
