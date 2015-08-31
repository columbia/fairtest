"""
populate_stores.py
Copyright (C) 2015, V. Atlidakis vatlidak@cs.columbia.edu>

Script to populate Staples stores from a csv file into the DB.
"""

import sys
from optparse import make_option

from django.core.management.base import BaseCommand
from django.db import transaction

from api.models import Store


class Command(BaseCommand):
    """A class used to register stores from a csv file"""

    help = "Command to import a set of stores"
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
            Store.objects.all().delete()
            print("Registering stores...")
            self._populate_stores(options['filename'])
        else:
            print("No filename given... :-(", file=sys.stderr)

    @transaction.commit_manually
    def _populate_stores(self, filename):
        "Open and parse a csv file, and register stores"
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
                zipcode = str(line.split(',')[0])
                latitude = float(line.split(',')[1])
                longitude = float(line.split(',')[2])
            except IndexError as error:
                print("Malformed line: <%s>" % line, file=sys.stderr)
                continue
            s = Store(zipcode, latitude, longitude)
            s.save()
            print("Store:%s saved"% (zipcode))
        transaction.commit()
        f.close()
