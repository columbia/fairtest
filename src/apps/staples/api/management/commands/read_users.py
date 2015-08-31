"""
read_users.py
Copyright (C) 2015, V. Atlidakis vatlidak@cs.columbia.edu>

Script to read users from the DB, shortcuting the REST-ful API.
"""
import sys

from django.core.management.base import BaseCommand
from api.models import User
from optparse import make_option


class Command(BaseCommand):
    """A class used to read users from the DB"""

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
        self._read_users()

    def _read_users(self):
        "Read users from the DB"
        for u in  User.objects.all():
            print("%d,%s,%d,%d,%d"% (u.uid, u.zipcode, u.sex, u.race, u.income))
