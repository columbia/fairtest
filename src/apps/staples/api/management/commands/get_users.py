"""
get_users.py
Copyright (C) 2015, V. Atlidakis vatlidak@cs.columbia.edu>

Script to GET users from the REST-ful API.
"""
import sys
import requests

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    """A class used to GET users from the API"""

    help = "Command to GET a set of users"

    def handle(self, *args, **options):
        "Register handle"
        self._get_users()

    def _get_users(self):
        "GET users from the API"
        response = requests.get('http://127.0.0.1:8000/users/',
                            auth=('root', 'os15')).json()
        for j in response:
            print(j)
