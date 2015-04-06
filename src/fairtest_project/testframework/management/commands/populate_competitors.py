import sys

from django.core.management.base import BaseCommand
from testframework.models import Competitor
from optparse import make_option


class Command(BaseCommand):
    """A class used to register competitors read from a csv file"""

    help = "Command to import a set of competitors"
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
            Competitor.objects.all().delete()
            print("Registering comoetitors...")
            self._populate_competitors(options['filename'])
        else:
            print("No filename given... :-(", file=sys.stderr)

    def _populate_competitors(self, filename):
        "Open and parse a csv file, and register competitors"
        try:
            f = open(filename)
        except IOError as error:
            print("I/O error while opening file: <%s>" % filename,\
                    file=sys.stderr)
            return

        competitor_id = 0
        for line in f:
            try:
                line = line.split('\n')[0]
                competitor_name = line.split('|')[0]
                competitor_addr = line.split('|')[1]
                competitor_zip = line.split('|')[4]
            except IndexError as error:
                print("Malformed line: <%s>" % line, file=sys.stderr)
                continue
            c = Competitor(competitor_id, competitor_name,
                            competitor_addr, competitor_zip)
            c.save()
            competitor_id += 1

        f.close()
