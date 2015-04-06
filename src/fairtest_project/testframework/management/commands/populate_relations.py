import sys

from django.core.management.base import BaseCommand
from testframework.models import HasProduct
from optparse import make_option


class Command(BaseCommand):
    """A class used to register competitor-product relation
        read from a csv file
    """

    help = "Command to import a set of products and respective prices"
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
            HasProduct.objects.all().delete()
            print("Registering relations...")
            self._populate_relations(options['filename'])
        else:
            print("No filename given... :-(", file=sys.stderr)

    def _populate_relations(self, filename):
        "Open and parse a csv file, and register relations"
        try:
            f = open(filename)
        except IOError as error:
            print("I/O error while opening file: <%s>" % filename,\
                    file=sys.stderr)
            return

        relation_id = 0
        for line in f:
            try:
                line = line.split('\n')[0]
                competitor_id = line.split('|')[0]
                product_id = line.split('|')[1]
            except IndexError as error:
                print("Malformed line: <%s>" % line, file=sys.stderr)
                continue
            r = HasProduct(relation_id, competitor_id, product_id)
            r.save()
            relation_id += 1

        f.close()
