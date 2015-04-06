import sys

from django.core.management.base import BaseCommand
from testframework.models import Product
from optparse import make_option


class Command(BaseCommand):
    """A class used to register produncts read from a csv file"""

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
            Product.objects.all().delete()
            print("Registering products...")
            self._populate_products(options['filename'])
        else:
            print("No filename given... :-(", file=sys.stderr)

    def _populate_products(self, filename):
        "Open and parse a csv file, and register products and prices"
        try:
            f = open(filename)
        except IOError as error:
            print("I/O error while opening file: <%s>" % filename,\
                    file=sys.stderr)
            return

        product_id = 0
        for line in f:
            try:
                line = line.split('\n')[0]
                product_name = line.split('|')[0]
                product_baseprice = float(line.split('|')[1])
            except IndexError as error:
                print("Malformed line: <%s>" % line, file=sys.stderr)
                continue
            p = Product(product_id, product_name, product_baseprice)
            p.save()
            product_id += 1

        f.close()
