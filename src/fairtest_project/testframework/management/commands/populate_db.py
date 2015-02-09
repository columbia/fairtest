import sys
from django.core.management.base import BaseCommand
from testframework.models import Product
from optparse import make_option

class Command(BaseCommand):
    help = "Command to import a set of products and respective prices"
    option_list = BaseCommand.option_list + (
                        make_option(
                                "-f",
                                "--file",
                                dest = "filename",
                                help = "specify import file", metavar = "FILE"
                        ),

                    )

    def register_products(self, filename):
        "Open and parse a csv file, and register products and prices"
        try:
            f = open(filename)
        except IOError, error:
            print >> sys.stderr, "I/O error while opening file: <%s>" % filename
            return
        for line in f:
            line = line.split('\n')[0]
            Product(name = line.split(',')[0], baseprice = line.split(',')[1]).save()
        f.close()

    def handle(self, *args, **options):
        "Register handle for filename option"
        Product.objects.all().delete()
        if options['filename']:
            self.register_products(options['filename'])

