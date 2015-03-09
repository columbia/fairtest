import sys
from optparse import make_option

from correlation.dummy_correlation  import Correlation
from django.core.management.base import BaseCommand
from testframework.models import Competitor
from testframework.models import HasProduct
from testframework.models import Product

from decimal import Decimal
from random import randint


class Command(BaseCommand):
    """A class used to run user tests"""

    help = "Command to run custom user tests"

    def handle(self, *args, **options):
        "Register handle"
        dummy_test1()


def dummy_test1():
    print(Correlation.inputs)
    print(Correlation.outputs)

    for p in Product.objects.all():
        print(p)
