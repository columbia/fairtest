from django.shortcuts import render
from django.http import HttpResponse

from testframework.models import Product
from testframework.models import Competitor

def index(request):
    products = []
    for p in Product.objects.all():
        products.append(p.name + ':' + str(p.baseprice))

    output = ', '.join(products)

    return HttpResponse(output)
