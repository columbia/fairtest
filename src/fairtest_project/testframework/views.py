from django.shortcuts import render
from django.http import HttpResponse
from django.template import Context
from django.template.loader import get_template

from testframework.models import Product
from testframework.models import Competitor
from testframework.models import Zipcode

def index(request):
    products = []
    for p in Product.objects.all():
        pz = {'name': p.name,  'price': str(float(p.adjusted_price(request)))}
        products.append(pz)

    t = get_template('product_view')
    return HttpResponse(t.render(Context({'products': products})))
