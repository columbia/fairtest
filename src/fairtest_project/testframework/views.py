from django.shortcuts import render
from django.http import HttpResponse
from django.template import Context
from django.template.loader import get_template

from testframework.models import Product
from testframework.models import Competitor

def index(request):
    products = []
    for p in Product.objects.all():
        pz = {'name': p.name,  'price': str(p.baseprice)}
        print (pz)
        products.append(pz)
    t = get_template('product_view')
    return HttpResponse(t.render(Context({'products': products})))
