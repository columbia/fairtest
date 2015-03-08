from django.contrib import admin
from testframework.models import Product
from testframework.models import Competitor
from testframework.models import Zipcode

# Register your models here.
admin.site.register(Product)
admin.site.register(Competitor)
admin.site.register(Zipcode)
