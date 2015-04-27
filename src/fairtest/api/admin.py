from django.contrib import admin
from .models import User, Store, Competitor, Zipcode

admin.site.register(User)
admin.site.register(Store)
admin.site.register(Competitor)
admin.site.register(Zipcode)
