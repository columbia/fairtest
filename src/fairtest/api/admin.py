from django.contrib import admin
from .models import User, Transaction

admin.site.register(User)
admin.site.register(Transaction)
