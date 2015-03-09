from django.conf.urls import patterns
from django.contrib import admin
from django.http import HttpResponse
from correlation.dummy_correlation import Correlation


def correlation_view(request):
    return HttpResponse(repr(Correlation.inputs) + repr(Correlation.outputs));


def get_admin_urls(urls):
    def get_urls():
        my_urls = patterns('',
            (r'^correlation/$', admin.site.admin_view(correlation_view))
        )
        return my_urls + urls
    return get_urls

admin_urls = get_admin_urls(admin.site.get_urls())
admin.site.get_urls = admin_urls
