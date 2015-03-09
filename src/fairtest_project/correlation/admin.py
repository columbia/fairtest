from django.conf.urls import patterns
from django.contrib import admin
from django.http import HttpResponse
from correlation.dummy_correlation import Correlation

from django.template import Context
from django.template.loader import get_template

def correlation_view(request):
    t = get_template('correlation_admin');
    co = []
    for i in range(len(Correlation.inputs)):
        co.append((str(Correlation.inputs[i][0][1]), str(Correlation.outputs[i])));
    print(co)
    return HttpResponse(t.render(Context({'co' : co})))


def get_admin_urls(urls):
    def get_urls():
        my_urls = patterns('',
            (r'^correlation/$', admin.site.admin_view(correlation_view))
        )
        return my_urls + urls
    return get_urls

admin_urls = get_admin_urls(admin.site.get_urls())
admin.site.get_urls = admin_urls
