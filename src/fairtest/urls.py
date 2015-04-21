from django.contrib import admin
from django.conf.urls import patterns, include, url
from rest_framework.views import APIView

from api import views as api_views
from bugreport import views as bugreport_views


urlpatterns = patterns('',
    url(r'^usertransactions/', api_views.UserTransactionsListView.as_view()),
    url(r'^usertransaction/(?P<tid>[0-9]+)/$', api_views.UserTransactionsUpdateView.as_view()),
    url(r'^bugreport/', bugreport_views.BugreportView),
    url(r'^admin/', include(admin.site.urls)),
    url(r'^api-auth/', include('rest_framework.urls',
                               namespace='rest_framework')),
)
