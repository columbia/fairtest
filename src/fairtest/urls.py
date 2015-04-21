from django.contrib import admin
from django.conf.urls import patterns, include, url
from rest_framework.views import APIView

from api import views as api_views
from bugreport import views as bugreport_views


urlpatterns = patterns('',
    url(r'^users/', api_views.UsersListView.as_view()),
    url(r'^user/(?P<uid>[0-9]+)/$', api_views.UserUpdateView.as_view()),
    url(r'^bugreport/', bugreport_views.BugreportView),
    url(r'^admin/', include(admin.site.urls)),
    url(r'^api-auth/', include('rest_framework.urls',
                               namespace='rest_framework')),
)
