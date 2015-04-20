from django.conf.urls import patterns, include, url
from django.contrib import admin
from rest_framework import routers, serializers, viewsets
from api.models import UserTransaction
from bugreport import views

# Serializers define the API representation.
class UserTransactionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = UserTransaction
        fields = ('uid', 'zipcode', 'sex', 'race', 'price')


# ViewSets define the view behavior.
class UserTransactionViewSet(viewsets.ModelViewSet):
    queryset = UserTransaction.objects.all()
    serializer_class = UserTransactionSerializer


# Routers provide an easy way of automatically determining the URL conf.
router = routers.DefaultRouter()
router.register(r'usertransactions', UserTransactionViewSet)

urlpatterns = patterns('',
    url(r'^', include(router.urls)),
    url(r'^bugreport/', views.Bugreport),
    url(r'^admin/', include(admin.site.urls)),
    url(r'^api-auth/', include('rest_framework.urls',
                               namespace='rest_framework')),
)
