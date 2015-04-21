from rest_framework import serializers
from api.models import UserTransaction

# Serializers define the API representation.
class UserTransactionsSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserTransaction
        fields = ('tid', 'uid', 'zipcode', 'sex', 'race', 'price')
