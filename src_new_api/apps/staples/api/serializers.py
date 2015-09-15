from rest_framework import serializers
from api.models import User

# Serializers define the API representation.
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('uid', 'zipcode', 'sex', 'race', 'income')
