from rest_framework import status
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response

from api.models import User
from api.serializers import UserSerializer


class UsersListView(APIView):
    '''List all transactions (GET), create new transactions (POST)'''

    queryset = User.objects.all()

    def get(self, request, format=None):
        users = self.queryset
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = UserSerializer(data=request.DATA)
        if serializer.is_valid():
             serializer.save()
             return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserUpdateView(APIView):
    '''Update transactions (PUT)'''

    queryset = User.objects.all()

    def _get_object(self, uid):
        try:
            return User.objects.get(uid=uid)
        except User.DoesNotExist:
            raise Http404

    def put(self, request, uid, format=None):
        user = self._get_object(uid)
        serializer = UserSerializer(user, data=request.DATA)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
