from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from api.models import UserTransaction
from api.serializers import UserTransactionsSerializer


class UserTransactionsListView(APIView):
    ''' List all transactions (GET), create new transactions (POST)'''

    queryset = UserTransaction.objects.all()

    def get(self, request, format=None):
        usertransactions = self.queryset
        serializer = UserTransactionsSerializer(usertransactions, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = UserTransactionsSerializer(data=request.DATA)
        if serializer.is_valid():
             serializer.save()
             return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserTransactionsUpdateView(APIView):
    '''Update transactions (PUT)'''

    queryset = UserTransaction.objects.all()

    def _get_object(self, tid):
        try:
            return UserTransaction.objects.get(tid=tid)
        except UserTransaction.DoesNotExist:
            raise Http404

    def put(self, request, tid, format=None):
        usertransaction = self._get_object(tid)
        serializer = UserTransactionsSerializer(usertransaction, data=request.DATA)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
