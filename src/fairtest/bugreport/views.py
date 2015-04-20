from django.http import HttpResponse
from api.models import UserTransaction
import sys

def Bugreport(request):
    return HttpResponse("Hello, world from BugReport.")
