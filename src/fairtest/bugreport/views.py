from django.http import HttpResponse
from api.models import UserTransaction
from django.db.models import Count
import sys

def Bugreport(request):
    prices_by_race = {}

    for tr in UserTransaction.objects.all():
        if tr.race not in prices_by_race:
            prices_by_race[tr.race] = {'high': 0, 'low': 0}
        if tr.price == 100.0:
            prices_by_race[tr.race]['low'] += 1
        else:
            prices_by_race[tr.race]['high'] += 1
    total = sum([prices_by_race[k]['high'] + prices_by_race[k]['low'] for k\
                in prices_by_race])
    total_low = sum([prices_by_race[k]['low'] for k in prices_by_race])
    total_high = sum([prices_by_race[k]['high'] for k in prices_by_race])

    for race in prices_by_race:
        cur = prices_by_race[race]

        #Probability that a member of the current race receives high price
        p1 = cur['high'] / (cur['high'] + cur['low'])
        #Probability that a member of any race, but current, receives high price
        p2 = (total_high - cur['high']) /(total - cur['high'] - cur['low'])

        delta =  abs(p1 - p2)
        print(delta)

    print(prices_by_race, total, total_low, total_high)
    return HttpResponse("Hello, world from BugReport.")

