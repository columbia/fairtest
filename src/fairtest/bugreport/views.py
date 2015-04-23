import sys
import random

from django.http import HttpResponse
from django.db.models import Count
from django.template.loader import get_template
from django.template import Context

from api.models import User, Zipcode, Store, Competitor

from geopy.distance import vincenty

#constants
price = {'low': 0, 'high': 1}
epsilon = 0.3

def _get_price(user):
    """
    Helper function to simulate price shown to a user.
    The appriach taken is the one followed in the Staples case
    which is summarized by the following rules:

    a) ZIP Codes whose center was farther than 20 miles from a Staples
       competitor saw higher prices 67% of the time. By contrast, ZIP
       Codes within 20 miles of a rival saw the high price least often,
       only 12% of the time.

    b) Staples.com showed higher prices most often—86% of the time—when
       the ZIP Code actually had a brick-and-mortar Staples store in it,
       but was also far from a competitor's store
    """

    def check_distance(models, loc, cutoff):
        for model in models:
            dist = vincenty(loc, (model.latitude, model.longitude)).miles
            if dist <= cutoff:
                return True
        return False

    def randBinary(prob):
        # Prob in percentile integer
        if random.randint(0, 99) < prob:
            return 0
        return 1

    user_zip = Zipcode.objects.get(zipcode=user.zipcode)
    loc = (user_zip.latitude, user_zip.longitude) # Use zipcode for location

    if check_distance(models.Competitor, loc, 20):
        return randBinary(12)
    elif check_distance(models.Store, loc, 20):
        return randBinary(86)
    else:
        return randBinary(67)


def BugreportView(request):
    """
    Evaluate statistical parity condition for all values of race
    which is the protected attribute on which we want to examine
    for discriminatory behavior
    """

    #Query DB and create an in-memory structure (dictionary)
    prices_by_race = {}
    for user in User.objects.all():
        if user.race not in prices_by_race:
            prices_by_race[user.race] = {'high': 0, 'low': 0}
        if _get_price(user) == price['low']:
            prices_by_race[user.race]['low'] += 1
        else:
            prices_by_race[user.race]['high'] += 1

    total = sum([prices_by_race[k]['high'] + prices_by_race[k]['low'] for k\
                in prices_by_race])
    total_low = sum([prices_by_race[k]['low'] for k in prices_by_race])
    total_high = sum([prices_by_race[k]['high'] for k in prices_by_race])

    content = []
    #Evaluate condition for each value of the protected attribute
    for race in prices_by_race:
        cur = prices_by_race[race]

        #Probability that a member of the current race receives high price
        p1 = cur['high'] / (cur['high'] + cur['low'])
        #Probability that a member of any race, but current, receives high price
        p2 = (total_high - cur['high']) /(total - cur['high'] - cur['low'])

        delta =  abs(p1 - p2)
        if delta > epsilon:
            content.append((race, cur['low'], cur['high']))
            print(delta)

    print(prices_by_race, total, total_low, total_high)
    template = get_template('bugreport')

    return HttpResponse(template.render(Context({'content': content})))

