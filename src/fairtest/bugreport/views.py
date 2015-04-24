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

zipcode_coordinates = {}
competitor_stores_coordinates = {}
staples_stores_coordinates = {}

def _get_price(user):
    """
    Helper function to simulate price shown to a user.
    The approach taken is the one followed in the Staples case
    which is summarized by the following rules:

    a) ZIP Codes whose center was farther than 20 miles from a Staples
       competitor saw higher prices 67% of the time. By contrast, ZIP
       Codes within 20 miles of a rival saw the high price least often,
       only 12% of the time.

    b) Staples.com showed higher prices most often—86% of the time—when
       the ZIP Code actually had a brick-and-mortar Staples store in it,
       but was also far from a competitor's store
    """

    def check_distance_from_store(stores_coordinates, loc, cutoff):
        for zipcode in stores_coordinates:
            dist = vincenty(loc,
                           (stores_coordinates[zipcode][0],
                           stores_coordinates[zipcode][1])).miles
            if dist <= cutoff:
                return True
        return False

    def randBinary(prob):
        # Prob in percentile integer
        if random.randint(0, 99) < prob:
            return price['low']
        return price['high']

    #cache zipcodes to coordinates mapping
    if not zipcode_coordinates:
        for z in Zipcode.objects.all():
            zipcode_coordinates[z.zipcode] = [z.latitude, z.longitude]
            print("zipcodes base")


    #cache competitor stores
    if not competitor_stores_coordinates:
        for c in Competitor.objects.all():
            competitor_stores_coordinates[c.zipcode] = [c.latitude, c.longitude]
            print("competitors")

    #cache staples stores
    if not staples_stores_coordinates:
        for s in Store.objects.all():
            staples_stores_coordinates[s.zipcode] = [s.latitude, s.longitude]
            print("staples")


    try:
        user_location = (zipcode_coordinates[user.zipcode][0],
                         zipcode_coordinates[user.zipcode][1])
    except Exception as error:
        print("Zipcode %s error: %s " % (user.zipcode, error))
        return price['low']

    if check_distance_from_store(competitor_stores_coordinates,
                                 user_location,
                                 20):
        return randBinary(12)
    elif check_distance_from_store(staples_stores_coordinates,
                                   user_location,
                                   20):
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

