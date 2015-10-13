import sys
import random

from django.http import HttpResponse
from django.db.models import Count
from django.template.loader import get_template
from django.template import Context

from api.models import User, Zipcode, Store, Competitor

from bugreport.helpers.distance import haversine

#constants
epsilon = 0.05
price = {'low': 0, 'high': 1}
logfile = sys.stdout
logfile = open("./simulation/staples.csv", "a+")

#caches
zipcode_coordinates = {}
competitor_stores_coordinates = {}
staples_stores_coordinates = {}

def _get_price(user, location_dependency):
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

    #helper
    def check_distance_from_store(location, stores_coordinates, cutoff):
        for zipcode in stores_coordinates:
            dist = haversine(location[0],
                             location[1],
                             stores_coordinates[zipcode][0],
                             stores_coordinates[zipcode][1])
            if dist <= cutoff:
                return True
        return False

    # helper which returns 1 with probability "prob" in percentile integer
    def randBinary(prob):
        if random.randint(0, 99) < prob:
            return 1
        return 0

    # Use location dependency to decide what pricing approach to follow
    if not randBinary(location_dependency):
        return price['high']

    #cache zipcodes to coordinates mapping
    if not zipcode_coordinates:
        for z in Zipcode.objects.all():
            zipcode_coordinates[z.zipcode] = [z.latitude, z.longitude]

    #cache competitor stores
    if not competitor_stores_coordinates:
        for c in Competitor.objects.all():
            competitor_stores_coordinates[c.zipcode] = [c.latitude, c.longitude]

    #cache staples stores
    if not staples_stores_coordinates:
        for s in Store.objects.all():
            staples_stores_coordinates[s.zipcode] = [s.latitude, s.longitude]

    #if we don't know this zipcode, let's be random
    if user.zipcode not in zipcode_coordinates:
        if randBinary(50):
            return price['low']
        else:
            return price['high']

    user_location = (zipcode_coordinates[user.zipcode][0],
                         zipcode_coordinates[user.zipcode][1])

    if check_distance_from_store(user_location,
                                 competitor_stores_coordinates,
                                 20):
        print("%s,%s,%s,%s,%s,%s,%s,%s" % \
                ("near", user.zipcode, user.city, user.state,
                 user.SEX_CHOICES[user.sex][1],
                 user.RACE_CHOICES[user.race - 1][1],
                 user.INCOME_CHOICES[user.income - 1][0], "low"), file=logfile)
        return price['low']
    else:
        print("%s,%s,%s,%s,%s,%s,%s,%s" % \
                ("far", user.zipcode, user.city, user.state,
                 user.SEX_CHOICES[user.sex][1],
                 user.RACE_CHOICES[user.race - 1][1],
                 user.INCOME_CHOICES[user.income - 1][0], "high"), file=logfile)
        return price['high']

def BugreportView(request,
                  location_dependency="100",
                  protected_attr="race",
                  visits_per_user=1):
    """
    Evaluate statistical parity condition for all values of a protected
    attribute which we examine for discriminatory behavior.
    """

    location_dependency = int(location_dependency)
    #Query DB and create an in-memory structure (dictionary)
    prices_by_attr = {}
    for user in User.objects.all():
        for _ in range(0, visits_per_user):
            attr_val = user.get_attribute(protected_attr)
            if not attr_val:
                print("Attribute: <%s> was not found in models"
                      % protected_attr, file=sys.stderr)
                continue
            if attr_val not in prices_by_attr:
                prices_by_attr[attr_val] = {'high': 0, 'low': 0}
            if _get_price(user, location_dependency) == price['low']:
                prices_by_attr[attr_val]['low'] += 1
            else:
                prices_by_attr[attr_val]['high'] += 1

    total = sum([prices_by_attr[k]['high'] + prices_by_attr[k]['low'] for k\
                in prices_by_attr])
    total_low = sum([prices_by_attr[k]['low'] for k in prices_by_attr])
    total_high = sum([prices_by_attr[k]['high'] for k in prices_by_attr])

    content = []
    #Evaluate condition for each value of the protected attribute
    for attr_val in prices_by_attr:
        cur = prices_by_attr[attr_val]
        #Probability that a member of the current race receives high price
        p1 = cur['high'] / (cur['high'] + cur['low'])
        #Probability that a member of any race, but current, receives high price
        p2 = (total_high - cur['high']) /(total - cur['high'] - cur['low'])

        flag = "green"
        delta =  abs(p1 - p2)
        if delta > epsilon:
            flag = "red"
            content.append((attr_val, cur['low'], cur['high']))
        #print("attr_val:%s,delta:%.4f,total:%d,low:%d,high:%d,flag:%s,location_dependency:%d" %
        #      (attr_val, delta, cur['high'] + cur['low'], cur['low'], cur['high'], flag, location_dependency),
        #      file = logfile)

    #print("total:%d,low:%d,high:%d,discriminated:%d,location_dependency:%d" %
    #      (total, total_low, total_high, len(content), location_dependency), file = logfile)

    template = get_template('bugreport')
    return HttpResponse(template.render(Context({'content': content})))
