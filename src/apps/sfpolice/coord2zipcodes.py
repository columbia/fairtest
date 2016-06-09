"""
    Helper that maps coordinates (i.e., longtitude and latitude) into zipcodes,
    using Google's GeoCofing APIs. The input file, "train.csv", contains the
    coordinates of 880K incidents. Due to rate limiting, the output
    file, "coord2zipcodes.py", contains 25K mapped coordinates that correspond
    to 430K incidents. In other words, 430K incidentes (a subset of the
    complete "train.csv") are augmented with zipcodes.

    NOTE: Google rate-limits to 2500 requests per key!

    @input: "train.csv"
    @output: "coord2zipcode.py"
"""
import csv
import time

import json, requests

KEY1 = "AIzaSyDzJZOqH30En6-ZQpLj21C50MK3o8peCXc"
KEY2 = "AIzaSyDPZVS3BFNlEv0Dpm62Q49W215wY-Hg2E4"
KEY3 = "AIzaSyByScHlFeUwcS-b_sCtA1RXn4iCoK_iVJg"
KEY4 = "AIzaSyC54N0rTMZxsENGmR7Vqii1hqQ35gokczs"
KEY5 = "AIzaSyC5YaO0j5qRNpdoBi99qLhhPZvWwYuEv18"
KEY6 = "AIzaSyBlq3YTQmG9P3FoZNpOTr5JAwY0ckV4ZUg"

with open("train.csv") as csvfile:
    addresses = {}
    reader = csv.reader(csvfile, delimiter=',', quotechar='\"')
    for line in reader:
        dates = line[0]
        category = line[1]
        descript = line[2]
        dayOfWeek = line[3]
        pdDistrict = line[4]
        resolution = line[5]
        address = line[6]
        x = line[7]
        y = line[8]
        addresses[address] = (x, y)

_dict = {}
with open("coord2zipcode.csv", "w") as output:
    for i, addr in enumerate(addresses):

        y = addresses[addr][0]
        x = addresses[addr][1]

        if i % 6 == 0:
            KEY = KEY1
        if i % 6 == 1:
            KEY = KEY2
        if i % 6 == 2:
            KEY = KEY3
        if i % 6 == 3:
            KEY = KEY4
        if i % 6 == 4:
            KEY = KEY5
        if i % 6 == 5:
            KEY = KEY6

        if i < 8000 and i > 10000:
            continue

        url = "https://maps.googleapis.com/maps/api/geocode/json?latlng={},{}&key={}".format(x, y, KEY)
        resp = requests.get(url=url, verify=False)
        resp = resp.json()
        try:
            result0 = resp['results'][0]
            formatted_address = result0['formatted_address']
    #             print >> output, "\"%s\",\"%s\"" % (addr, formatted_address)
            print >> output, "\"%s\",\"%s\",\"%s\",\"%s\"" % (addr, formatted_address, x, y)
        except Exception, error:
            print "error on address:", addr
        output.flush()
    #    time.sleep(0.02)
