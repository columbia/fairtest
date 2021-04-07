"""
    Helper that queries census ACS5 2010-2014 5-year report for demographics of
    zipcode areas: http://www.census.gov/data/developers/data-sets/acs-survey-5-year-data.html


    @output: "acs5demographics.csv"

    Features Explanation
    --------------------
    B01003_001E: Total Population
    B00002_001E: Unweighted Sample Housing Units
    B01002_001E: Median Age by Sex Median age --!!Total
    B02001_002E: White alone
    B02001_003E: Black or African American alone
    B02001_004E: American Indian and Alaska Native alone
    B02001_005E: Asian alone
    B02001_006E: Native Hawaiian and Other Pacific Islander alone
    B02001_007E: Some other race alone
    B02001_008E: Two or more races
    B08302_002E: 12:00 a.m. to 4:59 a.m
    B08302_015E: 4:00 p.m. to 11:59 p.m.
    B08302_014E: 12:00 p.m. to 3:59 p.m.
    B08302_009E: 8:00 a.m. to 8:29 a.m.
    B08302_010E: 8:30 a.m. to 8:59 a.m.
    B08302_011E: 9:00 a.m. to 9:59 a.m.
    B08302_012E: 10:00 a.m. to 10:59 a.m.
    B11001_002E: Family households:  B11001. Household Type (including Living Alone)
    B11001_008E: Nonfamily households:!!Householder living alone B11001. Household Type (including Living Alone)
    B19301_001E: Per capita income in the past 12 months (in 2009 inflation-adjusted dollars)
    B25001_001E: Total Housing Units
    B25021_001E: Median number of rooms --!!Total:   B25021. MEDIAN NUMBER OF ROOMS BY TENURE
    B25058_001E: Median contract rent    B25058. MEDIAN CONTRACT RENT (DOLLARS)
    B16010_002E: Less than high school graduate: B16010.
    B19013_001E: Median household income in the past 12 months (in 2009 inflation-adjusted dollars)
    B19025_001E: Aggregate household income in the past 12 months (in 2009 inflation-adjusted dollars)

    Full manual: http://api.census.gov/data/2014/acs5/variables.html
"""
import csv
from census import Census


KEY = "_"

QUERY = {
    'B01003_001E':'Total Population',
    'B00002_001E':'Housing Units',
    'B01002_001E':'Median Age',
    'B02001_002E':'White',
    'B02001_003E':'Black or African American',
    'B02001_004E':'AIAN',
    'B02001_005E':'Asian',
    'B02001_006E':'NHOPI',
    'B02001_007E':'Other',
    'B02001_008E':'Two or more races',
    'B08302_002E':'12:00am to 4:59am',
    'B08302_015E':'4:00pm to 11:59pm',
    'B08302_014E':'12:00pm to 3:59pm',
    'B08302_009E':'8:00am to 8:29am',
    'B08302_010E':'8:30am to 8:59am',
    'B08302_011E':'9:00am to 9:59am',
    'B08302_012E':'10:00am to 10:59am',
    'B11001_002E':'Family households',
    'B11001_008E':'Nonfamily households',
    'B19301_001E':'PPP income 12 months',
    'B25001_001E':'Total Housing Units',
    'B25021_001E':'Median number of rooms',
    'B25058_001E':'Median contract rent',
    'B16010_002E':'Less than high school graduate',
    'B19013_001E':'Median hh income 12 months',
    'B19025_001E':'Aggregate hh income 12 months'
}

ZIPCODES = [
    '94130', '94131', '94132', '94133', '94134', '94114', '94118', '94112',
    '94110', '94111', '94116', '94117', '94158', '94115', '94127', '94124',
    '94123', '94122', '94121', '94129', '94109', '94108', '94103',
    '94102', '94105', '94104', '94107', '94014'
]


c = Census(KEY)
with open("acs5demographics.csv", "w") as outputcsv:
    header = ['ZipCode'] + [str(QUERY[k]) for k in sorted(QUERY)]
    writer = csv.DictWriter(outputcsv, fieldnames=header)
    writer.writeheader()

    for zipcode in ZIPCODES:

        row = {}
        result = c.acs5.zipcode(QUERY.keys(), zipcode)
        try:
            result = result[0]
            row['ZipCode'] = str(zipcode)
            for key in ['ZipCode'] + sorted(QUERY):
                if key == 'ZipCode':
                    continue
                row[QUERY[key]] = str(result[key])
            writer.writerow(row)
        except Exception, error:
            print "Error:", error
            print "QUERY:", QUERY.keys()
            print "ZipCode:", zipcode
