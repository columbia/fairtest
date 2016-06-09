"""
    Helper that maps coordinates with zipcodes and their demographics
    to produse an augmented "train.csv" file.

    @input: "train.csv"
    @input: "coord2zipcodes.csv"
    @input: "acs5demographics.csv"

    @output: "augmented_train.csv"
"""
import os
import csv


def map_zipcodes(input_file, zipcodes_file, output_file):
    # load zipcodes file
    with open(zipcodes_file, "r") as csvfile:
         coordinates = {}
         reader = csv.reader(csvfile, delimiter=',', quotechar='\"')
         for line in reader:
            try:
                addr = line[0]
                lat = line[1]
                lng = line[2]
                coordinates[(lat, lng)] = addr.split(",")[-2].split(" ")[2]
                #print "OK", addr
            except Exception, error:
                continue

    print "Mapping Phase:: Number of mapped coordinates:", len(coordinates.keys())
    # load input file and augment into output
    with open(input_file, "r") as inputcsv, open(output_file, "w") as outputcsv:
        row = {}
        headers=[
            'Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict',
            'Resolution', 'Address', 'X', 'Y', 'ZipCode'
        ]
        writer = csv.DictWriter(outputcsv, fieldnames=headers)
        writer.writeheader()
        reader = csv.reader(inputcsv, delimiter=',', quotechar='\"')
        reader.next()
        mapped = 0
        for line in reader:
            lat = line[8]
            lng = line[7]
            if (lat, lng) not in coordinates:
                continue
            mapped += 1
            zipcode = str(coordinates[(lat, lng)])
            row['ZipCode'] = zipcode
            row['Dates'] = line[0]
            row['Category'] = line[1]
            row['Descript'] = line[2]
            row['DayOfWeek'] = line[3]
            row['PdDistrict'] = line[4]
            row['Resolution'] = line[5]
            row['Address'] = line[6]
            row['Y'] = line[8]
            row['X'] = line[7]
            writer.writerow(row)

        print "Mapping Phase:: Mapped Incidents:", mapped


def merge(file1, file2, file3):
    demographics = {}
    with open(file1, "r") as f1:
        for line in f1:
            line = line.strip()
            demographics[line.split(",")[0]] = ",".join(line.split(",")[1:])

    with open(file2, "r") as f1, open(file3, "w") as f2:
        for line in f1:
            line = line.strip()
            zipcode = line.rsplit(",")[-1]
            print >> f2, "%s,%s" % (line, demographics[zipcode])

    # rm intermediate temp file
    os.remove(file2)


map_zipcodes("train.csv", "coord2zipcodes.csv", "temp.csv")
merge("acs5demographics.csv", "temp.csv", "augmented_train.csv")
