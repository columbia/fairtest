#!/usr/bin/python3
"""
Police department patrol scheduling application

A data-driven application to help Dallas police departments efficiently
distribute their police forces in order patrol Dallas. To this end, I used past
public Dallas police data to profile the relative risk of various districts in
Dallas during the three 8-hours police shifts. Then, based on the time of day
(morning, noon, afternoon, or night), differences in the relative risk of areas
are being taken into consideration and provide insights to better utilize a
finite amount of police units.

The result is a plan describing the optimal allocation of police units into
districts in order to maximize the coverage of relatively dangerous areas and
keep a balanced number of police units available to handle each report. That is,
if in area A 10 incidents are reported, while in area B 100 incidents are
reported, in order for the police department to be fair, 10x more police units
must be allocated to resolve the reports of area B versus area A.

In essence, the application is a predictor that uses past data to predict the
current relative risk amongst areas. This feels close to what the police
departments might be doing -- i.e, empirically estimating the risk of areas
based on historical evidences, and then allocating their forces. For my
scenario, I used all reports prior to 2015 to profile the areas (training set),
and then I used all 2015 reports (testing set) to evaluate the efficiency of
allocating police using my application. The end-goal was to have a uniform
distribution of the value "police-units-available-per-report" across all
different Dallas areas.

The result of using this application has the following values for police units
available per report: Mean: 0.24, Mode:0.25, Stdev:0.06, Var:0.004.
Then, it comes FairTest to help me investigate further and better understand
whether there is anything questionable with my seemingly OK application. First,
I run an investigation to check if there is any association between the output
of my application (police units available to resolve each report) and the
median age of the population of an area.

At first sight, the report says that overall -- on the global population --
there is very little correlation between median age and the output of the
application. This is in-line with my initial naive approach to check the quality
of my application. However, a more fine grained analysis, there are some
contexts of subpopulation that have very negative correlation between median age
and the output of my application. (see attached picture and bug report.) These
populations are residents of areas that were under-represented in the "training"
set of reports used to profile the relative risk of areas. There were very few
incidents reported in the past, and these minorities received "unfavorable"
treatment from the application.

Usage: ./scheduling.py <reports> <demographics> <police_units>
"""
import sys
import json
from statistics import mean, mode, stdev
from helpers import columns
from helpers import utils


def main(argv=sys.argv):
    '''
    Application entry-point
    '''
    if len(argv) != 4:
        usage(argv)

    filename1 = argv[1]
    filename2 = argv[2]
    total_units = int(argv[3])

    with open(filename1, "r") as f:
        data = json.load(f)

    demographics = {}
    with open(filename2, "r") as f:
        for line in f:
            try:
                line = line.strip()
                zipcode = line.split(",")[0]
                rest = line.split(",")[1:]
            except Exception:
                continue
            demographics[zipcode] = rest

    with open(filename2, "r") as f:
        for line in f:
            header = line.strip()
            break


    utils.print_stats(data, columns.COLUMNS)
    weights, _records, report_dict = utils.zipcode_weights_per_shift(
        data, columns.COLUMNS
    )
    schedule = make_schedule(weights, total_units)

    filename = "./shifts" + ".txt"
    print("Creating file: <%s>" % filename)

    vals = []
    with open(filename, "w") as f:
        print(
            "Zipcode,PoliceUnitsPerReport,PoliceUnits,Shift," + header[4:],
            file=f
        )
        for record in _records:
            zipcode = record[0]
            shift = record[1]
            n_reports = 1
            try:
                population = float(demographics[zipcode][0])
                units = float(schedule[shift][zipcode])
                n_reports = float(max(1, int(report_dict[zipcode][shift])))
            except KeyError:
                # no zipcode in entry -- malformed record
                if not zipcode:
                    continue
                # no reports for this area
                else:
                    units = 0
                    population = 1

            cat = {1: "A", 2: "B", 3: "C"}
            print(
                "%s,%.5f,%d,%s,%s" % (
                    zipcode,
                    units / n_reports,
                    units,
                    cat[shift],
                    ",".join(demographics[zipcode][:])
                ),
                file=f
            )
            vals.append(units/n_reports)

    print(
        "Mean:%.2f, Mode:%.2f, Stdev:%.2f" % (
            mean(vals), mode(vals), stdev(vals)
        )
    )
    # utils.print_schedule(schedule)


def make_schedule(weights, total_units):
    '''
    Core function constructing patrol schedule in a greedy fashion
    '''
    schedule = {1:{}, 2:{}, 3:{}}

    # dedicate units based on the percentage of reports
    for shift in weights:
        allocated = 0
        for i in range(len(weights[shift])):
            zipcode =  weights[shift][i][0]
            units = int(weights[shift][i][1] * total_units)
            if units < 1:
                continue
                units = 0
            schedule[shift][zipcode] = units
            allocated += units
        if not allocated:
            raise Exception("Too few units")
        while True:
            for i in range(len(weights[shift])):
                zipcode =  weights[shift][i][0]
                if allocated == total_units:
                    break
                if zipcode not in schedule[shift]:
                   continue
                schedule[shift][zipcode] += 1
                allocated += 1
            if allocated == total_units:
                break

    # remap final scheduling to zipcode
    for shift in weights:
        for i in range(len(weights[shift])):
            zipcode =  weights[shift][i][0]
            if zipcode not in schedule[shift]:
                break

    return schedule


def usage(argv):
    print('Usage:%s \"report\" \"demographics\" \"police_units\"', argv[0])
    exit(-1)


if __name__ == '__main__':
    sys.exit(main())
