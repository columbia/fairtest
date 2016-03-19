from collections import Counter
from collections import OrderedDict



def zipcode_weights_per_shift(data, columns):
    '''
    This function calculates the percentage of reports that have occured
    in each area defined by a zipcode for eachof the three different shifts
    '''

    time_pos = 40
    year_pos = 37
    zipcode_pos = 24

    # convert absolute time to police shifts
    _records =  _apply_on_column(
        data,
        time_pos,
        _time_to_shift
    )
    # keep only zipcodes and shifts
    _records = list(
        map(
            lambda l:(l[zipcode_pos],l[time_pos], l[year_pos]), _records
        )
    )

    TESTING_YEAR = '2015'
    training_records = list(filter(lambda l: l[2] != TESTING_YEAR, _records))
    testing_records = list(filter(lambda l: l[2] == TESTING_YEAR, _records))

    # build zipcodes, report-times dictionary
    # Training set: all reports except those of 2015
    training_report_dict = {}
    for i,record in enumerate(training_records):
        _zipcode = record[0]
        _shift = record[1]
        _year = record[2]
        if _year == TESTING_YEAR:
            continue
        if _zipcode == None or _shift == None:
            continue
        if _zipcode not in training_report_dict:
            training_report_dict[_zipcode] = {1:0, 2:0, 3:0}
        training_report_dict[_zipcode][_shift] += 1

    # Testing set: only reports of 2015
    testing_report_dict = {}
    for i,record in enumerate(testing_records):
        _zipcode = record[0]
        _shift = record[1]
        _year = record[2]
        if _year != TESTING_YEAR:
            continue
        if _zipcode == None or _shift == None:
            continue
        if _zipcode not in testing_report_dict:
            testing_report_dict[_zipcode] = {1:0, 2:0, 3:0}
        testing_report_dict[_zipcode][_shift] += 1

    # build zipcodes, report probability per-shift dictionary
    training_report_dict_prob = {}
    for _zipcode in training_report_dict:
        if _zipcode == None:
            continue
        total = sum(training_report_dict[_zipcode].values())
        for _shift in training_report_dict[_zipcode]:
            if _shift == None:
                continue
            if _zipcode not in training_report_dict_prob:
                training_report_dict_prob[_zipcode] = {1:0, 2:0, 3:0}
            training_report_dict_prob[_zipcode][_shift] = \
                        training_report_dict[_zipcode][_shift] #/ total

    # sort zipcodes per shift in a decreasing order of risk
    shifts = {1:{}, 2:{}, 3:{}}
    for shift in shifts:
        for _zipcode in training_report_dict_prob:
            shifts[shift][_zipcode] = training_report_dict_prob[_zipcode][shift]

    # calculate final weights
    weights = {}
    for shift in shifts:
        total = sum(list(shifts[shift].values()))
        for k in shifts[shift]:
            shifts[shift][k] /= total
            # shifts[shift][k] *= 100
        weights[shift] = sorted(
            list(shifts[shift].items()),
            key=lambda l:l[1],
            reverse=True
        )

    return (
        weights,
        training_records,
        testing_records,
        training_report_dict,
        testing_report_dict
    )


def print_schedule(schedule):
    filename = "shifts" + ".txt"
    print("Creating file: <%s>" % filename)
    with open(filename, "w") as f:
        for shift in schedule:
            for zipcode in OrderedDict(
                sorted(schedule[shift].items(),
                key=lambda t: t[1],
                reverse=True)
            ):
                print(
                    "%s,%d,%s" % (zipcode, schedule[shift][zipcode], shift),
                    file=f
                )


def print_records(data, columns):
    '''
    Helper printing the records -- entries of police department
    '''
    for i, record in enumerate(data['data']):
        for column in sorted(columns.keys()):
            if columns[column] == 'time1':
                print(record[column], "|", end="")
                continue
            print(record[column], "|", end="")
        print("")


def print_stats(data, columns):
    '''
    Helper printing some stats to get a flavor of the size of the areas
    '''
    offincidents = []
    premises = []
    zipcodes = []
    cities = []
    states = []
    sectors = []
    districts = []
    times = []

    for record in data['data']:
        for column in sorted(columns.keys()):
            if columns[column] == 'offincident':
                offincidents.append(record[column])
            if columns[column] == 'premise':
                premises.append(record[column])
            if columns[column] == 'zipcode':
                zipcodes.append(record[column])
            if columns[column] == 'city':
                cities.append(record[column])
            if columns[column] == 'state':
                states.append(record[column])
            if columns[column] == 'sector':
                sectors.append(record[column])
            if columns[column] == 'district':
                districts.append(record[column])
            if columns[column] == 'time1':
                times.append(record[column])


def _apply_on_column(data, pos, func):
    '''
    Modify records pf data by applying function "func"
    on field at position "pos" and return the modified records.
    '''
    records = []
    for record in data['data']:
        _record = []
        for i, field in enumerate(record):
            if i == pos:
                _record.append(func(record[i]))
                continue
            _record.append(record[i])
        records.append(_record)
    return records


def _time_to_shift(time):
    '''
    Helper converting absolute time of day to police time shifts
    '''
    shift1 = ['08', '09', '10', '11', '12', '13', '14', '15']
    shift2 = ['00', '01', '02', '03', '04', '05', '06', '07']
    shift3 = ['16', '17', '18', '19', '20', '21', '22', '23']

    if not time:
        return None
    time = time[:2]

    if time in shift1:
        return 1
    elif time in shift2:
        return 2
    elif time in shift3:
        return 3
    else:
        None

