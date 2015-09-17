#!/usr/bin/python
"""
join_users_scores.py
Copyright (C) 2015, V. Atlidakis vatlidak@cs.columbia.edu>

Joins users file with users' scores file and turns continious
MSE impovement score variable into categorical variable from 0 to 5.

@argv[1]: A file with user demographics

@argv[2]: A file with user scores

output: The join result

References
----------
http://ampcamp.berkeley.edu/big-data-mini-course/movie-recommendation-with-mllib
"""
import sys
import statistics
from os.path import isfile

def parse_zipcode(line):
    """
    Parses a zipcode zipcode,city,state,...

    Parameters
    ----------
    line : str
        The line that containing zipcode info

    Returns
    -------
    list : list
        A list containing zipcode, city, state
    """
    fields = line.strip().split(",")
    return [str(fields[0]), str(fields[1]),
            str(fields[2])]


def parse_user(line):
    """
    Parses a user in MovieLens format userID::gender::age::occupation::zip

    Parameters
    ----------
    line : str
        The line that contains user information

    Returns
    -------
    list : list
        A list containing userID, gender, age, occupation, zip
    """
    fields = line.strip().split("::")
    return [str(fields[0]), str(fields[1]),
            str(fields[2]), str(fields[3]), str(fields[4])]

def parse_score(line):
    """
    Parses a score record. Format: userId,movieType,mode,median,improvement

    Parameters
    ----------
    line : str
        The line that contains user information

    Returns
    -------
    list : list
        A list containing userID, movieType, mode, median, improvement
    """
    fields = line.strip().split(",")[:]
    return [str(fields[0]), str(fields[1]),
            str(fields[2]), str(fields[3]), str(fields[4]), str(''.join(fields[5:]).split(':')[:])]

def load_users(user_file):
    """
    Load ratings from file.

    Parameters
    ----------
    user_file : str
        The file that contain user information entries

    Returns
    -------
    users : dict
        A user dictionary in which userID is the key.
    """
    if not isfile(user_file):
        print "File %s does not exist." % user_file
        sys.exit(1)

    f_in = open(user_file, 'r')
    # drop header
    f_in.next()
    users = {}
    for line in f_in:
        users[parse_user(line)[0]] = parse_user(line)[1:]
    f_in.close()

    return users

def load_scores(scores_file):
    """
    Load ratings from file.

    Parameters
    ----------
    scores_file : str
        The file that contains the rating info

    Returns
    -------
    scores : dict
        A scores dictionary in which movieId is the key.
    """
    if not isfile(scores_file):
        print "File %s does not exist." % scores_file
        sys.exit(1)

    scores = {}
    f_in = open(scores_file, 'r')
    # drop header
    f_in.next()
    for line in f_in:
        scores[parse_score(line)[0]] = parse_score(line)[1:]
    f_in.close()

    return scores


def load_zipcode(zipcodes_file):
    """
    Load ratings from file.

    Parameters
    ----------
    scores_file : str
        The file that contains the rating info

    Returns
    -------
    scores : dict
        A scores dictionary in which movieId is the key.
    """
    if not isfile(zipcodes_file):
        print "File %s does not exist." % zipcodes_file
        sys.exit(1)

    zipcodes = {}
    f_in = open(zipcodes_file, 'r')
    # drop header
    f_in.next()
    for line in f_in:
        zipcodes[parse_zipcode(line)[0]] = parse_zipcode(line)[1:]
    f_in.close()

    return zipcodes


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Usage: %s user_file scores_file zipcodes" % sys.argv[0]
        sys.exit(1)

    # load ratings and movie titles
    USER_FILE = sys.argv[1]
    ZIPCODE_FILE = sys.argv[2]
    SCORE_FILE = sys.argv[3]

    users = load_users(USER_FILE)
    zipcodes = load_zipcode(ZIPCODE_FILE)
    scores = load_scores(SCORE_FILE)


    print "userID\tgender\tage\toccupation\tzip\tcity\tstate\tmovieType\tmode\tmedian\timprovement\tLabels"
    for userId in scores:
        zipcode =  users[userId][3]
        if zipcode not in zipcodes:
            city = "-"
            state = "-"
        else:
            city = zipcodes[zipcode][0]
            state = zipcodes[zipcode][1]

        print '\t'.join([userId] + users[userId][:] + [city, state] + \
                       scores[userId][:])
