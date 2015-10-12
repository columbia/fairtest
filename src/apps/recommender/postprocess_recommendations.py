#!/usr/bin/python
"""
Join recommendations with user attributes and movie genres
"""
import sys
import statistics
import ast
import re
from collections import Counter
from os.path import isfile

occupations = {0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin", 4: "college/grad student", 5: "customer service", 6: "doctor/health care", 7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed", 17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer"}

def parse_movie(line):
    """
    Parses a movie in MovieLens format movieID::Title::types

    Parameters
    ----------
    line : str
        The line that contains user information

    Returns
    -------
    list : list
        A list containing Title, types
    """
    fields = line.strip().split("::")
    return [str(fields[1]).decode('utf-8', 'ignore'),str(fields[2]).split("|")]

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
        A list containing userID, gender, age, occupation
    """
    fields = line.strip().split("::")
    return [str(fields[0]), str(fields[1]),
            str(fields[2]), occupations[int(fields[3])]]

def parse_recommendation(line):
    """
    Parses a recommendation record. Format: userId\t avg_rating\t rmse\t labels

    Parameters
    ----------
    line : str
        The line that contains user information

    Returns
    -------
    list : list
        A list containing userID, avg_rating, rmse, labels
    """
    fields = line.strip().split("\t")[:]
    return [str(fields[0]), str(fields[1]),
            str(fields[2]), ast.literal_eval(fields[3])]

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
    users = {}
    for line in f_in:
        users[parse_user(line)[0]] = parse_user(line)[1:]
    f_in.close()

    return users

def load_recommendations(recommendations_file):
    """
    Load ratings from file.

    Parameters
    ----------
    scores_file : str
        The file that contains the rating info

    Returns
    -------
    recs : dict
        A recommendations dictionary in which userID is the key.
    """
    if not isfile(recommendations_file):
        print "File %s does not exist." % recommendations_file
        sys.exit(1)

    recs = {}
    f_in = open(recommendations_file, 'r')
    for line in f_in:
        recs[parse_recommendation(line)[0]] = parse_recommendation(line)[1:]
    f_in.close()

    return recs

def load_movies(movie_file):
    """
    Load ratings from file.

    Parameters
    ----------
    movie_fule : str
        The file that contain movie information entries

    Returns
    -------
    movies : dict
        A movie dictionary in which movieID is the key.
    """
    if not isfile(movie_file):
        print "File %s does not exist." % movie_file
        sys.exit(1)

    f_in = open(movie_file, 'r')
    movies = {}
    for line in f_in:
        movies[parse_movie(line)[0]] = parse_movie(line)[1]
    f_in.close()

    return movies

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Usage: %s user_file movies_file recommendations_file" % sys.argv[0]
        sys.exit(1)

    # load user file and recommendations titles
    USER_FILE = sys.argv[1]
    MOVIES_FILE = sys.argv[2]
    RECOMMENDATIONS_FILE = sys.argv[3]

    users = load_users(USER_FILE)
    movies = load_movies(MOVIES_FILE)
    recs = load_recommendations(RECOMMENDATIONS_FILE)
    
    f_out = open('output/recommendations.txt', 'w')
    
    print >> f_out, "Gender\tAge\tOccupation\tAvg Movie Rating\tRMSE\tAvg Movie Age\tTypes"
    
    for userId in recs:
        rec = recs[userId]
        user = users[userId]
        movieTypes = []
        tot_age = 0
        for movie in rec[2]:
            movieTypes += movies[movie]
            year = re.search(r"\((\d+)\)", movie).group(1)
            tot_age +=  2015-int(year)
        avg_age = (1.0*tot_age)/len(rec[2])
        counter = Counter(movieTypes)
        top5 = map(lambda x: x[0], counter.most_common(5))
        print >> f_out, "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(user[0], user[1], user[2], rec[0], rec[1], avg_age, top5)
