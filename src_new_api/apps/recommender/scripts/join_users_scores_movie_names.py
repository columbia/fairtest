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
    for line in f_in:
        scores[parse_score(line)[0]] = parse_score(line)[1:]
    f_in.close()

    return scores

def categorize(score, avg, stdev):
    """
    Turn a continous variable into categorical from 0 to 5

    Parameters
    ----------
    score : float
        Improvement score
    avg : float
        Average of improvements
    stdev : float
        Standard deviation of improvements

    Returns
    -------
    int : int
        Normalized improvement

    Notes
    -----
    We may get rid of this function and let improvement be a continious
    float variable.

    """
    if score < avg:
        return 0
    if score < avg - stdev and score >= avg - 2 * stdev:
        return 1
    if score < avg and  score >= avg - stdev:
        return 2
    if score < avg + stdev and score >= avg:
        return 3
    if score < avg + 2 * stdev and score >= avg + stdev:
        return 4
    if score >= avg + 2 * stdev:
        return 5


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: %s user_file scores_file" % sys.argv[0]
        sys.exit(1)

    # load ratings and movie titles
    USER_FILE = sys.argv[1]
    SCORE_FILE = sys.argv[2]

    users = load_users(USER_FILE)
    scores = load_scores(SCORE_FILE)

    avg = sum(map(lambda x: float(x[3]), scores.values())) / \
          len(scores.values())
    stdev = statistics.stdev(map(lambda x: float(x[3]), scores.values()))

    print "userID\tgender\tage\toccupation\tzip\tmovieType\tmode\tmedian\timprovement\tLabels"
    for userId in scores:
        improvement = categorize(float(scores[userId][3]), avg, stdev)
        print '\t'.join([userId] + users[userId][:] + \
                       scores[userId][:] )
