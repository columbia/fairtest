#!/usr/bin/python
"""
join_users_scores.py
Copyright (C) 2015, V. Atlidakis vatlidak@cs.columbia.edu>

Joins users file with users' scores file and turns continious
MSE impovement score variable into categorical variable from 0 to 5.

@argv[1]: A file with user demographics

@argv[2]: A file with user scores

output: The join result
"""
import sys
import itertools
import statistics
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

def parseUser(line):
    """
    Parses a user in MovieLens format userID::gender::age::occupation::zip
    """
    fields = line.strip().split("::")
    return [str(fields[0]), str(fields[1]),
            str(fields[2]), str(fields[3]), str(fields[4])]

def parseScore(line):
    """
    Parses a score record. Format: userId,movieType,mode,median,improvement
    """
    fields = line.strip().split(",")[:]
    return [str(fields[0]), str(fields[1]),
            str(fields[2]), str(fields[3]), str(fields[4])]

def loadUsers(usersFile):
    """
    Load ratings from file.
    """
    if not isfile(usersFile):
        print "File %s does not exist." % usersFile
        sys.exit(1)
    f = open(usersFile, 'r')
    users = {}
    for line in f:
        users[parseUser(line)[0]] = parseUser(line)[1:]
    f.close()
    return users

def loadScores(scoresFile):
    """
    Load ratings from file.
    """
    if not isfile(scoresFile):
        print "File %s does not exist." % scoresFile
        sys.exit(1)

    scores = {}
    f = open(scoresFile, 'r')
    for line in f:
        scores[parseScore(line)[0]] = parseScore(line)[1:]
    f.close()
    return scores

def categorize(score, avg, stdev):
    """
    Turn a continous variable into categorical from 0 to 5
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
        print "Usage: %s UsersFile ScoresFile" % sys.argv[0]
        sys.exit(1)

    # load ratings and movie titles
    UsersFile = sys.argv[1]
    ScoresFile = sys.argv[2]

    users = loadUsers(UsersFile)
    scores = loadScores(ScoresFile)

    avg = sum(map(lambda x: float(x[3]), scores.values())) / len(scores.values())
    stdev = statistics.stdev(map(lambda x: float(x[3]), scores.values()))

    print "#userID,gender,age,occupation,zip,movieType,mode,median,improvement"
    for userId in scores:
        improvement = categorize(float(scores[userId][3]), avg, stdev)
        print ','.join([userId] + users[userId][:] + scores[userId][:-1] + [str(improvement)])
