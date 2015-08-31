#!/usr/bin/python
"""
render_personal_ratings.py
Copyright (C) 2015, V. Atlidakis vatlidak@cs.columbia.edu>


@argv[1]: A file with movie ratings

@argv[2]: A template file with user's movie ratings template

@argv[3]: A folder to move rendered templates

output: A batch of rendered templates with users' movie ratings
"""
import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

def parseRating(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .

    Returns userID,movieId,rating
    """
    fields = line.strip().split("::")
    return (int(fields[0]), int(fields[1]), float(fields[2]))

def parseMovie(line):
    """
    Parses a movie record in MovieLens format movieId::movieTitle .
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1]

def loadRatings(ratingsFile):
    """
    Load ratings from file.
    """
    if not isfile(ratingsFile):
        print "File %s does not exist." % ratingsFile
        sys.exit(1)
    f = open(ratingsFile, 'r')

    ratings = {}
    for line in f:
        userId , movieId, rating = parseRating(line)
        if userId not in ratings:
            ratings[userId] = {}
        ratings[userId][movieId] = rating
    f.close()
    if not ratings:
        print "No ratings provided."
        sys.exit(1)
    else:
        return ratings

def fillTemplate(templateFile, outDir, userRatings, userId):
    """
    Load ratings from file.
    """
    if not isfile(templateFile):
        print "File %s does not exist." % templateFile
        sys.exit(1)
    fIn = open(templateFile, 'r') 
    fOut =  open(join(outDir,"personalRatings." + str(userId) + ".txt"), 'w')

    for line in fIn:
        current = line.strip().split("::")[:]
        movieId = int(current[1])

        #hasn't seen movie
        if movieId not in userRatings:
            current[2] = str(0)
        else:
            current[2] = str(int(userRatings[movieId]))
        current[0] = str(userId)
        print >> fOut, '::'.join(current)
    fIn.close()
    fOut.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Usage: %s ratingsFile templateFile outDir" % sys.argv[0]
        sys.exit(1)

    # load ratings and movie titles
    ratingsFile = sys.argv[1]
    templateFile = sys.argv[2]
    outDir = sys.argv[3]

    # ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
    ratings = loadRatings(ratingsFile)
    for userId in ratings:
        fillTemplate(templateFile, outDir, ratings[userId], userId)
