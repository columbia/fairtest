#!/usr/bin/env python
"""
MovieLensALS.py
Copyright (C) 2015, V. Atlidakis vatlidak@cs.columbia.edu>

Tuned version of spark summit exersice to produce movie recommendations.
For each user caclulate the RMSE improvement, the most prominent movie
type of the ones recommended, the mode decade of movies recommended, and
the median decade of the movies recommended.

@argv[1]: A file with movie ratings

@argv[2]: A template file with user's movie ratings template

@argv[3]: A folder to move rendered templates

output: A batch of rendered templates with users' movie ratings
"""
import os
import sys
import itertools
import statistics
from math import sqrt
from operator import add
from collections import Counter
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

def parseRating(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    fields = line.strip().split("::")
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseMovie(line):
    """
    Parses a movie record in MovieLens format movieId::movieTitle::movieType .
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1] + ': ' + fields[2]

def loadRatings(ratingsFile):
    """
    Load ratings from file.
    """
    if not isfile(ratingsFile):
        print "File %s does not exist." % ratingsFile
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        return ''
    else:
        return ratings

def computeRmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))

if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "MovieLensALS.py movieLensDataDir personalRatingsbatch outputFile"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
      .setAppName("MovieLensALS")
    sc = SparkContext(conf=conf)

    # load ratings and movie titles
    movieLensHomeDir = sys.argv[1]

    # ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
    ratings = sc.textFile(join(movieLensHomeDir, "ratings.dat")).map(parseRating)

    # movies is an RDD of (movieId, movieTitle)
    movies = dict(sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseMovie).collect())

    fOut = open(sys.argv[3], 'w')

    counter = 0
    for fileName in os.listdir(sys.argv[2]):
        counter += 1

        # load personal ratings
        myRatings = loadRatings(join(sys.argv[2], fileName))

        if not myRatings:
            continue

        userId = myRatings[0][0]
        myRatings = [(0, x[1], x[2]) for x in myRatings]
        myRatingsRDD = sc.parallelize(myRatings, 1)

        numRatings = ratings.count()
        numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
        numMovies = ratings.values().map(lambda r: r[1]).distinct().count()

        # split ratings into train (60%), validation (20%), and test (20%) based on the 
        # last digit of the timestamp, add myRatings to train, and cache them
        # training, validation, test are all RDDs of (userId, movieId, rating)
        numPartitions = 4
        training = ratings.filter(lambda x: x[0] < 6) \
          .values() \
          .union(myRatingsRDD) \
          .repartition(numPartitions) \
          .cache()

        validation = ratings.filter(lambda x: x[0] >= 6 and x[0] < 8) \
          .values() \
          .repartition(numPartitions) \
          .cache()

        test = ratings.filter(lambda x: x[0] >= 8).values().cache()

        numTraining = training.count()
        numValidation = validation.count()
        numTest = test.count()

        # train models and evaluate them on the validation set

        ranks = [8]
        lambdas = [0.1]
        numIters = [10]
        bestModel = None
        bestValidationRmse = float("inf")
        bestRank = 0
        bestLambda = -1.0
        bestNumIter = -1

        for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
            model = ALS.train(training, rank, numIter, lmbda)
            validationRmse = computeRmse(model, validation, numValidation)
            if (validationRmse < bestValidationRmse):
                bestModel = model
                bestValidationRmse = validationRmse
                bestRank = rank
                bestLambda = lmbda
                bestNumIter = numIter

        testRmse = computeRmse(bestModel, test, numTest)

        # compare the best model with a naive baseline that always returns the mean rating
        meanRating = training.union(validation).map(lambda x: x[2]).mean()
        baselineRmse = sqrt(test.map(lambda x: (meanRating - x[2]) ** 2).reduce(add) / numTest)
        improvement = (baselineRmse - testRmse) / baselineRmse * 100

        # make personalized recommendations
        myRatedMovieIds = set([x[1] for x in myRatings])
        candidates = sc.parallelize([m for m in movies if m not in myRatedMovieIds])
        predictions = bestModel.predictAll(candidates.map(lambda x: (0, x))).collect()
        recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:50]
        if not recommendations:
            continue

        myRatings = [(userId, x[1], x[2]) for x in myRatings]
        movieTypes = []
        movieDates = []
        for i in xrange(len(recommendations)):
            try:
                movieTypes += (movies[recommendations[i][1]]).encode('ascii', 'ignore').split(": ")[1].split("|")[:]
                movieDates.append(int((movies[recommendations[i][1]]).encode('ascii', 'ignore').split(": ")[0][-5:][:4]))
            # In case if misparsed line int() raises exception
            except Exception, error:
                continue

        movieDecades = map(lambda l: ((l % 100) / 10 ) * 10, movieDates)

        print >> fOut, "%d,%s,%d,%d,%.2f" % (myRatings[0][0],
                                # Avoid statistics.mode; StatisticsError is
                                # raised if there is not one most common value
                                Counter(movieTypes).most_common(1)[0][0],
                                Counter(movieDecades).most_common(1)[0][0],
                                int(statistics.median(movieDecades)),
                                improvement
                                )
        print "%d,%d,%s,%d,%d,%.2f" % (counter,
                                myRatings[0][0],
                                Counter(movieTypes).most_common(1)[0][0],
                                Counter(movieDecades).most_common(1)[0][0],
                                int(statistics.median(movieDecades)),
                                improvement
                                )
    sc.stop()
