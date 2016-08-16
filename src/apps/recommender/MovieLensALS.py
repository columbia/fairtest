#!/usr/bin/env python

import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
import numpy as np
import operator

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

def parseRating(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::count .
    """
    fields = line.strip().split("::")
    return int(fields[3]), (int(fields[0]), int(fields[1]), float(fields[2]))

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
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print "No ratings provided."
        sys.exit(1)
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
    if (len(sys.argv) != 2):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "MovieLensALS.py movieLensDataDir"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
      .setAppName("MovieLensALS") \
      .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir('checkpoint/')
    
    # load ratings and movie titles
    movieLensHomeDir = sys.argv[1]

    # ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
    ratings = sc.textFile(join(movieLensHomeDir, "ratings_new.dat")).map(parseRating)
    
    # movies is an RDD of (movieId, movieTitle)
    movies = dict(sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseMovie).collect())

    numRatings = ratings.count()
    numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
    numMovies = ratings.values().map(lambda r: r[1]).distinct().count()

    print "Got %d ratings from %d users on %d movies." % (numRatings, numUsers, numMovies)
    
    
    # compute average rating of each movie
    rating_list = {}
    movie_avgs = {}
    for rating in ratings.values().collect():
        userID = rating[0]
        movieID = rating[1]
        if userID not in rating_list:
            rating_list[userID] = []
        if movieID not in movie_avgs:
            movie_avgs[movieID] = [0,0]
            
        rating_list[userID].append(movieID)
        movie_avgs[movieID][0] += rating[2]
        movie_avgs[movieID][1] += 1
    
    for movie in movie_avgs:
        movie_avgs[movie] = [(1.0*movie_avgs[movie][0])/movie_avgs[movie][1], movie_avgs[movie][1]]
    
    # popular = [(idx, avg) for (idx, (avg, n)) in movie_avgs.iteritems() if n > 10]
    # print 'Top movies:'
    # for (idx, avg) in sorted(popular, key=operator.itemgetter(1), reverse=True)[0:50]:
    #     print movies[idx].encode('ascii', 'ignore'), avg
    
    
    # keep 10 ratings per user as a test set and split the rest into training
    # and validation sets

    # training, validation, test are all RDDs of (userId, movieId, rating)
    numPartitions = 4
    
    (trainingData, validData) = ratings.filter(lambda x: x[0] >= 10).randomSplit([0.67, 0.33])
    
    training = trainingData \
      .values() \
      .repartition(numPartitions) \
      .cache()
      
    validation = validData \
      .values() \
      .repartition(numPartitions) \
      .cache()
    
    # hold-out 10 ratings per user
    test = ratings.filter(lambda x: x[0] < 10).values().cache()

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    print "Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)
    
    # train models and evaluate them on the validation set

    ranks = [12]
    lambdas = [0.1]
    numIters = [20]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1

    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        model = ALS.train(training, rank, numIter, lmbda)
        validationRmse = computeRmse(model, validation, numValidation)
        print "RMSE (validation) = %f for the model trained with " % validationRmse + \
              "rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter)
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter

    testRmse = computeRmse(bestModel, test, numTest)

    # evaluate the best model on the test set
    print "The best model was trained with rank = %d and lambda = %.1f, " % (bestRank, bestLambda) \
      + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse)

    # compare the best model with a naive baseline that always returns the mean rating
    meanRating = training.union(validation).map(lambda x: x[2]).mean()
    baselineRmse = sqrt(test.map(lambda x: (meanRating - x[2]) ** 2).reduce(add) / numTest)
    improvement = (baselineRmse - testRmse) / baselineRmse * 100
    print "The best model improves the baseline by %.2f" % (improvement) + "%."
    
    f_out = open('recommendations_raw.txt', 'w')
    for user in test.map(lambda r: r[0]).distinct().collect():
        # get the movies rated by the user
        user_movies = set(rating_list[user])
        
        # get all movies not rated by the user
        candidates = sc.parallelize([m for m in movies if m not in user_movies])
        
        # find 50 predictions
        predictions = bestModel.predictAll(candidates.map(lambda x: (user, x)))
        recommendations = sorted(predictions.collect(), key=lambda x: x[2], reverse=True)[:50]
        labels = [movies[x[1]].encode('ascii', 'ignore') for x in recommendations]
        
        # average ratings of the movies
        user_movies_avgs = [movie_avgs[x[1]][0] for x in recommendations]
        user_weighted_avg = np.average(user_movies_avgs, weights=None)
        
        # average ratings of movies seen
        user_movies_avgs_seen = np.average([movie_avgs[x][0] for x in user_movies])

        # get the user's test set and compute the error
        user_test = test.filter(lambda r: r[0] == user)
        n = user_test.count()
        assert n == 10
        rmse = computeRmse(bestModel, user_test, n)
        
        user_movies_avgs = [movie_avgs[x[1]][0] for x in recommendations]
        user_movies_weights = [movie_avgs[x[1]][1] for x in recommendations]
        user_weighted_avg = np.average(user_movies_avgs, weights=user_movies_weights)
        avg_score = np.mean([x[2] for x in recommendations])
        print '{}\t{}\t{}\t{}\t{}'.format(user, user_weighted_avg, user_movies_avgs_seen, rmse, labels)
        print >> f_out, '{}\t{}\t{}\t{}\t{}'.format(user, user_weighted_avg, user_movies_avgs_seen, rmse, labels)
    # clean up
    sc.stop()
    