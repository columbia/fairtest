#!/usr/bin/python
"""
render_personal_ratings.py
Copyright (C) 2015, V. Atlidakis vatlidak@cs.columbia.edu>


@argv[1]: A file with movie ratings

@argv[2]: A template file with user's movie ratings template

@argv[3]: A folder to move rendered templates

output: A batch of rendered templates with users' movie ratings

References
----------
http://ampcamp.berkeley.edu/big-data-mini-course/movie-recommendation-with-mllib
"""
import sys
from os.path import join, isfile

def parse_rating(line):
    """
    Parses a rating record in MovieLens format 
    user_id::movie_id::rating::timestamp

    Parameters
    ----------
    line : str
        A line that contains movie rating info

    Returns
    -------
    tuple : tuple
        A tuple containing userID, movie_id, and rating
    """
    fields = line.strip().split("::")
    return (int(fields[0]), int(fields[1]), float(fields[2]))

def parse_movie(line):
    """
    Parses a movie record in MovieLens format movie_id::movieTitle

    Parameters
    ----------
    line : str
        A line that contains movie info

    Returns
    -------
    tuple : tuple
        A tuple containing movie_id and movieTitle
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1]

def load_ratings(retings_file):
    """
    Load ratings from file.

    Parameters
    ----------
    retings_file : str
        The name of a file to be loaded

    Returns
    -------
    ratings : dict
        A dictionary with user ratings information.
    """
    if not isfile(retings_file):
        print "File %s does not exist." % retings_file
        sys.exit(1)
    f_in = open(retings_file, 'r')

    ratings = {}
    for line in f_in:
        user_id, movie_id, rating = parse_rating(line)
        if user_id not in ratings:
            ratings[user_id] = {}
        ratings[user_id][movie_id] = rating
    f_in.close()
    if not ratings:
        print "No ratings provided."
        sys.exit(1)
    else:
        return ratings

def fill_template(template_file, out_dir, user_ratings, user_id):
    """
    Load ratings from file.

    Parameters
    ----------
    template_file : str
        The name of the template to be loaded

    out_dir : str
        The name of the output directory

    user_ratings : dict
        A dictionary containing user ratings

    user_id : str
        The id of the user in whose rating we are interested.
    """
    if not isfile(template_file):
        print "File %s does not exist." % template_file
        sys.exit(1)
    f_in = open(template_file, 'r')
    f_out = open(join(out_dir, "personalRatings." + str(user_id) + ".txt"), 'w')

    for line in f_in:
        current = line.strip().split("::")[:]
        movie_id = int(current[1])

        #hasn't seen movie
        if movie_id not in user_ratings:
            current[2] = str(0)
        else:
            current[2] = str(int(user_ratings[movie_id]))
        current[0] = str(user_id)
        print >> f_out, '::'.join(current)
    f_in.close()
    f_out.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Usage: %s retings_file template_file out_dir" % sys.argv[0]
        sys.exit(1)

    # load ratings and movie titles
    RATINGS_FILE = sys.argv[1]
    TEMPLATE_FILE = sys.argv[2]
    OUTPUT_DIR = sys.argv[3]

    # ratings is a RDD of (last digit of timestamp, (user_id, movie_id, rating))
    RATINGS = load_ratings(RATINGS_FILE)
    for user_id in RATINGS:
        fill_template(TEMPLATE_FILE, OUTPUT_DIR, RATINGS[user_id], user_id)
