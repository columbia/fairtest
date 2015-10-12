"""
Compute the average score of movies in different genres and decades
"""
import sys
import statistics
from os.path import isfile
from collections import Counter
import ast
import re

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
            str(fields[2])]

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
    year = int(re.search(r"\((\d+)\)", fields[1]).group(1))
    decade = year-(year%10)
    return [str(fields[0]),str(fields[2]).split("|"),decade]

def parse_rating(line):
    """
    Parses a recommendation. Format: userId\tgender\tage\toccupation\tavg_rating\trmse\tlabels

    Parameters
    ----------
    line : str
        The line that contains user information

    Returns
    -------
    list : list
        A list containing gender, age, labels
    """
    fields = line.strip().split("::")[:]
    return [str(fields[1]), str(fields[2]), str(fields[0])]

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

def load_movies(movie_file):
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
    if not isfile(movie_file):
        print "File %s does not exist." % movie_file
        sys.exit(1)

    f_in = open(movie_file, 'r')
    movies = {}
    for line in f_in:
        movies[parse_movie(line)[0]] = parse_movie(line)[1:]
    f_in.close()

    return movies

def load_ratings(ratings_file):
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
    if not isfile(ratings_file):
        print "File %s does not exist." % ratings_file
        sys.exit(1)

    f_in = open(ratings_file, 'r')
    ratings = map(parse_rating, f_in.readlines())
    f_in.close()

    return ratings

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Usage: %s ratings_file movies_file users_file" % sys.argv[0]
        sys.exit(1)

    # load ratings and movie titles
    RATINGS_FILE = sys.argv[1]
    MOVIE_FILE = sys.argv[2]
    USERS_FILE = sys.argv[3]

    ratings = load_ratings(RATINGS_FILE)
    movies = load_movies(MOVIE_FILE)
    users = load_users(USERS_FILE)
    
    type_scores = {}
    type_decades = {}
    decade_scores = {}
    gender_scores = {}
    
    for rating in ratings:
        movie = rating[0]
        score = int(rating[1])
        types = movies[movie][0]
        decade = movies[movie][1]
        userID = rating[2]
        gender = users[userID][0]
        #print gender, userID, score
        
        for m_type in types:
            s,n = type_scores.get(m_type, [0,0]) 
            type_scores[m_type] = [s+score, n+1]
            
            d,n = type_decades.get(m_type, [0,0]) 
            type_decades[m_type] = [d+decade, n+1]
            
        s,n = decade_scores.get(decade, [0,0]) 
        decade_scores[decade] = [s+score, n+1]
        
        s,n = gender_scores.get(gender, [0,0]) 
        gender_scores[gender] = [s+score, n+1]
    
    res = {}
    for m_type in type_scores:
        avg = (1.0*type_scores[m_type][0])/type_scores[m_type][1]
        res[m_type] = avg
    
    for (avg, m_type) in sorted(((v,k) for k,v in res.iteritems()), reverse=True):
        print '{}: {:.2f}'.format(m_type, avg)
        
    print    
    
    res = {}
    for m_type in type_decades:
        avg = (1.0*type_decades[m_type][0])/type_decades[m_type][1]
        res[m_type] = avg
    
    for (avg, m_type) in sorted(((v,k) for k,v in res.iteritems()), reverse=True):
        print '{}: {:.2f}'.format(m_type, avg)
        
    print  
    
    res = {}
    for decade in decade_scores:
        avg = (1.0*decade_scores[decade][0])/decade_scores[decade][1]
        res[decade] = avg
    
    for (avg, decade) in sorted(((v,k) for k,v in res.iteritems()), reverse=True):
        print '{}: {:.2f}'.format(decade, avg)
        
    print
    res = {}
    for gender in gender_scores:
        avg = (1.0*gender_scores[gender][0])/gender_scores[gender][1]
        res[gender] = avg
    
    for (avg, gender) in sorted(((v,k) for k,v in res.iteritems()), reverse=True):
        print '{}: {:.2f}'.format(gender, avg)
    
    print gender_scores
        
    '''
      Film-Noir: 4.08
    Documentary: 3.93
            War: 3.89
          Drama: 3.77
          Crime: 3.71
      Animation: 3.68
        Mystery: 3.67
        Musical: 3.67
        Western: 3.64
        Romance: 3.61
       Thriller: 3.57
         Comedy: 3.52
         Action: 3.49
      Adventure: 3.48
         Sci-Fi: 3.47
        Fantasy: 3.45
     Children's: 3.42
         Horror: 3.22
         
    1940: 4.05
    1930: 4.02
    1920: 4.00
    1950: 3.96
    1960: 3.92
    1970: 3.83
    1980: 3.59
    1990: 3.47
    1910: 3.47
    2000: 3.3
    '''

    
