"""
Parse ratings file and replace timestamp with a counter for each user
"""
import sys
import statistics
from os.path import isfile
from random import shuffle

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: %s ratings_file output_file" % sys.argv[0]
        sys.exit(1)

    # load ratings and movie titles
    ratings_file = sys.argv[1]
    output_file = sys.argv[2]
    
    min_num = float('inf')
    with open(ratings_file) as inf:
        with open(output_file, 'w') as outf:
            ratings = map(lambda l: l.split('::'), inf.readlines())
            
            num_ratings = {}
            
            for r in ratings:
                userID = r[0]
                if userID not in num_ratings:
                    num_ratings[userID] = 1
                else:
                    num_ratings[userID] += 1
            
            numbers = []
            curr_id = -1
            for r in ratings:
                userID = r[0]
                if userID != curr_id:
                    curr_id = userID
                    user_count = num_ratings[userID]
                    numbers = range(user_count)
                    shuffle(numbers)
                
                print >> outf, '::'.join(r[0:-1]+[str(numbers.pop(0))])
    
    print 'minimum number of ratings = {}'.format(min_num+1)
