"""
Simple examples
"""
from fairtest.bugreport import api2 as api
from fairtest.bugreport.helpers import prepare



#########################################
#                                       #
# Experiment 1                          #
#                                       #
# Using Fairtest with in memory data    #
#                                       #
#########################################


#
# X = [ 'race', 'city', 'state']
# 'race' in [0, 1, 2]
# 'city' in [0, 1, 2, 3, 4]
# 'state' in [0, 1, 2, 3, 4]
#
X = [[1, 2, 2], [0, 0, 2], [0, 0, 1], [0, 0, 3], [2, 4, 3], [0, 0, 3],\
        [1, 4, 3], [2, 2, 2], [0, 0, 0], [0, 0, 3], [1, 2, 0], [0, 1, 1],\
        [2, 4, 4], [2, 4, 2], [2, 1, 3], [1, 4, 3], [1, 4, 3], [0, 3, 4],\
        [2, 2, 4], [2, 4, 2], [0, 0, 3], [2, 3, 3], [0, 0, 2], [0, 0, 2],\
        [0, 2, 2], [1, 3, 4], [0, 3, 4], [2, 1, 1], [2, 4, 4], [2, 1, 4],\
        [0, 4, 4], [0, 4, 4], [1, 2, 2], [1, 2, 2], [0, 0, 1], [1, 2, 1],\
        [0, 0, 1], [1, 1, 1], [0, 2, 2], [1, 2, 4], [1, 2, 0], [0, 0, 1],\
        [2, 3, 2], [0, 0, 3], [2, 3, 1], [2, 1, 0], [1, 2, 0], [0, 4, 2],\
        [2, 3, 3], [0, 4, 1]]*1000

#
# y = [ 'price']
# 'price' in [0, 1]
#
# 'city' == 0 will always get 'price' == 1 and  have people with 'race' == 0
#
y = [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1,\
        1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1,\
        0, 1, 1, 1, 0]*1000

# Preapre data into FairTest friendly format
data = prepare.data_from_mem(X, y)


# Initializing parameters for experiment
EXPL = []
SENS = [0]
TARGET = 3


# Instanciate the experiment
FT2 = api.Experiment(data, SENS, TARGET, EXPL)

# Train the classifier
FT2.train()

# Evaluate on the testing set
FT2.test()

# Create the report
FT2.report("demoset-1")



#########################################
#                                       #
# Experiment 2                          #
#                                       #
# Using Fairtest with in csv file       #
#                                       #
#########################################


# Preapre data into FairTest friendly format
FILENAME = "/home/vatlidak/repos/fairtest/data/staples/staples.csv"
data = prepare.data_from_csv(FILENAME, ['zipcode', 'distance'])


# Initializing parameters for experiment
EXPL = []
SENS = ['income']
TARGET = 'price'

# Instanciate the experiment
FT2 = api.Experiment(data, SENS, TARGET, EXPL)

# Train the classifier
FT2.train()

# Evaluate on the testing set
FT2.test()

# Create the report
FT2.report("demoset-2")
