"""
Simple classifier use to showcase Fairtest API
"""
from sklearn import svm

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
        [2, 3, 3], [0, 4, 1]]
#
# y = [ 'price']
# 'price' in [0, 1]
#
# 'city' == 0 will always get 'price' == 1 and  have people with 'race' == 0
#
Y = [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1,\
        1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1,\
        0, 1, 1, 1, 0]



SPLIT = 10
clf = svm.SVC()

# training the model
x_train = X[:SPLIT]
y_train = Y[:SPLIT]
clf.fit(x_train, y_train)

# testing the model
x_test = X[SPLIT:]
y_test = map(lambda x: clf.predict(x), x_test)

# evaluating the model
print clf.score(x_test, y_test)

# using Fairtest
from fairtest.bugreport import api

testcase = api.Fairtest()

testcase.set_attribute_types(('race', 'city', 'state', 'price'),\
                             ('cat', 'cat', 'cat', 'cat'))
testcase.set_sensitive_attribute('race')
testcase.set_output_attribute('price')

# passing exactly the same arguments
# that were used in clf.score
testcase.bug_report(x_test, y_test)
