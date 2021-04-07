import unittest
from fairtest.modules.metrics.regression import *
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
import numpy as np


class TestingRegression(unittest.TestCase):
    def setUp(self):
        numpy2ri.activate()
        ro.r('set.seed({})'.format(0))
        np.random.seed(0)

    def test_corner_cases(self):
        NUM_SAMPLES = 1000
        NUM_FEATURES = 50

        data = np.random.rand(NUM_SAMPLES, NUM_FEATURES)
        data = pd.DataFrame(data)
        data[data.columns[-1]] = np.random.randint(0, 2, NUM_SAMPLES)

        stats = REGRESSION(topk=100).compute(data, conf=0.95).stats
        stats = stats.reset_index(drop=True)
        count = 0
        for idx in stats.index:
            if stats.iloc[idx][2] <= 0.05:
                count += 1

        # we expect roughly 5% of the p-values to be significant
        self.assertTrue(0.025 <= count/(1.0 * NUM_SAMPLES) <= 0.075)

if __name__ == '__main__':
    unittest.main()

