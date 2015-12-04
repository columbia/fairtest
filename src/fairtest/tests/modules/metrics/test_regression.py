import unittest
from fairtest.modules.metrics.regression import *
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
import numpy as np
from . import assertAlmostEqualTuples


class TestingMutualInfo(unittest.TestCase):
    def setUp(self):
        numpy2ri.activate()
        ro.r('set.seed({})'.format(0))
        np.random.seed(0)

    def test_corner_cases(self):
        data = np.random.rand(1000, 50)
        data = pd.DataFrame(data)

        stats = REGRESSION(topk=10).compute(data, conf=0.95).stats
        stats = stats.reset_index(drop=True)
        for idx in stats.index:
            assertAlmostEqualTuples(self, stats.iloc[idx][0:1], (0, 0),
                                    delta=0.01)

if __name__ == '__main__':
    unittest.main()

