import unittest
from fairtest.modules.metrics.correlation import *
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
import numpy as np
import pandas as pd
from . import assertAlmostEqualTuples


class TestingMutualInfo(unittest.TestCase):
    def setUp(self):
        numpy2ri.activate()
        ro.r('set.seed({})'.format(0))
        np.random.seed(0)

    def test_corner_cases(self):
        # degenerate 1
        data = np.array([0, 0, 0, 0, 0, 100])
        self.assertTrue(correlation(data, conf=None) == 0)
        self.assertTrue(correlation(data, conf=0.95)[2] == 1)

        # degenerate 2
        data = pd.DataFrame(np.zeros((100, 2)))
        self.assertTrue(correlation(data, conf=None) == 0)
        self.assertTrue(correlation(data, conf=0.95)[2] == 1)

        # max dependency
        data = np.random.rand(1000, 2)
        data[:, 1] = 2 * data[:, 0]
        data = pd.DataFrame(data)
        self.assertAlmostEqual(correlation(data, conf=None), 1.0)

        assertAlmostEqualTuples(self, correlation(data, conf=0.95), (1, 1, 0),
                                delta=0.01)

        # min dependency
        data = np.random.rand(100000, 2)
        data = pd.DataFrame(data)
        self.assertAlmostEqual(correlation(data, conf=None), 0, delta=0.02)
        assertAlmostEqualTuples(self, correlation(data, conf=0.95)[0:1], (0, 0),
                                delta=0.03)

    def test_approx_stats(self):
        data = np.random.rand(100, 2)
        data = pd.DataFrame(data)

        assertAlmostEqualTuples(self,
                                CORR().compute(data, conf=0.95,
                                               exact=True).stats[0:1],
                                CORR().compute(data, conf=0.95,
                                               exact=False).stats[0:1],
                                delta=0.01)


if __name__ == '__main__':
    unittest.main()
