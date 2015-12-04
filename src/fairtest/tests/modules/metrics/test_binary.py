import unittest
from fairtest.modules.metrics.binary_metrics import *
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
        # degenerate 1
        data = np.array([[10, 20]])
        self.assertTrue(difference(data, conf=None) == 0)
        self.assertTrue(difference(data, conf=0.95) == (0, 1, 1))
        self.assertTrue(ratio(data, conf=None) == 1)
        self.assertTrue(ratio(data, conf=0.95) == (1, 1, 1))

        # degenerate 2
        data = np.array([[10], [20]])
        self.assertTrue(difference(data, conf=None) == 0)
        self.assertTrue(difference(data, conf=0.95) == (0, 1, 1))
        self.assertTrue(ratio(data, conf=None) == 1)
        self.assertTrue(ratio(data, conf=0.95) == (1, 1, 1))

        # max dependency
        data = np.array([[1000, 0], [0, 1000]])
        self.assertAlmostEqual(difference(data, conf=None), -1,
                               delta=0.02)
        self.assertAlmostEqual(ratio(data, conf=None), 0,
                               delta=0.02)

        # min dependency
        data = np.array([[1000, 100], [1000, 100]])
        self.assertAlmostEqual(difference(data, conf=None), 0,
                               delta=0.02)
        self.assertAlmostEqual(ratio(data, conf=None), 1,
                               delta=0.02)

    def test_corner_cases_cond_diff(self):
        # degenerate 1
        data = np.array([[[100, 0]]])
        self.assertTrue(cond_difference(data) == 0)

        # max dependency
        data = np.array([[[1000, 0], [0, 1000]], [[1000, 0], [0, 1000]]])
        self.assertAlmostEqual(cond_difference(data), -1, delta=0.02)

        # average
        data = np.array([[[1000, 0], [0, 1000]], [[0, 1000], [1000, 0]]])
        self.assertAlmostEqual(cond_difference(data), 0, delta=0.02)

        # min dependency
        data = np.array([[[1000, 1000], [1000, 1000]],
                         [[1000, 1000], [1000, 1000]]])
        self.assertAlmostEqual(cond_difference(data), 0, delta=0.02)

    def test_approx_stats(self):
        data = np.array([[300, 150], [150, 300]])

        assertAlmostEqualTuples(self,
                                DIFF().compute(data, conf=0.95,
                                               exact=True).stats[0:1],
                                DIFF().compute(data, conf=0.95,
                                               exact=False).stats[0:1],
                                delta=0.01)

        assertAlmostEqualTuples(self,
                                RATIO().compute(data, conf=0.95,
                                               exact=True).stats[0:1],
                                RATIO().compute(data, conf=0.95,
                                               exact=False).stats[0:1],
                                delta=0.01)


if __name__ == '__main__':
    unittest.main()
