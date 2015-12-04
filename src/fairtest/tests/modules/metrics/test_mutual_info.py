import unittest
from fairtest.modules.metrics.mutual_info import *
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
import numpy as np
from math import log
from . import assertAlmostEqualTuples


class TestingMutualInfo(unittest.TestCase):
    def setUp(self):
        numpy2ri.activate()
        ro.r('set.seed({})'.format(0))
        np.random.seed(0)

    def test_corner_cases_mi(self):
        # degenerate 1
        data = np.array([[10, 20]])
        self.assertTrue(mutual_info(data, norm=True, conf=None) == 0)
        self.assertTrue(mutual_info(data, norm=False, conf=None) == 0)
        self.assertTrue(mutual_info(data, norm=True, conf=0.95) == (0, 1, 1))

        # degenerate 2
        data = np.array([[10], [20]])
        self.assertTrue(mutual_info(data, norm=True, conf=None) == 0)
        self.assertTrue(mutual_info(data, norm=False, conf=None) == 0)
        self.assertTrue(mutual_info(data, norm=True, conf=0.95) == (0, 1, 1))

        # max dependency
        data = np.array([[1000, 0], [0, 1000]])
        self.assertAlmostEqual(mutual_info(data, norm=True, conf=None), 1,
                               delta=0.02)
        self.assertAlmostEqual(mutual_info(data, norm=False, conf=None), log(2),
                               delta=0.02)

        # min dependency
        data = np.array([[1000, 100], [1000, 100]])
        self.assertAlmostEqual(mutual_info(data, norm=True, conf=None), 0,
                               delta=0.02)
        self.assertAlmostEqual(mutual_info(data, norm=False, conf=None), 0,
                               delta=0.02)

        data = np.array([[1000, 0], [0, 1000]])
        tuple1 = mutual_info(data, norm=False, conf=0.95)
        tuple2 = (0.64, 0.74, 0)
        assertAlmostEqualTuples(self, tuple1, tuple2, delta=0.01)

    def test_approx_stats_mi(self):
        data = np.array([[300, 150], [150, 300]])
        assertAlmostEqualTuples(self,
                                NMI().compute(data, conf=0.95,
                                              exact=True).stats[0:1],
                                NMI().compute(data, conf=0.95,
                                              exact=False).stats[0:1],
                                delta=0.01)

    def test_corner_cases_cond_mi(self):
        # degenerate 1
        data = np.array([[[100, 0]]])
        self.assertTrue(cond_mutual_info(data, norm=True) == 0)
        self.assertTrue(cond_mutual_info(data, norm=False) == 0)

        # max dependency
        data = np.array([[[1000, 0], [0, 1000]], [[0, 1000], [1000, 0]]])
        self.assertAlmostEqual(cond_mutual_info(data, norm=True), 1,
                               delta=0.02)
        self.assertAlmostEqual(cond_mutual_info(data, norm=False), log(2),
                               delta=0.02)

        # min dependency
        data = np.array([[[1000, 1000], [1000, 1000]],
                         [[1000, 1000], [1000, 1000]]])
        self.assertAlmostEqual(cond_mutual_info(data, norm=True), 0,
                               delta=0.02)
        self.assertAlmostEqual(cond_mutual_info(data, norm=False), 0,
                               delta=0.02)

if __name__ == '__main__':
    unittest.main()
