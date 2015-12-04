"""
Unit Tests for Metrics Module
"""


def assertAlmostEqualTuples(test, t1, t2, delta):
    [test.assertAlmostEqual(i1, i2, delta=delta) for (i1, i2) in zip(t1, t2)]
