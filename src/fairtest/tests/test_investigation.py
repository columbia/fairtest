import unittest
import fairtest.utils.prepare_data as prepare
from fairtest import Investigation, Testing, train, test, report, DataSource
import numpy as np


class InvestigationTestCase(unittest.TestCase):
    def setUp(self):
        FILENAME = "fairtest/tests/data/tiny_berkeley.csv"
        self.data = prepare.data_from_csv(FILENAME)
        self.SENS = ['gender']
        self.TARGET = 'accepted'
        self.EXPL = None

    def test_randomness(self):
        # First Experiment
        inv1 = Testing(DataSource(self.data), self.SENS, self.TARGET, self.EXPL,
                       random_state=0)
        train([inv1])
        test([inv1])
        report([inv1], "berkeley1", "/tmp")

        shuffle_cols = self.data.columns.copy().tolist()
        np.random.shuffle(shuffle_cols)
        self.data = self.data[shuffle_cols]

        inv2 = Testing(DataSource(self.data), self.SENS, self.TARGET, self.EXPL,
                       random_state=0)
        train([inv2])
        test([inv2])
        report([inv2], "berkeley2", "/tmp")

        with open("/tmp/report_berkeley1.txt") as inf1:
            res1 = inf1.readlines()
        with open("/tmp/report_berkeley2.txt") as inf2:
            res2 = inf2.readlines()

        for (l1, l2) in zip(res1, res2)[4:]:
            self.assertTrue(l1 == l2)

    def test_parameter_check(self):
        data = DataSource(self.data)

        # can't instantiate an abstract investigation
        with self.assertRaises(TypeError):
            Investigation(data, self.SENS, self.TARGET, self.EXPL)

        # must specify a target
        with self.assertRaises(ValueError):
            Testing(data, self.SENS, None)

        # must specify a protected feature
        with self.assertRaises(ValueError):
            Testing(data, None, self.TARGET)

        # must specify a dataset
        with self.assertRaises(ValueError):
            Testing(None, self.SENS, self.TARGET)

        # protected feature must be a list
        with self.assertRaises(ValueError):
            Testing(data, 'single', self.TARGET)

        inv = Testing(data, self.SENS, self.TARGET)

        # inv must be a list
        with self.assertRaises(ValueError):
            train(inv)

        # max_depth should not be negative
        with self.assertRaises(ValueError):
            train(inv, max_depth=-1)

        # min_leaf_size should be non null
        with self.assertRaises(ValueError):
            train([inv], min_leaf_size=0)

        # max_bins should be non null
        with self.assertRaises(ValueError):
            train([inv], max_bins=0)

        # can't test before training
        with self.assertRaises(RuntimeError):
            test([inv])

        # can't report before training and testing
        with self.assertRaises(RuntimeError):
            report([inv], None)

        train([inv])

        # can't report before testing
        with self.assertRaises(RuntimeError):
            report([inv], None)

        test([inv])

        # inv must be a list
        with self.assertRaises(ValueError):
            report(inv, None)

        # filter_conf can't be 1
        with self.assertRaises(ValueError):
            report([inv], None, filter_conf=1)

if __name__ == '__main__':
    unittest.main()
