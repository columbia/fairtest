import unittest
import fairtest.utils.prepare_data as prepare
from fairtest import Testing, DataSource


class TestingTestCase(unittest.TestCase):
    def setUp(self):
        FILENAME = "../data/adult/adult.csv"
        self.data = DataSource(prepare.data_from_csv(FILENAME))
        self.SENS = ['sex']
        self.TARGET = 'income'
        self.EXPL = None

    def test_metrics(self):
        # can't have more than one target
        with self.assertRaises(ValueError):
            Testing(self.data, self.SENS, ['income', 'age'], self.EXPL)

        # sensitive feature not found
        with self.assertRaises(ValueError):
            Testing(self.data, ['sexx'], self.TARGET, self.EXPL)

        # target feature not found
        with self.assertRaises(ValueError):
            Testing(self.data, self.SENS, 'incme', self.EXPL)

        # unknown metric
        with self.assertRaises(ValueError):
            metrics = {'sex': 'unknown'}
            Testing(self.data, self.SENS, self.TARGET, self.EXPL,
                    metrics=metrics)

        # correlation with categorical feature
        with self.assertRaises(ValueError):
            SENS = ['education']
            metrics = {'education': 'CORR'}
            Testing(self.data, SENS, self.TARGET, self.EXPL, metrics=metrics)

        # categorical + continuous feature
        with self.assertRaises(ValueError):
            SENS = ['education']
            TARGET = 'age'
            Testing(self.data, SENS, TARGET, self.EXPL)

if __name__ == '__main__':
    unittest.main()
