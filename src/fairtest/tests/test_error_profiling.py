import unittest
import fairtest.utils.prepare_data as prepare
from fairtest import ErrorProfiling, DataSource


class TestingTestCase(unittest.TestCase):
    def setUp(self):
        FILENAME = "fairtest/tests/data/tiny_predictions_reg.csv"
        self.data = DataSource(prepare.data_from_csv(FILENAME))
        self.SENS = ['Age']
        self.TARGET = 'Prediction'
        self.GROUND_TRUTH = 'Ground_Truth'
        self.EXPL = None

    def test_metrics(self):
        # can't have more than one target
        with self.assertRaises(ValueError):
            ErrorProfiling(self.data, self.SENS, ['Prediction', 'Age'],
                           self.GROUND_TRUTH)

        # sensitive feature not found
        with self.assertRaises(ValueError):
            ErrorProfiling(self.data, ['agee'], self.TARGET, self.GROUND_TRUTH)

        # target feature not found
        with self.assertRaises(ValueError):
            ErrorProfiling(self.data, self.SENS, 'Predction', self.GROUND_TRUTH)

        # ground truth not found
        with self.assertRaises(ValueError):
            ErrorProfiling(self.data, self.SENS, self.TARGET, 'Ground Trth')

        # unknown metric
        with self.assertRaises(ValueError):
            metrics = {'Age': 'unknown'}
            ErrorProfiling(self.data, self.SENS, self.TARGET, self.GROUND_TRUTH,
                           metrics=metrics)

if __name__ == '__main__':
    unittest.main()
