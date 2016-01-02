import unittest
import fairtest.utils.prepare_data as prepare
from fairtest import Discovery, DataSource
import ast
from sklearn import preprocessing
import pandas as pd


class TestingTestCase(unittest.TestCase):
    def setUp(self):
        FILENAME = "../data/images/overfeat_raw.txt"
        data = prepare.data_from_csv(FILENAME, sep='\\t')

        TARGET = 'Labels'
        self.SENS = ['Race']
        self.EXPL = []

        labeled_data = [ast.literal_eval(s) for s in data[TARGET]]
        for l in labeled_data:
            assert len(l) == 5
        label_encoder = preprocessing.MultiLabelBinarizer()
        labeled_data = label_encoder.fit_transform(labeled_data)
        labels = label_encoder.classes_
        df_labels = pd.DataFrame(labeled_data, columns=labels)
        self.data = DataSource(pd.concat([data.drop(TARGET, axis=1), df_labels],
                                         axis=1))
        self.TARGET = labels.tolist()

    def test_parameters(self):
        # topk should be positive
        with self.assertRaises(ValueError):
            Discovery(self.data, self.SENS, self.TARGET, topk=0)

        # topk can't be bigger than the number of labels
        with self.assertRaises(ValueError):
            Discovery(self.data, self.SENS, self.TARGET,
                      topk=len(self.TARGET) + 1)

    def test_metrics(self):
        # can't have a single target
        with self.assertRaises(ValueError):
            Discovery(self.data, self.SENS, self.TARGET[0], self.EXPL)

        # sensitive feature not found
        with self.assertRaises(ValueError):
            Discovery(self.data, ['Racc'], self.TARGET, self.EXPL)

        # target feature not found
        with self.assertRaises(ValueError):
            Discovery(self.data, self.SENS, self.TARGET + ['unknown'], self.EXPL)

        # unknown metric
        with self.assertRaises(ValueError):
            metrics = {'Race': 'unknown'}
            Discovery(self.data, self.SENS, self.TARGET, self.EXPL,
                      metrics=metrics)

        # explanatory feature
        with self.assertRaises(ValueError):
            Discovery(self.data, self.SENS, self.TARGET, 'Race',
                      metrics=metrics)

if __name__ == '__main__':
    unittest.main()
