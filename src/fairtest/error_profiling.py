"""
Error Profiling Investigations
"""

from fairtest.investigation import Investigation, metric_from_string
from fairtest.modules.metrics import NMI, CondDIFF, CORR
import numpy as np
import logging


class ErrorProfiling(Investigation):
    """
    A FairTest Error Profiling Investigation
    """

    def __init__(self, data, protected, output, ground_truth, expl=None,
                 metrics=None, train_size=0.5, conf=0.95,
                 random_state=None):
        """
        Initializes a FairTest Testing Investigation.

        Parameters
        ----------
        data :
            the dataset
        protected :
            list of names of protected features
        output :
            name of output feature
        ground_truth :
            name of ground truth column in the dataset
        expl :
            name of explanatory feature
        metrics :
            dictionary of custom metrics indexed by a protected feature
        train_size :
            fraction of the data to keep for training
        conf :
            confidence level
        random_state :
            seed for random generators
        """

        logging.info('New Error Profiling Investigation')

        if len(np.asarray([output]).flatten()) != 1:
            raise ValueError('Error Profiling investigation excepts a single '
                             'target')

        if output not in data.columns:
            raise ValueError('Unknown target feature %s' % output)

        if len(np.asarray([ground_truth]).flatten()) != 1:
            raise ValueError('Error Profiling investigation excepts a single '
                             'ground truth')

        if ground_truth not in data.columns:
            raise ValueError('Unknown ground truth feature %s' % ground_truth)

        if set(data[output].unique()) == set([0, 1]) and \
                set(data[ground_truth].unique()) == set([0, 1]):
            # binary classification
            logging.info('Computing Binary Classification Error')
            error_name = "Class. Error"
            data[error_name] = ['Correct' if pred == truth
                                else 'FP' if pred else 'FN' for (pred, truth)
                                in zip(data[output], data[ground_truth])]
        elif data.dtypes[output] == np.object:
            # multi-valued classification
            logging.info('Computing Mutlivalued Classification Error')
            error_name = "Class. Error"
            data[error_name] = ['Correct' if pred == truth else 'Incorrect'
                                for (pred, truth) in (zip(data[output],
                                                          data[ground_truth]))]
        else:
            # regression
            logging.info('Computing Absolute Regression Error')
            error_name = "Abs. Error"
            data[error_name] = abs(np.array(data[output]) -
                                   np.array(data[ground_truth]))

        data = data.drop(ground_truth, axis=1)
        data = data.drop(output, axis=1)

        Investigation.__init__(self, data, protected, error_name, expl, metrics,
                               train_size, conf, random_state)

    def set_default_metrics(self):
        out = self.output
        if out.num_labels != 1:
            raise ValueError('Error Profiling investigation excepts a single '
                             'target')

        for sens_str in self.sens_features:
            sens = self.feature_info[sens_str]
            if sens_str in self.metrics:
                if isinstance(self.metrics[sens_str], basestring):
                    self.metrics[sens_str] = \
                        metric_from_string(self.metrics[sens_str])

                expl = self.feature_info[self.expl] if self.expl else None
                self.metrics[sens_str].validate(sens, out, expl)
            else:
                if self.expl:
                    if not self.feature_info[self.expl].arity:
                        raise ValueError(
                            'Only categorical explanatory features allowed')
                    if sens.arity != 2 or out.arity != 2:
                        raise ValueError('Only binary protected features '
                                         'and outputs supported')
                    logging.info('Choosing metric CondDIFF for feature %s' %
                                 sens_str)
                    self.metrics[sens_str] = CondDIFF()
                elif sens.arity and out.arity:
                    logging.info('Choosing metric NMI for feature %s'
                                 % sens_str)
                    self.metrics[sens_str] = NMI()
                else:
                    if sens.arity > 2 or out.arity > 2:
                        raise ValueError('No Metric available for continuous '
                                         'and multi-valued features')
                    logging.info('Choosing metric CORR for feature %s' %
                                 sens_str)
                    self.metrics[sens_str] = CORR()
