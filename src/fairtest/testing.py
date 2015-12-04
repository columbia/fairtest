"""
Testing Investigations
"""

from fairtest.investigation import Investigation, metric_from_string
from fairtest.modules.metrics import NMI, CondDIFF, CORR
import logging


class Testing(Investigation):
    """
    A FairTest Testing Investigation
    """

    def __init__(self, data, protected, output, expl=None, metrics=None,
                 train_size=0.5, conf=0.95, random_state=None):
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
        logging.info('New Testing Investigation')
        Investigation.__init__(self, data, protected, output, expl, metrics,
                               train_size, conf, random_state)

    def set_default_metrics(self):
        out = self.output
        expl = self.feature_info[self.expl] if self.expl else None

        if out.num_labels != 1:
            raise ValueError('Testing investigation excepts a single target')

        for sens_str in self.sens_features:
            sens = self.feature_info[sens_str]
            if sens_str in self.metrics:
                if isinstance(self.metrics[sens_str], basestring):
                    self.metrics[sens_str] = \
                        metric_from_string(self.metrics[sens_str])
                self.metrics[sens_str].validate(sens, out, expl)
            else:
                if expl is not None:
                    if not expl.arity:
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
                    logging.info('Choosing metric CORR for feature %s'
                                 % sens_str)
                    self.metrics[sens_str] = CORR()

