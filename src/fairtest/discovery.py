"""
Discovery Investigations
"""

from fairtest.investigation import Investigation, metric_from_string
from fairtest.modules.metrics import REGRESSION
import logging


class Discovery(Investigation):
    """
    A FairTest Discovery Investigation
    """
    def __init__(self, data, protected, output, expl=None, metrics=None,
                 train_size=0.5, conf=0.95, topk=10, random_state=None):
        """
        Initializes a FairTest Testing Investigation.

        Parameters
        ----------
        data :
            the dataset
        protected :
            list of names of protected features
        output :
            name of output features
        expl :
            name of explanatory feature
        metrics :
            dictionary of custom metrics indexed by a protected feature
        train_size :
            fraction of the data to keep for training
        conf :
            confidence level
        topk :
            number of output features with highest association to report on
        random_state :
            seed for random generators
        """

        self.topk = topk
        logging.info('New Discovery Investigation')
        Investigation.__init__(self, data, protected, output, expl, metrics,
                               train_size, conf, random_state)

    def set_default_metrics(self):
        out = self.output
        expl = self.feature_info[self.expl] if self.expl else None

        if out.num_labels < 2:
            raise ValueError('Discovery investigation excepts multiple targets')

        for sens_str in self.sens_features:
            sens = self.feature_info[sens_str]
            if sens_str in self.metrics:
                if isinstance(self.metrics[sens_str], basestring):
                    self.metrics[sens_str] = \
                        metric_from_string(self.metrics[sens_str],
                                           topk=self.topk)

                self.metrics[sens_str].validate(sens, out, expl)
            else:
                if sens.arity != 2:
                    raise ValueError('Discovery investigation excepts binary '
                                     'protected feature')
                if expl is not None:
                    raise ValueError('Discovery investigation does not support '
                                     'explanatory features')
                logging.info('Choosing metric REGRESSION for feature %s',
                             sens_str)
                self.metrics[sens_str] = REGRESSION(topk=self.topk)
