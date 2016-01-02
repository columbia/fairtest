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
    def __init__(self, data_source, protected, output, expl=None,
                 metrics=None, topk=10, random_state=None, to_drop=None):
        """
        Initializes a FairTest Testing Investigation.

        Parameters
        ----------
        data_source :
            the data source
        protected :
            list of names of protected features
        output :
            name of output features
        expl :
            name of explanatory feature
        metrics :
            dictionary of custom metrics indexed by a protected feature
        topk :
            number of output features with highest association to report on
        random_state :
            seed for random generators
        to_drop :
            features to drop from the training set
        """

        self.topk = topk

        logging.info('New Discovery Investigation')
        Investigation.__init__(self, data_source, protected, output,
                               expl, metrics, random_state, to_drop)

        if self.topk < 1 or self.topk > self.output.num_labels:
            raise ValueError('topk should be in [1, %d]'
                             % self.output.num_labels)

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
                logging.info('Choosing metric REGRESSION for feature %s' %
                             sens_str)
                self.metrics[sens_str] = REGRESSION(topk=self.topk)
