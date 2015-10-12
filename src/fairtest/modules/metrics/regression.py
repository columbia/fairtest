"""
Regression metric.
"""

import fairtest.modules.statistics.hypothesis_test as tests
import fairtest.modules.statistics.confidence_interval as intervals
from .metric import Metric
from .binary_metrics import DIFF
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


class REGRESSION(Metric):
    """
    Regression metric.
    """
    dataType = Metric.DATATYPE_REG

    def __init__(self, topk=10):
        Metric.__init__(self)
        self.topk = topk

    def compute(self, data, level, exact=True):

        # regression not yet trained
        if self.stats is None:
            sens = data[data.columns[-1]]
            labels = data[data.columns[0:-1]]

            reg = LogisticRegression()
            reg.fit(labels, sens)
            sens_pred = reg.predict(labels)

            # approximate the standard errors for all regression coefficients
            mse = np.mean((sens - sens_pred.T)**2)
            var_est = mse * np.diag(np.linalg.pinv(np.dot(labels.T, labels)))
            std_est = np.sqrt(var_est)
            coeffs = reg.coef_[0].tolist()

            # compute confidence intervals and p-values for all coefficients
            results = pd.DataFrame(coeffs, columns=['coeff'])

            results['std'] = std_est
            results['pval'] = tests.z_test(results['coeff'], results['std'])

            ci_s = intervals.ci_norm(level, results['coeff'], results['std'])
            results['ci_low'] = ci_s[0]
            results['ci_high'] = ci_s[1]

            # compute a standardized effect size
            # and return the topK coefficients
            results['effect'] = [intervals.z_effect(low, high)
                                 for (low, high) in
                                 zip(results['ci_low'], results['ci_high'])]
            sorted_results = results.sort(columns=['effect'], ascending=False)

            self.stats = \
                sorted_results[['ci_low', 'ci_high', 'pval']].head(self.topk)
            # print self.stats
            return self
        else:
            # model was already trained, get the top labels
            top_labels = self.stats.index

            for idx in top_labels:
                ct = pd.crosstab(data[data.columns[idx]],
                                 data[data.columns[-1]])
                self.stats.loc[idx] = DIFF().compute(ct, level=level,
                                                     exact=exact).stats
            return self

    def abs_effect(self):
        effects = np.array([intervals.z_effect(ci_low, ci_high)
                            for (ci_low, ci_high, _) in self.stats.values])

        effects[np.isnan(effects)] = 0
        return np.mean(effects)

    @staticmethod
    def approx_stats(data, level):
        raise NotImplementedError()

    @staticmethod
    def exact_test(data):
        raise NotImplementedError()

    @staticmethod
    def exact_ci(data, level):
        raise NotImplementedError()

    @staticmethod
    def validate(sens, output, expl):
        if output.num_labels < 2:
            raise ValueError('REGRESSION metric expects multiple targets')
        if expl is not None:
            raise ValueError('REGRESSION metric not usable with explanatory '
                             'features')
        if not sens.arity == 2:
            raise ValueError('REGRESSION metric only usable with binary '
                             'protected features')

    def __str__(self):
        return 'REG(topk={})'.format(self.topk)
