"""
Abstract Fairness Metric.
"""

import abc

import numpy as np


class Metric(object):
    """
    An abstract fairness metric.
    """

    __metaclass__ = abc.ABCMeta

    # Types of metrics
    DATATYPE_CT = 'ct'      # Metrics over a contingency table
    DATATYPE_CORR = 'corr'  # Correlation metrics
    DATATYPE_REG = 'reg'    # Regression metrics

    # this Metric's data type
    dataType = None

    # max data size for approximate tests
    approx_LIMIT_P = None
    # max data size for approximate confidence intervals
    approx_LIMIT_CI = None

    def __init__(self):
        self.stats = None

    def get_size(self, data):
        """
        Returns the size of the data for this metric.

        Parameters
        ----------
        data :
            the data to be evaluated

        Returns
        -------
        size :
            the size of the data
        """
        if self.dataType == self.DATATYPE_CT:
            size = np.array(data).sum()
        elif self.dataType == self.DATATYPE_CORR:
            if np.array(data).shape == (6,):
                size = data[5]
            else:
                size = len(data)
        else:
            size = len(data)
        return size

    def compute(self, data, conf, exact=True):
        """
        Computes a confidence interval and p-value for given data.

        Exact methods are used for confidence intervals and p-values when
        `exact' is set to `True' and the size of the data is smaller than
        respective class attributes `approx_LIMIT_CI' and `approx_LIMIT_P'

        Parameters
        ----------
        data :
            the data to be evaluated
        conf :
            the confidence level for confidence intervals
        exact :
            indicates whether exact methods should be used

        Returns
        -------
        self :
            a pointer to the current Metric object. The computed statistics
            are stored as an attribute `stats'
        """
        size = self.get_size(data)

        if not exact or size > min(self.approx_LIMIT_P, self.approx_LIMIT_CI):
            try:
                ci_low, ci_high, pval = self.approx_stats(data, conf)
            except ValueError:
                ci_low, ci_high, pval = 0, 0, 10*10

        if exact and size <= self.approx_LIMIT_P:
            pval = self.exact_test(data)

        if exact and size <= self.approx_LIMIT_CI:
            ci_low, ci_high = self.exact_ci(data, conf)

        self.stats = [ci_low, ci_high, pval]
        return self

    @abc.abstractmethod
    def abs_effect(self):
        """
        Converts a confidence interval into an absolute effect size that can
        be compared over different contexts.

        Returns
        -------
        effect :
            the absolute effect of this Metric
        """
        return

    @staticmethod
    @abc.abstractmethod
    def exact_test(data):
        """
        Performs an exact test of independence.

        Parameters
        ----------
        data :
            the data to be evaluated

        Returns
        -------
        pval :
            the p-value
        """
        return

    @staticmethod
    @abc.abstractmethod
    def validate(sens, output, expl):
        """
        Validates the use of this metric for the current investigation.

        Parameters
        ----------
        sens :
            the sensitive feature

        output :
            the target feature

        expl :
            the explanatory feature
        """
        return

    @staticmethod
    @abc.abstractmethod
    def exact_ci(data, conf):
        """
        Computes an exact confidence interval.

        Parameters
        ----------
        data :
            the data to be evaluated
        conf :
            the confidence level

        Returns
        -------
        ci_low :
            the lower end of the confidence interval
        ci_high :
            the higher end of the confidence interval
        """
        return

    @staticmethod
    @abc.abstractmethod
    def approx_stats(data, conf):
        """
        Computes an approximate confidence interval and p-value.

        Parameters
        ----------
        data :
            the data to be evaluated
        conf :
            the confidence level

        Returns
        -------
        ci_low :
            the lower end of the confidence interval
        ci_high :
            the higher end of the confidence interval
        pval :
            the p-value
        """
        return
