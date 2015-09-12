"""
Implement the Fairtest API
"""
from fairtest.bugreport.trees import tree_builder as builder
from fairtest.bugreport.clustering import tree_clustering as tc
from fairtest.bugreport.clustering import display_clusters as displ
from fairtest.bugreport.statistics import fairness_measures as fm
from fairtest.bugreport.statistics import multiple_testing as multitest

import pandas as pd
import numpy as np

class Fairtest:
    """
    The main class
    """
    def __init__(self, feature_info, output_info, measures={},
                 max_depth=5, min_leaf_size=100, agg_type="AVG", max_bins=10,
                 topk=50, ci_level=0.95):
        """
        Initialize a FairTest experiment

        Parameters
        ----------
        feature_info :
            a list of @Feature objects for each data feature

        output_info :
            a @Target object for the target feature

        measures :
            a dictionary with a mapping sensitive_feature -> fairness metric. If
            not specified, default fairness metrics are chosen

        max_depth :
            maximum tree generation depth

        min_min_leaf_size :
            minimum size of a tree leaf

        agg_type :
            aggregation method for child fairness score

        max_bins :
            maximum number of bins for continuous features

        topk :
            maximum number of labels to report for multi-output discrimination
            discovery

        ci_level :
            confidence level for fairness confidence intervals
        """

        # TODO validate input

        self.output = output_info
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.agg_type = agg_type
        self.max_bins = max_bins
        self.topk = topk
        self.ci_level = ci_level

        self.trained_trees = {}
        self.contexts = {}
        self.stats = {}

        # get the names of the sensitive features
        self.sens_features = [f.name for f in feature_info if f.ftype == 'sens']

        # get the name of the explanatory feature (if any)
        expl_list = [f.name for f in feature_info if f.ftype == 'expl']
        self.expl = expl_list[0] if expl_list else None

        # store a dictionary to easily access feature information
        self.feature_info = {f.name: f for f in feature_info}
        self.feature_names = [f.name for f in feature_info]

    def train(self, x, y):
        """
        Forms hypotheses about discrimination contexts

        Parameters
        ----------
        x :
            the user data

        y :
            the target data
        """
        # TODO validate input

        # create a Pandas DataFrame with columns index by feature name
        data = prepare_data(x, y, self.feature_names, self.output)

        # find discrimination contexts for each sensitive feature
        for sens in self.sens_features:

            # TODO use provided measure if available
            # get a default measure
            measure = get_measure(self.feature_info[sens], self.output,
                                      self.ci_level, self.topk, self.expl)

            tree = builder.train_tree(data, self.feature_info, sens,
                                      self.expl, self.output, measure,
                                      self.max_depth,self.min_leaf_size,
                                      self.agg_type, self.max_bins)
            self.trained_trees[sens] = tree

    def test(self, x, y, prune_insignificant=False, approx=True, fdr=0.05):
        """
        Tests formed hypotheses about discrimination

        Parameters
        ----------
        x :
            the user test data

        y :
            the target test data

        prune_insignificant :
            if True, hypotheses that are unlikely to be true on the training
            set are discarded

        approx :
            if True, approximate asymptotically correct methods are used to
            generate p-values and confidence intervals

        fdr :
            global False-discovery rate
        """
        # TODO validate input, check that tree was trained

        # create a Pandas DataFrame with columns index by feature name
        data = prepare_data(x, y, self.feature_names, self.output)

        # prepare testing data for all hypotheses
        for sens in self.sens_features:
            tree = self.trained_trees[sens]
            self.contexts[sens] \
                = tc.find_clusters_cat(tree, data, self.feature_info,
                                       sens, self.expl, self.output,
                                       prune_insignificant)

        # compute p-values and confidence intervals with FDR correction
        self.stats = multitest.compute_stats(self.contexts, approx, fdr)

    def report(self, filename,
               sort_by=displ.SORT_BY_EFFECT,
               filter_by=displ.FILTER_BETTER_THAN_ANCESTORS, encoders=None):
        """
        Output a FairTest bug report for each sensitive feature

        Parameters
        ----------
        filename :
            file to output the report to

        sort_by :
            method used to sort bugs in each report

        filter_by :
            method used to filter bugs in each report

        encoders :
            scikit data encoders used to encode categorical features (for pretty
            printing)
        """

        # TODO validate input, output reports to files instead of stdout

        for sens in self.sens_features:
            stats = self.stats[sens]
            clusters = self.contexts[sens]
            displ.bug_report(clusters, stats, sens, self.expl, self.output, sort_by,
                             filter_by, encoders)


def measure_from_string(m_str, ci_level, topk):
    if m_str == "NMI" or m_str == "MI":
        return fm.NMI(ci_level=ci_level)
    elif m_str == "Corr":
        return fm.CORR(ci_level=ci_level)
    elif m_str == "Diff":
        return fm.DIFF(ci_level=ci_level)
    elif m_str == "Ratio":
        return fm.Ratio(ci_level=ci_level)
    elif m_str == "Reg":
        return fm.REGRESSION(ci_level=ci_level, topK=topk)
    elif m_str == "CondNMI":
        return fm.CondNMI(ci_level=ci_level)
    raise ValueError('Unknown fairness Metric {}'.format(m_str))


def get_measure(sens, out, ci_level, topk, expl):
    if expl:
        assert sens.arity and out.arity
        return fm.CondNMI(ci_level=ci_level)

    if out.num_labels > 1:
        assert sens.arity == 2
        return fm.REGRESSION(ci_level=ci_level, topK=topk)
    elif sens.arity and out.arity:
        return fm.NMI(ci_level=ci_level)
    else:
        assert not sens.arity >= 2
        assert not out.arity >= 2
        return fm.CORR(ci_level=ci_level)


def prepare_data(x, y, feature_names, output):
    x = pd.DataFrame(np.array(x))
    x.columns = feature_names
    y = pd.DataFrame(np.array(y))
    y.columns = output.names

    #
    # Concatenate things for now. In the future we might look into keeping
    # x and y separate, so that we can handle sparse y
    #
    data = pd.concat([x, y], axis=1)
    return data


class Feature:
    """
    Class holding information about user features
    """

    # types of user features
    TYPES = ['context', 'sens', 'expl']

    def __init__(self, name, ftype, arity=None):
        assert ftype in Feature.TYPES
        self.name = name
        self.ftype = ftype
        self.arity = arity

    def __repr__(self):
        return "%s(name=%s, type=%s, arity=%s)" \
               % (self.__class__.__name__, self.name, self.ftype, self.arity)


class Target:
    """
    Class holding information about the target feature(s)
    """
    def __init__(self, names, arity=None):
        self.names = names
        self.num_labels = len(names)
        self.arity = arity