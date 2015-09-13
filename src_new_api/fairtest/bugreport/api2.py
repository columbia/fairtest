"""
Implement the Fairtest API
"""
from fairtest.bugreport.trees import tree_builder as builder
from fairtest.bugreport.clustering import tree_clustering as tc
from fairtest.bugreport.clustering import display_clusters as displ
from fairtest.bugreport.statistics import fairness_measures as fm
from fairtest.bugreport.statistics import multiple_testing as multitest
from sklearn import preprocessing as preprocessing
from sklearn import cross_validation as cross_validation

import pandas as pd
import numpy as np


class Fairtest:
    """
    The main class
    """
    def __init__(self, data, sens, target, expl=None, measures={},
                 train_size=0.5, topk=50, ci_level=0.95, random_state=None):
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
        self.topk = topk
        self.ci_level = ci_level
        self.measures = measures
        self.trained_trees = {}
        self.contexts = {}
        self.stats = {}
        self.train_params = {}
        self.test_params = {}
        self.display_params = {}

        data = pd.DataFrame(data)

        self.encoders = {}
        for column in data.columns:
            if data.dtypes[column] == np.object:
                self.encoders[column] = preprocessing.LabelEncoder()
                data[column] = self.encoders[column].fit_transform(data[column])

        self.feature_info = {}
        for col in data.columns.drop(target):
            ftype = 'sens' if col in sens \
                else 'expl' if col in expl \
                else 'context'
            arity = None if col not in self.encoders \
                else len(self.encoders[col].classes_)
            self.feature_info[col] = Feature(ftype, arity)

        # get the names of the sensitive features
        self.sens_features = [name for (name, f) in self.feature_info.items()
                              if f.ftype == 'sens']

        # get the name of the explanatory feature (if any)
        expl_list = [name for (name, f) in self.feature_info.items()
                     if f.ftype == 'expl']
        self.expl = expl_list[0] if expl_list else None

        try:
            target_arity = len(self.encoders[target].classes_)
        except:
            target_arity = None
        self.output = Target(np.asarray([target]).flatten(), arity=target_arity)

        self.train_set, self.test_set = \
            cross_validation.train_test_split(data, train_size=train_size,
                                              random_state=random_state)

    def train(self, max_depth=5, min_leaf_size=100, agg_type="AVG", max_bins=10):
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
        data = self.train_set

        self.train_params = {'max_depth': max_depth,
                             'min_leaf_size': min_leaf_size,
                             'agg_type': agg_type,
                             'max_bins': max_bins}

        # find discrimination contexts for each sensitive feature
        for sens in self.sens_features:
            print 'TRAINING WITH SENSITIVE FEATURE {}'.format(sens)

            if sens in self.measures:
                measure = self.measures[sens]
                # TODO validate the choice of measure
            else:
                # get a default measure
                measure = get_measure(self.feature_info[sens], self.output,
                                      self.ci_level, self.topk, self.expl)

            tree = builder.train_tree(data, self.feature_info, sens,
                                      self.expl, self.output, measure,
                                      max_depth, min_leaf_size,
                                      agg_type, max_bins)
            self.trained_trees[sens] = tree

    def test(self, prune_insignificant=True, approx=True, fdr=0.05):
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

        self.test_params = {'prune_insignificant': prune_insignificant,
                            'approx': approx,
                            'fdr': fdr}

        # create a Pandas DataFrame with columns index by feature name
        data = self.test_set

        # prepare testing data for all hypotheses
        for sens in self.sens_features:
            tree = self.trained_trees[sens]
            self.contexts[sens] \
                = tc.find_clusters_cat(tree, data, self.feature_info,
                                       sens, self.expl, self.output,
                                       prune_insignificant)

        # compute p-values and confidence intervals with FDR correction
        self.stats = multitest.compute_all_stats(self.contexts, approx, fdr)

    def report(self, dataname, filename,
               sort_by=displ.SORT_BY_EFFECT,
               filter_by=displ.FILTER_BETTER_THAN_ANCESTORS):
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

        self.display_params = {'sort_by': sort_by,
                               'filter_by': filter_by
                               }

        # TODO validate input, output reports to files instead of stdout

        train_size = len(self.train_set)
        test_size = len(self.test_set)
        sensitive = self.sens_features
        contextual = [name for (name, f) in self.feature_info.items()
                      if f.ftype == 'context']

        displ.print_report_info(dataname, train_size, test_size, sensitive,
                                contextual, self.expl, self.output.names,
                                self.train_params, self.test_params,
                                self.display_params)

        for sens in self.sens_features:
            stats = self.stats[sens]
            clusters = self.contexts[sens]
            displ.bug_report(clusters, stats, sens, self.expl, self.output,
                             sort_by, filter_by, self.encoders)


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

class Feature:
    """
    Class holding information about user features
    """

    # types of user features
    TYPES = ['context', 'sens', 'expl']

    def __init__(self, ftype, arity=None):
        assert ftype in Feature.TYPES
        self.ftype = ftype
        self.arity = arity

    def __repr__(self):
        return "%s(type=%s, arity=%s)" \
               % (self.__class__.__name__, self.ftype, self.arity)


class Target:
    """
    Class holding information about the target feature(s)
    """
    def __init__(self, names, arity=None):
        self.names = names
        self.num_labels = len(names)
        self.arity = arity