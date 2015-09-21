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
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()

from os import path
from copy import copy

import pandas as pd
import numpy as np
import sys


class Experiment:
    """
    A FairTest experiment
    """
    def __init__(self, data, protected, output, expl=None, measures={},
                 train_size=0.5, topk=60, ci_level=0.95, random_state=None):
        """
        Initialize a FairTest experiment

        Parameters
        ----------
        data :
            array-like, shape = [n_samples, n_features+n_outputs]

        protected :
            list of columns in ``data`` containing protected features

        output :
            list of columns in ``data`` containing algorithmic outputs. If
            multiple columns are provided, the values in each column are viewed
            as binary indicator variable for the presence of a particular label
            in the output.

        expl :
            column in ``data`` containing an explanatory feature. If provided,
            the association between a sensitive feature and output is computed
            conditioned on the value of this feature

        measures :
            a map from a sensitive_feature to a fairness metric. If no metric
            is specified for some sensitive feature, a default metric is
            selected

        topk :
            maximum number of labels to report when discovering association bugs
            over a large output space

        ci_level :
            confidence level for fairness confidence intervals

        random_state :
            seed used for the random number generator
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
        if random_state:
            self.random_state = random_state
        else:
            self.random_state = 0
        ro.r('set.seed({})'.format(self.random_state))

        data = pd.DataFrame(data)

        self.encoders = {}
        for column in data.columns:
            if data.dtypes[column] == np.object:
                self.encoders[column] = preprocessing.LabelEncoder()
                data[column] = self.encoders[column].fit_transform(data[column])

        self.feature_info = {}
        for col in data.columns.drop(output):
            ftype = 'sens' if col in protected \
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
            target_arity = len(self.encoders[output].classes_)
        except:
            target_arity = None
        self.output = Target(np.asarray([output]).flatten(), arity=target_arity)

        self.train_set, self.test_set = \
            cross_validation.train_test_split(data, train_size=train_size,
                                              random_state=random_state)

    def train(self, max_depth=5, min_leaf_size=100, score_aggregation="avg", max_bins=10):
        """
        Form hypotheses about discrimination contexts for each protected feature

        Parameters
        ----------
        max_depth :
            maximum tree generation depth. Maximum number of features that
            define a discrimination context

        min_min_leaf_size :
            minimum size of a tree leaf. Minimum size of a discrimination
            context (on the training set)

        score_aggregation :
            method used to compute the score of a split. If "avg", averages the
            association scores of all the sub-populations generated by the
            split. If "weighted_avg", computes the average of the child scores
            weighted by the size of the sub-populations. If "max", returns
            the maximal association scores of all generated sub-populations.

        max_bins :
            maximum number of bins used for finding splits on continuous
            features
        """
        # TODO validate input

        # create a Pandas DataFrame with columns index by feature name
        data = self.train_set

        self.train_params = {'max_depth': max_depth,
                             'min_leaf_size': min_leaf_size,
                             'agg_type': score_aggregation,
                             'max_bins': max_bins}

        # find discrimination contexts for each sensitive feature
        for sens in self.sens_features:
            # print 'TRAINING WITH SENSITIVE FEATURE {} ...'.format(sens)

            if sens not in self.measures:
                # get a default measure
                self.measures[sens] = get_measure(self.feature_info[sens],
                                                  self.output, self.ci_level,
                                                  self.topk, self.expl)
            elif sens in self.measures and isinstance(self.measures[sens], str):
                # get specified measure
                self.measures[sens] = measure_from_string(self.measures[sens],
                                                          self.ci_level,
                                                          self.topk)

            tree = builder.train_tree(data, self.feature_info, sens, self.expl,
                                      self.output, copy(self.measures[sens]),
                                      max_depth, min_leaf_size,
                                      score_aggregation, max_bins)
            self.trained_trees[sens] = tree
            # print ""

    def test(self, prune_insignificant=True, approx_stats=True, fdr=0.05):
        """
        Compute effect sizes and p-values for the discrimination contexts
        discovered on the training set.

        Parameters
        ----------
        prune_insignificant :
            if ``True``, prune discrimination contexts for which the effect size
            confidence interval on the training set contains 0.

        approx_stats :
            if ``True``, approximate asymptotically correct methods are used to
            generate p-values and confidence intervals. Otherwise, confidence
            intervals are generated with bootstrapping techniques and p-values
            via Monte-Carlo permutation tests.

        fdr :
            false-discovery rate to guarantee
        """
        # TODO validate input, check that tree was trained

        self.test_params = {'prune_insignificant': prune_insignificant,
                            'approx': approx_stats,
                            'fdr': fdr}

        # create a Pandas DataFrame with columns index by feature name
        data = self.test_set

        num_contexts = 0
        # prepare testing data for all hypotheses
        for sens in self.sens_features:
            tree = self.trained_trees[sens]
            self.contexts[sens] \
                = tc.find_clusters_cat(tree, data, self.feature_info,
                                       sens, self.expl, self.output,
                                       prune_insignificant)
            num_contexts += len(self.contexts[sens])

        # print 'RUNNING {} HYPOTHESIS TESTS...'.format(num_contexts)
        # compute p-values and confidence intervals with FDR correction
        np.random.seed(self.random_state)
        self.stats = multitest.compute_all_stats(self.contexts, approx_stats, fdr)

    def report(self, dataname, output_dir=None,
               sort_by=displ.SORT_BY_EFFECT,
               filter_by=displ.FILTER_BETTER_THAN_ANCESTORS):
        """
        Output a FairTest bug report for each protected feature

        Parameters
        ----------
        dataname :
            name of the dataset used in the experiment

        output_dir :
            directory to which bug reports shall be output.
            Should be an absolute path. Default is None and
            report are sent to stdout.

        sort_by :
            method used to sort bugs in each report. Either "effect" to sort by
            effect size or "significance" to sort by p-values

        filter_by :
            method used to filter bugs in each report. Bugs that are not
            statistically significant for the provided false discovery rate are
            filtered out automatically. Setting this parameter to
            "better_than_ancestors" additionally filters out a contexts if it
            does not exhibit a larger association than the larger contexts that
            it is part of.
        """

        self.display_params = {'sort_by': sort_by,
                               'filter_by': filter_by
                               }

        # TODO validate inputs
        if not output_dir:
            output_stream = sys.stdout
        elif not path.isdir(output_dir):
            raise IOError("Directory \"%s\" does not exist" % output_dir)
        else:
            try:
                filename = path.join(output_dir, "report_" + dataname + ".txt")
                output_stream = open(filename, "w+")
            except IOError:
                print "Error: Cannot open file: %s for writting" % (filename)

        train_size = len(self.train_set)
        test_size = len(self.test_set)
        sensitive = self.sens_features
        contextual = [name for (name, f) in self.feature_info.items()
                      if f.ftype == 'context']

        displ.print_report_info(dataname, train_size, test_size, sensitive,
                                contextual, self.expl, self.output.names,
                                self.train_params, self.test_params,
                                self.display_params, output_stream)

        for sens in self.sens_features:
            print >> output_stream, 'Report of associations on Si = {}:'.format(sens)
            print >> output_stream, 'Association metric: {}'.format(self.measures[sens])
            print >> output_stream
            stats = self.stats[sens]
            clusters = self.contexts[sens]
            np.random.seed(self.random_state)
            # dirty nasty hack for the benchmark
            txt = displ.bug_report(clusters, stats, sens, self.expl, self.output,
                             output_stream, sort_by, filter_by, self.encoders)
            if len(self.sens_features) == 1:
                return txt


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
        assert not sens.arity > 2
        assert not out.arity > 2
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
