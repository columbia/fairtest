"""
Implement the Fairtest API
"""
from fairtest.bugreport.core import dataset
from fairtest.bugreport.trees import tree_builder
from fairtest.bugreport.trees import categorical_tree
from fairtest.bugreport.clustering import tree_clustering
from fairtest.bugreport.clustering import display_clusters
from fairtest.bugreport.statistics import fairness_measures

import os
from random import randint


class Fairtest:
    """
    The main class
    """
    def __init__(self):
        """
        Constructor
        """
        self.types = ()
        self.names = ()

        self.output = ''
        self.sensitive = ''
        self.output_type = ''
        self.sensitive_type = ''

        self.measure = fairness_measures.NMI(0.95)
        self.random_seed = 0

        self.display_params = {'approx': True, 'fdr': 0.05}

        self.tree_params = {'max_depth': 5, 'min_leaf_size': 100,
                            'agg_type': categorical_tree.ScoreParams.AVG}

        self.data = dataset.Dataset()
        print "Class FairTest initialized"


    def set_attribute_types(self, names, types):
        """
        setting attribute names and types

        Notes:
        ------
        tuples are ordered
        """
        assert len(names) == len(types)

        self.names = names
        self.types = types


    def set_output_attribute(self, name):
        """
        setting the name of the output attribute
        """
        self.output = name

        #TODO: exception
        self.output_type = self.types[self.names.index(name)]


    def set_sensitive_attribute(self, name):
        """
        setting the name of the sensitive attribute
        """
        self.sensitive = name
        #TODO: exception
        self.sensitive_type = self.types[self.names.index(name)]


    def set_metric(self, name, ci_level=0.95, top_k=100):
        """
        setting desired measure

        Notes:
        ------
        Default is NMI with ci_level=0.95
        """
        if name == 'NMI':
            self.measure = fairness_measures.NMI(ci_level)
        elif name == 'CORR':
            self.measure = fairness_measures.CORR(ci_level)
        elif name == 'REGRESSION':
            self.measure = fairness_measures.REGRESSION(ci_level, top_k)
        else:
            raise ValueError("measure must be NMI, CORR, or REGRESSION")


    def set_tree_params(self, tree_params):
        """
        function setting tree params

        Notes:
        ------
        Default is {'max_depth': 5, 'min_leaf_size': 100,
                    'agg_type': categorical_tree.ScoreParams.AVG}
        """
        self.tree_params = tree_params
        self.tree_params['agg_type'] = categorical_tree.ScoreParams.AVG


    def set_display_params(self, display_params):
        """
        function setting display params

        Notes:
        ------
        Default is {'approx': True, 'fdr': 0.05}
        """
        self.display_params = display_params


    def bug_report(self, x, y):
        """
        prints the bug report
        """
        print "Creating bug report"

        # try to create temporary csv file
        temp_file = "/tmp/tempfile" + str(randint(0, 99999)) + ".csv"

        # TODO: Error checking
        self._flush(temp_file, x, y)

        # try to load file in memory and remove temp file
        # TODO: Error checking
        self.data.load_data_csv(temp_file)
        os.remove(temp_file)

        self.data.set_sens_feature(self.sensitive,
                                   feature_type=self.sensitive_type)

        self.data.set_output_feature(self.output,
                                     feature_type=self.output_type)

        self.data.encode_data(binary=False)

        self.data.train_test_split(train_size=0.25, seed=self.random_seed)
        self.data.data_train.tail()

        tree = tree_builder.train_tree(self.data, measure=self.measure,
                                       **self.tree_params)
        clusters = tree_clustering.find_clusters_cat(tree, self.data)

        self.display_params['sort_by'] = display_clusters.SORT_BY_EFFECT
        self.display_params['node_filter'] = \
                display_clusters.FILTER_BETTER_THAN_ANCESTORS

        display_clusters.print_report_info(self.data, self.measure,
                                  self.tree_params, self.display_params)
        display_clusters.bug_report(clusters, **self.display_params)


    def _flush(self, name, x, y):
        """
        helper function to flush matrixes x and y into
        a csv format appropriate for loading from
        currrent version of Fairtest
        """
        assert len(x) == len(y)

        f = open(name, "w+")

        # write header
        for attr in self.names[:-1]:
            print >> f, "%s," % attr,
        print >> f, "%s" % self.names[-1]

        # write content
        for i in range(0, len(x)):
            for attr in x[i]:
                print >> f, "%s," % attr,
            print >> f, "%s" % y[i][0]

        # finalize flushing
        f.close()
