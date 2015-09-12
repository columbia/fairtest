'''
Script for testing the spark implementation of Fairtest
'''

import fairtest.bugreport.core.dataset as dataset
import fairtest.bugreport.trees.tree_builder_spark as tree_builder
import fairtest.bugreport.trees.categorical_tree as cat_tree
import fairtest.bugreport.clustering.tree_clustering as tc
import fairtest.bugreport.clustering.display_clusters as dc
from fairtest.bugreport.trees.categorical_tree import ScoreParams as SP


#
# Load a dataset and build a decision-tree to detect biases
#
# @args sc The SparkContext
#
def build_tree(sc):
    RANDOM_SEED = 0

    data = dataset.Dataset()
    data.load_data_csv('../data/adult/adult.csv')
    data.drop_feature('fnlwgt')

    data.set_sens_feature('sex', feature_type='cat')
    data.set_output_feature('predicted-income', feature_type='cat')

    data.encode_data(binary=False)
    data.train_test_split(train_size=0.25, seed=RANDOM_SEED)

    print '-- Data loaded and encoded --'

    import imp
    imp.reload(tree_builder)

    tree, measure = tree_builder.train_tree(sc, data, max_depth=5,
                                            min_leaf_size=100, agg=SP.AVG)
    print '-- {} --'.format(tree)
    root = tree._java_model.topNode()
    return root, data, measure


#
# Construct the clusters defined by the tree structure
#
# @args root       The root of the Spark tree
# @args data       The dataset object
# @args train_set  If True, uses the training set of the dataset rather than
#                   the test set.
#
def find_clusters(root, data, measure, train_set=False):
    clusters, pretty_tree = tc.find_clusters_spark(root, data,
                                                   measure, train_set)
    return clusters, pretty_tree


#
# Print the clusters
#
# @args clusters   The list of clusters
# @args sort_by    How to sort the clusters ('effect' or 'sig')
#
def print_clusters(clusters, sort_by=dc.SORT_BY_EFFECT):
    dc.bug_report(clusters, sort_by=sort_by,
                  node_filter=dc.FILTER_ALL, approx=True, fdr=0.05)


#
# Print the tree
#
# @args pretty_tree    The ete2 tree object to print
# @args outfile        The path to the output file
#
def print_tree(pretty_tree, outfile):
    cat_tree.print_tree(pretty_tree, outfile, encoders=None, is_spark=True)
