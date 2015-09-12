"""
Module to build tree
"""
import fairtest.bugreport.trees.categorical_tree as cat_tree
import pandas as pd
import numpy as np


def train_tree(data, feature_info, sens, expl, output, measure, max_depth=5,
               min_leaf_size=100, agg_type='AVG', max_bins=10):
    """
    Build a decision tree for bias detection

    Parameters
    ----------


    Returns
    -------
    tree :
        The tree built
    """

    # build a categorical decision tree
    tree = cat_tree.build_tree(data, feature_info, sens, expl, output, measure,
                               max_depth, min_leaf_size, agg_type,
                               max_bins)

    return tree
