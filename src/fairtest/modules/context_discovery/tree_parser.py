"""
Parser and Extractor for tree contexts
"""
from fairtest.modules.metrics import Metric
import pandas as pd
import numpy as np
from copy import deepcopy, copy


class Context(object):
    """
    Representation of an association context.

    Attributes
    ----------
    num :
        the contexts's number

    path :
        the predicate path leading from the root to the context

    isleaf :
        if the context is a tree leaf

    isroot :
        if the context is the tree root

    parent :
        the parent of this context

    data :
        the data for this context

    size :
        the context size

    metric :
        the metric associated with this context

    data :
        any additional data required for fairness metrics
    """
    def __init__(self, num, path, isleaf, isroot, parent,
                 data, size, metric=None, additional_data=None):
        self.num = num
        self.path = path
        self.isleaf = isleaf
        self.isroot = isroot
        self.parent = parent
        self.children = []
        self.size = size
        self.data = data
        self.metric = metric
        self.additional_data = additional_data


class Bound(object):
    """
    Representation of a bound for a continuous feature
    """
    def __init__(self):
        self.lower = -float('inf')
        self.upper = float('inf')

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        ret = '(' + str(self.lower) + ', ' + str(self.upper)
        if self.upper == float('inf'):
            ret += ')'
        else:
            ret += ']'
        return ret


def update_cont_path(feature_path, feature, lower_bound=None, upper_bound=None):
    """
    Update a bound for a continuous feature on the predicate path

    Parameters
    ----------
    feature_path:
        the feature path to update

    feature :
        the feature to consider

    lower_bound :
        the new lower bound

    upper_bound :
        the new upper bound
    """
    bound = feature_path.get(feature, Bound())
    if lower_bound:
        bound.lower = lower_bound
    else:
        bound.upper = upper_bound

    feature_path[feature] = bound


def find_contexts(tree, data, features_info, sens, expl, output,
                  prune_insignificant=False):
    """
    Traverse a tree and output contexts for each node.

    Parameters
    ----------
    tree :
        the tree to traverse

    data :
        the dataset

    features_info :
        information for contextual features

    sens :
        the name of the sensitive feature

    expl :
        the name of the explanatory feature

    output :
        the target feature

    prune_insignificant :
        whether contexts should be pruned if they show no significant
        association on the training set

    Returns
    -------
    contexts :
        The list of contexts uncovered
    """
    # list of contexts
    contexts = []
    #targets = data.columns[-output.num_labels:].tolist()
    targets = output.names.tolist()

    # assign an id to each node
    node_id = 0
    for tree_node in tree.traverse("levelorder"):
        tree_node.add_features(id=node_id)
        node_id += 1

    metric_type = tree.metric.dataType

    def bfs(node, parent, data_node, feature_path):
        """
        Simple BFS to traverse the tree

        Parameters
        ----------
        node :
            the current node

        parent :
            the parent node

        data_node :
            the sub-dataset rooted at this node

        feature_path :
            The predicate path from the root to this node
        """
        is_root = node.is_root()
        is_leaf = node.is_leaf()

        # current node
        if not is_root:
            feature = node.feature

            # check type of feature split
            if node.feature_type == 'continuous':
                threshold = node.threshold

                # update the bound on the continuous feature
                if node.is_left:
                    update_cont_path(feature_path,
                                     feature, upper_bound=threshold)
                    data_node = data_node[data_node[feature] <= threshold]
                else:
                    update_cont_path(feature_path,
                                     feature, lower_bound=threshold)
                    data_node = data_node[data_node[feature] > threshold]
            else:
                # categorical split
                category = node.category
                feature_path[feature] = category
                data_node = data_node[data_node[feature] == category]

        if metric_type == Metric.DATATYPE_CT:
            # categorical data

            if not expl:
                # create an empty contingency table
                ct = pd.DataFrame(0, index=range(output.arity),
                                  columns=range(features_info[sens].arity))
                # fill in available values
                ct = ct.add(pd.crosstab(np.array(data_node[targets[0]]),
                                        np.array(data_node[sens])),
                            fill_value=0)
                data = ct
            else:
                dim_expl = features_info[expl].arity
                cts = dim_expl * \
                      [pd.DataFrame(0, index=range(output.arity),
                                    columns=range(features_info[sens].arity))]

                for (key, group) in data_node.groupby(expl):
                    cts[key] = cts[key].add(
                        pd.crosstab(np.array(group[targets[0]]),
                                    np.array(group[sens])), fill_value=0)

                data = [ct.values for ct in cts]

            additional_data = None
            size = len(data_node)

        elif metric_type == Metric.DATATYPE_CORR:
            # continuous data
            data = data_node[[targets[0], sens]]
            size = len(data_node)
            additional_data = None
        else:
            # regression metric
            # keep all the data
            data = data_node[targets + [sens]]
            size = len(data_node)
            additional_data = {'data_node': data_node}

        # build a context class and store it in the list
        metric = copy(node.metric)

        ancestor_ptr = parent
        # prune non-significant contexts
        if (is_root or metric.abs_effect() > 0) or not prune_insignificant:
            clstr = Context(node.id, feature_path, is_leaf, is_root, parent,
                            data, size, metric, additional_data)
            if parent:
                parent.children.append(clstr)
            contexts.append(clstr)
            ancestor_ptr = clstr

        # recurse in children
        for child in node.get_children():
            bfs(child, ancestor_ptr, data_node, deepcopy(feature_path))

    # start bfs from the root with the full dataset and an empty path
    bfs(tree, None, data, {})
    return contexts
