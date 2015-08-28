import fairtest.bugreport.trees.categorical_tree as cat_tree
import fairtest.bugreport.statistics.fairness_measures as fm


#
# Build a decision tree for bias detection
#
# @args data            The dataset
# @args max_depth       The maximum depth of the tree
# @args min_leaf_size   The minimum number of samples in a leaf
# @args measure         The dependency measure to use
# @args agg_type        The child-score aggregation method to use
# @args conf            Confidence level for CIs
#
def train_tree(data, max_depth=5, min_leaf_size=100, measure=fm.NMI(ci_level=0.95),
               agg_type=cat_tree.ScoreParams.WEIGHTED_AVG):
    # prepare the function call parameters
    params = dict()
    params['dataset'] = data
    # params['dim'] = target_dim
    params['categorical'] = data.encoders.keys()
    params['max_depth'] = max_depth
    params['min_leaf_size'] = min_leaf_size
    params['measure'] = measure
    params['agg_type'] = agg_type
    
    # build a categorical decision tree
    tree = cat_tree.build_tree(**params)
    return tree
