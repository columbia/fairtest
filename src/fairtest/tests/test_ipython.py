import fairtest.bugreport.core.dataset as dataset
import fairtest.bugreport.trees.tree_builder as tree_builder
import fairtest.bugreport.trees.categorical_tree as cat_tree
import fairtest.bugreport.clustering.tree_clustering as tree_clustering
import fairtest.bugreport.clustering.display_clusters as display
import fairtest.bugreport.statistics.fairness_measures as fm
import imp

RANDOM_SEED = 0
imp.reload(dataset)
data = dataset.Dataset()

# Choose a Dataset
#
# # Adult dataset (GENDER x INCOME)
#
# data.load_data_csv('../data/adult/adult.csv')
# data.drop_feature('fnlwgt')
# data.set_sens_feature('sex', feature_type='cat')
# data.set_output_feature('predicted-income', feature_type='cat')
#
# # Medical dataset (Age x Prediction Correctness)
# '''
# data.load_data_csv('../data/medical/predictions2.csv')
# outputs = ['Correct', 'FP', 'FN']
# data.original_data['Prediction'] = map(lambda l: outputs[l.index(1)],
# zip(data.original_data['Pred_Correct'], data.original_data['Pred_FP'],
# data.original_data['Pred_FN']))
# data.drop_feature('Pred_Correct')
# data.drop_feature('Pred_FP')
# data.drop_feature('Pred_FN')
# data.drop_feature('Pred_TP')
# data.drop_feature('Pred_TN')
# data.set_sens_feature('Age', feature_type='cat')
# data.set_output_feature('Prediction', feature_type='cat')
# '''
#
# # Medical dataset (Age x False Positives)
# '''
# data.load_data_csv('../data/medical/predictions2.csv')
# data.drop_feature('Pred_Correct')
# data.drop_feature('Pred_FN')
# data.drop_feature('Pred_TP')
# data.drop_feature('Pred_TN')
# data.set_sens_feature('Age', feature_type='cat')
# data.set_output_feature('Pred_FP', feature_type='cat')
# '''
#
# # Medical dataset (Age x Average Predicton Error)
# '''
# data.load_data_csv('../data/medical/predictions_cont.csv')
# data.original_data['Error'] = abs(data.original_data['Error'])
# data.set_sens_feature('Age', feature_type='cont')
# data.set_output_feature('Error', feature_type='cont')
# '''
#
# # Berkeley dataset (Gender x Admission)
# '''
# data.load_data_csv('../data/berkeley/berkeley.csv')
# data.set_sens_feature('gender', feature_type='cat')
# data.set_output_feature('accepted', feature_type='cat')
# '''
#
# Staples dataset (Race x Price)

data.load_data_csv('../data/staples/staples.csv')
data.drop_feature('zipcode')
data.drop_feature('distance')
data.set_sens_feature('race', feature_type='cat')
data.set_output_feature('price', feature_type='cat')

## Recomender dataset (Gender x improvement)
#
#data.load_data_csv('../data/recommender/recommendations.csv')
#data.drop_feature('userID')
#data.drop_feature('zip')
#data.drop_feature('mode')
#data.drop_feature('median')
#data.drop_feature('improvement')
#data.drop_feature('movieType')
#data.set_sens_feature('gender', feature_type='cat')
#data.set_output_feature('improvement', feature_type='cat')
#
#
# # Image Labeling dataset (Race x Labels)
# '''
# data.load_data_csv('../data/images/overfeat.txt', separator='\t')
# data.set_sens_feature('Race', feature_type='cat')
# data.set_output_feature('Labels', feature_type='labeled')
# '''
#

data.encode_data(binary=False)
data.train_test_split(train_size=0.25, seed=RANDOM_SEED)
print 'Sample Size: {}'.format(len(data.original_data))
print 'Training Size: {}'.format(len(data.data_train))
print 'Testing Size: {}'.format(len(data.data_test))
data.data_train.tail()


imp.reload(cat_tree)
imp.reload(tree_builder)
imp.reload(fm)
measure = fm.NMI(ci_level=0.95)
#measure  = fm.CORR(ci_level=0.95)
#measure = fm.DIFF(ci_level=None)
#measure = fm.REGRESSION(ci_level=0.95, topK=10)

tree = tree_builder.train_tree(data, max_depth=5, min_leaf_size=100,
                               measure=measure, agg_type=cat_tree.ScoreParams.AVG)
imp.reload(tree_clustering)
clusters = tree_clustering.find_clusters_cat(tree, data)

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
ro.conversion.py2ri = numpy2ri
numpy2ri.activate()
R_LIBRARIES = ['survival', 'coin']
for lib in R_LIBRARIES:
    ro.r('library({})'.format(lib))

imp.reload(display)
imp.reload(fm)
node_filter=display.FILTER_ALL
#node_filter=display.FILTER_ROOT_ONLY
sort_by = display.SORT_BY_EFFECT
alt_measure = fm.NMI(ci_level=0.95)
#alt_measure = fm.CORR(ci_level=0.95)
#alt_measure = fm.REGRESSION(ci_level=0.95, topK=100)
#alt_measure = None

display.bug_report(clusters, columns=None, new_measure=alt_measure,
                   sort_by=sort_by, node_filter=node_filter, approx=False,
                   fdr=0.05)
