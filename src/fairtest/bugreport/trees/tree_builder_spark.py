from fairtest.bugreport.core import dataset as dataset
from fairtest.bugreport.trees import sparkTree
from fairtest.bugreport.trees import tree_builder 

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.sql import SQLContext

MAX_BINS = 10

#
# Build a decision tree with Spark
#
# @args sc              The SparkContext
# @args data            The dataset
# @args max_depth       The maximum depth of the tree
# @args min_leaf_size   The minimum number of samples in a leaf
# @args measure         The dependency measure to use
# @args agg             The child-score aggregation method to use
# @args conf            Confidence level for CIs
#
def train_tree(sc, data, max_depth=5, min_leaf_size=100, measure='MUTUAL_INFO', agg='WEIGHTED_AVG', conf=0.0):
    encoders = data.encoders
    (train_data, target_dim) = tree_builder.prepare_data(data)
    
    # convert pandas dataframe to spark RDD
    sqlContext = SQLContext(sc)
    spark_df = sqlContext.createDataFrame(train_data)
    spark_rdd = spark_df.rdd
    spark_rdd = spark_rdd.map(lambda row: LabeledPoint(row[0], row[1:]))
    
    # create a mapping {feature index : feature arity} for categorical features
    max_arity = 0
    mapping = {}
    for i in range(1,len(train_data.columns)):
        col = train_data.columns[i]
        if col in encoders:
            mapping[i-1] = len(encoders[col].classes_)
            max_arity = max(max_arity, len(encoders[col].classes_))
    
    # build all the parameters        
    params = {}
    params['data'] = spark_rdd
    params['numClasses'] = target_dim[0]*target_dim[1]
    params['dimOut'] = target_dim[0]
    params['dimSens'] = target_dim[1]
    params['measure'] = measure
    params['agg'] = agg
    params['conf'] = conf
    params['categoricalFeaturesInfo'] = mapping
    params['maxDepth'] = max_depth
    params['minInstancesPerNode'] = min_leaf_size
    params['maxBins'] = max(max_arity, MAX_BINS)
    
    # call the Scala method
    tree = sparkTree.DiscriminationTree.findBias(**params)
    return tree