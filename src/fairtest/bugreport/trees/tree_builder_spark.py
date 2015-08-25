
from fairtest.bugreport.trees import sparkTree
import fairtest.bugreport.statistics.fairness_measures as fm
from fairtest.bugreport.trees.categorical_tree import ScoreParams as SP

from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SQLContext

MAX_BINS = 10
TARGET = '__TARGET__'
SENS = '__SENS__'


#
# Prepare the data for decision-tree building. Encode the sensitive feature and
# output feature as a single feature.
#
# @args data    the dataset
#
def prepare_data(data):
    data_copy = data.data_train.copy()

    # get the dimensions of the OUTPUT x SENSITIVE contingency table
    target_dim = (len(data.encoders[data.OUT].classes_), len(data.encoders[data.SENS].classes_))

    # add a new target feature encoding the values in the contingency table
    data_copy.insert(0, TARGET, target_dim[1]*data_copy[data.OUT] + data_copy[data.SENS])
    data_copy = data_copy.drop([data.SENS, data.OUT], axis=1)

    return data_copy, target_dim


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
def train_tree(sc, data, max_depth=5, min_leaf_size=100, measure=fm.NMI(ci_level=0.95), agg=SP.AVG):
    encoders = data.encoders
    (train_data, target_dim) = prepare_data(data)

    assert isinstance(measure, fm.NMI)
    assert agg == SP.AVG or agg == SP.MAX

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
    params['measure'] = 'MUTUAL_INFO'
    params['agg'] = 'AVG' if agg == SP.AVG else 'MAX'
    params['conf'] = 0 if not measure.ci_level else measure.ci_level
    params['categoricalFeaturesInfo'] = mapping
    params['maxDepth'] = max_depth
    params['minInstancesPerNode'] = min_leaf_size
    params['maxBins'] = max(max_arity, MAX_BINS)
    
    # call the Scala method
    tree = sparkTree.DiscriminationTree.findBias(**params)
    return tree, measure
