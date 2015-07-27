from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.common import callMLlibFunc, inherit_doc, JavaModelWrapper
from pyspark.mllib.tree import DecisionTreeModel

#
# Call Scala methods for building decision trees through py4j
#
class DiscriminationTree(object):
    @classmethod
    def _findBias(cls, data, type, numClasses, dimOut, dimSens, measure, agg, conf, features, maxDepth, maxBins,
               minInstancesPerNode=1, minInfoGain=0.0):
        first = data.first()
        assert isinstance(first, LabeledPoint), "the data should be RDD of LabeledPoint"
        model = callMLlibFunc("trainDecisionTreeBiasDiscovery", data, type, numClasses, dimOut,
                              dimSens, measure, agg, conf, features, maxDepth, maxBins, minInstancesPerNode, minInfoGain)
        return DecisionTreeModel(model)
        
        
    @classmethod
    def findBias(cls, data, numClasses, dimOut, dimSens, measure='MUTUAL_INFO', agg='WEIGHTED_AVG', conf=0.0,
                categoricalFeaturesInfo={}, maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0):
        
        return cls._findBias(data, "classification", numClasses, dimOut, dimSens, measure, agg, conf,
                          categoricalFeaturesInfo, maxDepth, maxBins, minInstancesPerNode, minInfoGain)