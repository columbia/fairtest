#FairTest

===============
   SPARK   
===============

Build Spark (go grab a coffee):
    >> build/mvn -DskipTests clean package

Compile only mllib (if modified):
    >> build/mvn compile -DskipTests -pl mllib 

Pyspark
=======
If necessary, set the python path to an installed python distribution which has 
numpy, pandas, etc installed:
    >> export PYSPARK_PYTHON=/usr/bin/python

Tell pyspark to look for freshly compiled classes (custom mllib):
    >> export SPARK_PREPEND_CLASSES=1

Launch pyspark and attach the fairtest code
    >> ./bin/pyspark --py-files ../src/fairtest.zip

Compress 'src/fairtest/' to 'src/fairtest.zip'

In the pyspark shell:
    >> import fairtest.spark.test_spark as test_spark
    >> root, data = test_spark.build_tree(sc)
    >> clusters, pretty_tree = test_spark.find_clusters(root, data, train_set=False)
    >> test_spark.print_clusters(clusters, sort_by='effect')
    >> test_spark.print_tree(pretty_tree, '../graphs/tree_spark.pdf')
    

===================
||   Non-SPARK   ||
===================

Launch IPython:

>> cd src/
>> ipython notebook



============================
||    Code Organisation   ||
============================

src/fairtest/spark                  Test script for the Spark implementation

src/fairtest/bugreport/clustering   Module for extracting clusters from a tree
                                    structure, and displaying them
                                    
src/fairtest/bugreport/core         Contains a class for representing a dataset

src/fairtest/bugreport/statistics   Code for statistical measures and tests

src/fairtest/bugreport/trees        Code for building trees
