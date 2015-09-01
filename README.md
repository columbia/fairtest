#FairTest


SPARK
=====

Build Spark (go grab a coffee):

    build/mvn -DskipTests clean package

Compile only mllib (if modified):

    build/mvn compile -DskipTests -pl mllib 

Pyspark
-------
If necessary, set the python path to an installed python distribution which has 
numpy, pandas, etc installed:

    export PYSPARK_PYTHON=/usr/bin/python

Tell pyspark to look for freshly compiled classes (custom mllib):

    export SPARK_PREPEND_CLASSES=1

Compress the sources

    zip -r src/fairtest.zip src/fairtest

Launch pyspark and attach the fairtest code

    ./bin/pyspark --py-files ../src/fairtest.zip

In the pyspark shell:

    import fairtest.spark.test_spark as test_spark
    root, data, measure = test_spark.build_tree(sc)
    clusters, pretty_tree = test_spark.find_clusters(root, data, measure, train_set=False)
    test_spark.print_clusters(clusters)
    test_spark.print_tree(pretty_tree, '../graphs/tree_spark.pdf')


Non-SPARK
=========

Launch IPython:

    cd src/
    ipython notebook --pylab inline


Code Organisation
=================

* data: Demo datasets

* src/apps: Demo apps

* src/fairtest/tests: Test for spark and ipython implementations

* src/fairtest/bugreport/clustering:    Package for extracting clusters and displaying them

* src/fairtest/bugreport/core:          Package for representing a dataset

* src/fairtest/bugreport/statistics:    Package for statistical measures

* src/fairtest/bugreport/trees:         Package for building trees
