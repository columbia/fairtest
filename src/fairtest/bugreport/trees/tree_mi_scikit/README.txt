This is a modified decision-tree module for bias detection.

There is an added Mutual_Info class extending ClassificationCriterion, that can be used as a splitting criterion for the DecisionTreeClassifier class.

All the modifications appear in the _tree.pyx file.

To compile:

- run "cython _tree.pyx" to produce the _tree.c file
- run "python setup.py build_ext --inplace" to generate the _tree.so file