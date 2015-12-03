#FairTest

FairTest enables developers or auditing entities to *discover* and *test* for
*unwarranted associations* between an algorithm's outputs and certain user 
subpopulations identified by *protected* features.

FairTest works by learning a special *decision tree*, that splits a user
population into smaller subgroups in which the association between protected
features and algorithm outputs is maximized. FairTest supports and makes use 
of a variety of different fairness *metrics* each appropriate in a particular
situation. After finding these so-called *contexts of association*, FairTest 
uses statistical methods to assess their validity and strength. Finally, 
FairTest retains all statistically significant associations, ranks them by 
their strength, and reports them as *association bugs* to the user.

Installation
------------

FairTest is a `Python` application, developed and tested with `Python 2.7`.
FairTest uses `rpy2` python package that provides a python interface
for `R` programming language and requires `R` (version > 3.1) to be
installed. We provide a script to assist with the installation of FairTest.


* Run `make apt-dependencies` to install apt package dependencies and
  the latests version of `R` (for Ubuntu 12.04 and 14.04).

* Run `make pip-dependencies` to install python pip package dependencies.


* Run `make install` to install both apt and pip package dependencies.


Alternativelly, you can download an Ubuntu virtual machine with a complete,
up-to-date FairTest installation from
<a href="http://www.cs.columbia.edu/~vatlidak/UbuntuVM.tar.gz" title="FairTest VM">here</a>
and launch it with VMWare Player.

Quick Start
-----------

Different benchmark datasets in CSV format are located in `fairtest/data`. You
can use the `utils.prepare_data.data_from_csv()` function to load a dataset as
a `Pandas DataFrame`, the format expected by FairTest investigations:

```python
from fairtest.utils.prepare_data import data_from_csv

data = data_from_csv('fairtest/data/adult/adult.csv', to_drop=['fnlwgt'])
```

#### Testing
To test for associations between user income and race or gender, first create
the appropriate Fairtest `Investigation`:

```python
from fairtest.testing import Testing

SENS = ['gender', 'race']     # Protected features
TARGET = 'income'             # Output
EXPL = ''                     # Explanatory feature

inv = Testing(data, SENS, TARGET, EXPL)
```

After you instantiated all the `Investigations` you wish to perform, you can
`train` the guided decision tree, `test` the discovered association bugs (and
correct for multiple testing), and `report` the results:

```python
from fairtest.investigation import train, test, report

all_investigations = [inv]

train(all_investigations)
test(all_investigations)
report(all_investigations, 'adult', output_dir='temp/')
```

#### Discovery
`Discovery` investigations enable the search for potential associations over
a large output space, with no prior knowledge of which outputs to focus on.
An additional instance parameter `topk` specifies the maximum number of 
outputs that exhibit the strongest associations to consider:

```python
from fairtest.discovery import Discovery

SENS = [...]        # Protected features
TARGET = [...]      # List of output labels
EXPL = ''           # Explanatory feature

inv = Discovery(data, SENS, TARGET, EXPL, topk=10)
```

#### Error Profiling
`ErrorProfiling` investigations let you search for user subpopulations for which
an algorithm exhibits abnormally high error rates. The investigation expects
an additional input specifying the *ground truth* for the algorithm's 
predictions. An appropriate error measure is then computed:

```python
from fairtest.error_profiling import ErrorProfiling

SENS = [...]            # Protected features
TARGET = '...'          # Predicted output
GROUND_TRUTH = '...'    # Ground truth feature
EXPL = ''               # Explanatory feature

inv = ErrorProfiling(data, SENS, TARGET, GROUND_TRUTH, EXPL)
```

#### Explanatory Attribute
It is possible to specify a user attribute as *explanatory*, meaning that 
FairTest will only look for associations among users that are equal with
respect to this attribute. We currently support a single, categorical attribute
as explanatory for investigations with categorical protected features and 
outputs. Support for more general explanatory attributes can be enabled by
defining further *Fairness Metrics* (see Extensions section below).

#### Other Examples
Additional examples, demonstrating how to use FairTest, are at: `src/fairtest/examples`.


Extensions
----------

#### Metrics
FairTest currently supports the following metrics:

* Normalized Mutual Information (`NMI`)
    - For categorical protected feature and output
* Normalized Conditional Mutual Information (`CondNMI`)
    - For categorical protected feature and output with explanatory feature
* Binary Ratio (`RATIO`)
    - For binary protected feature and output
* Binary Difference (`DIFF`)
    - For binary protected feature and output
* Conditional Binary Difference (`CondDIFF`)
    - For binary protected feature and output with explanatory feature
* Pearson Correlation (`CORR`)
    - For ordinal protected feature and output
* Logistic Regression (`REGRESSION`)
    - For binary protected feature and multi-labeled output

By default FairTest selects an appropriate metric depending on the type of
investigation and of protected and output features provided. You can specify 
a particular metric to use (as long as that metric is applicable to the data at
hand) with the `metrics` parameter passed to an `Investigation`:

```python
from fairtest.testing import Testing

SENS = ['gender', 'race']   # Protected features
TARGET = 'income'           # Output
EXPL = ''                   # Explanatory feature

metrics = {'gender': 'DIFF'}  # Specify a metric for 'gender' and let FairTest
                            # select a default metric for 'race'

inv = Testing(data, SENS, TARGET, EXPL, metrics=metrics)
```

FairTest can be extended with custom metrics, in order to handle situations
where the above metrics are not applicable. The class 
`fairtest.modules.metrics.metric.Metric` defines an abstract metric. Metrics can
expect three types of data: in the form of a contingency table (categorical 
features), of aggregate statistics (ordinal features), or non-aggregated data
(for regression). The main method called on a `Metric` is `compute`, which 
calculates a confidence interval and p-value and stores these as the class
attribute `stats`. The abstract `Metric` class provides a default `compute` 
method that calls instance specific methods for computing either exact or 
approximate statistics. Subclasses of `Metric` can either implement these
specific methods (see `fairtest.modules.metrics.mutual_info.NMI` for instance) 
or redefine the `compute`method entirely (see for example 
`fairtest.modules.metrics.regression.REGRESSION`). 


#### Logging
FairTest uses `Python's` standard `logging` module to log simple information
about ongoing investigations, as well as more fine-grained debug information 
(mainly for the guided tree learning algorithm).


Code Organisation
-----------------

Directory or File                       | Description
--------------------------------------- | ------------------------------------
data                                    |  Demo datasets
src/apps                                |  Demo apps
src/fairtest/tests                      |  Benchmarks
src/fairtest/examples                   |  Examples
src/fairtest/modules/bug_report         |  Bug filter, rank and report module
src/fairtest/modules/context_discovery  |  Guided tree construction module
src/fairtest/modules/metrics            |  Fairness metrics module
src/fairtest/modules/statistics         |  Statistical tests module
src/fairtest/discovery.py               |  Discovery Investigations
src/fairtest/error_profiling.py         |  ErrorProfiling Investigations
src/fairtest/investigation.py           |  Train, Test, Report for arbitrary Investigations
src/fairtest/testing.py                 |  Testing Investigations
