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
installed. First, add the latests version of `R` (for Ubuntu 12.04 and 14.04).
```
sudo sh -c 'echo "deb http://cran.rstudio.com/bin/linux/ubuntu trusty/" >> /etc/apt/sources.list'
gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
gpg -a --export E084DAB9 | sudo apt-key add -
sudo apt-get update
sudo apt-get -y install r-base r-base-dev
```

Then, make sure python is properly installed, along with mongo-db-server (required when Fairtest is used as a service),
and numpy (a fundamental package for scientific computing).
```
sudo apt-get -y install python python-dev python-pip mongodb-org-server liblzma-dev python-numpy libfreetype6-dev 
```

Now, create and activate a python2.7 virtual environment
```
sudo apt-get install python-virtualenv
virtualenv -p /usr/bin/python2.7 venv
source venv/bin/activate
```

Finally, install pip package dependencies.
```
python setup.py install
```

Alternatively, you can download an Ubuntu virtual machine with a complete,
up-to-date FairTest installation from
<a href="http://www.cs.columbia.edu/~vatlidak/FairtestVM.tar.gz" title="FairTest VM">here</a>
and launch it with VMWare Player.

Quick Start
-----------

Different benchmark datasets in CSV format are located in `fairtest/data`. You
can use the `utils.prepare_data.data_from_csv()` function to load a dataset as
a `Pandas DataFrame`, the format expected by FairTest investigations. The first
line of the csv file should list the names of the different features.

```python
from fairtest.utils.prepare_data import data_from_csv

data = data_from_csv('fairtest/data/adult/adult.csv', to_drop=['fnlwgt'])
```

The data is then pre-processed and split into training and testing sets by
encapsulating it in a `DataSource` object.

```python
from fairtest.holdout import DataSource

data = DataSource(data, budget=1, conf=0.95)
```

This creates a training set and holdout set that can be used to perform a single
batch of investigations with an overall testing confidence of 95%. Budgets
larger than 1 allow for `adaptive data analysis`, where new
investigations may be performed based on previous results, and validated over
an independent testing set.

#### Testing
To test for associations between user income and race or gender, first create
the appropriate Fairtest `Investigation`:

```python
from fairtest.testing import Testing

SENS = ['sex', 'race']     # Protected features
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
report(all_investigations, 'adult', output_dir='/tmp/')
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
Additional examples, demonstrating how to use FairTest, are at:
`src/fairtest/examples`.


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


Citing This Work
----------------
If you use FairTest for academic research, you are highly encouraged to cite the following paper:

```
@article{tramer2015fairtest,
  title={FairTest: Discovering Unwarranted Associations in Data-Driven Applications},
  author={Tramer, Florian and Atlidakis, Vaggelis and Geambasu, Roxana and Hsu, Daniel 
          and Hubaux, Jean-Pierre and Humbert, Mathias and Juels, Ari and Lin, Huang},
  journal={arXiv preprint arXiv:1510.02377},
  year={2015}
}
```

Reproducing Results
-------------------

To reproduce the results from our paper above, you can run the IPython notebooks
`medical.ipynb`, `recommender.ipynb` and `test.ipynb`. Make sure to restart
the notebooks for each experiment, to ensure that you start from a freshly
fixed random seed.
