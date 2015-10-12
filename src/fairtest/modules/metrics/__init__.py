"""
Fairness Metrics
"""
from .metric import Metric
from .mutual_info import NMI
from .binary_metrics import DIFF, RATIO, CondDIFF
from .correlation import CORR
from .regression import REGRESSION
