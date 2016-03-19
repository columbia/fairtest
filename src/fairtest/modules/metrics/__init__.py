"""
Fairness Metrics
"""
from .metric import Metric
from .mutual_info import NMI, CondNMI
from .binary_metrics import DIFF, RATIO, CondDIFF
from .correlation import CORR, CondCORR
from .regression import REGRESSION
