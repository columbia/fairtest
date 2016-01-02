"""
FairTest Module
"""

from .testing import Testing
from .discovery import Discovery
from .error_profiling import ErrorProfiling
from .investigation import Investigation, metric_from_string
from .investigation import train, test, report
from .holdout import DataSource
