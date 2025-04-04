"""
multiplicity - A package for analyzing predictive multiplicity in machine learning models
"""

from . import metrics
from . import models
from .models import (
    MultiplicityModel,
    BaggingModel,
)
from .metrics.multiplicity import (
    rashomon_set,
    arbitrariness,
    pairwise_disagreement,
)
from .visualization.plots import (
    plot_prob_dist,
)
from .utils.reconcile import Reconcile

__version__ = "0.1.0"

__all__ = [
    "metrics",
    "models",
    "MultiplicityModel",  # For backward compatibility and convenience
    "BaggingModel",
    "rashomon_set",
    "arbitrariness",
    "pairwise_disagreement",
    "Reconcile",
    "plot_prob_dist",
] 