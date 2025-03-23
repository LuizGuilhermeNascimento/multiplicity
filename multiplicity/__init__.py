"""
multiplicity - A package for analyzing predictive multiplicity in machine learning models
"""

from . import metrics
from . import models
from .models import MultiplicityModel
from .metrics.multiplicity import (
    multiplicity_score,
    decision_boundary_analysis,
    stability_score,
)
from .visualization.plots import (
    plot_decision_boundaries,
    plot_regression_curves,
    plot_multiplicity_heatmap,
    plot_stability_analysis,
)

__version__ = "0.1.0"

__all__ = [
    "metrics",
    "models",
    "MultiplicityModel",  # For backward compatibility and convenience
    "multiplicity_score",
    "decision_boundary_analysis",
    "stability_score",
    "plot_decision_boundaries",
    "plot_regression_curves",
    "plot_multiplicity_heatmap",
    "plot_stability_analysis",
] 