from typing import Tuple, Dict, Any, Optional, Callable
import numpy as np
import pandas as pd
import itertools
from datetime import datetime
from numpy.typing import NDArray
from enum import Enum
from ..models.neural import NeuralPredictor

# Type variables for better type hints
FloatArray = NDArray[np.float64]

class Subscript(Enum):
    GREATER = 0
    SMALLER = 1

class Reconcile:

    def _calculate_probability_mass(data: pd.DataFrame, subset: pd.DataFrame) -> float:
        """Calculate the probability mass of a subset in relation to the full dataset.
        
        Args:
            data: The complete dataset
            subset: A subset of the data
            
        Returns:
            float: The probability mass (proportion) of the subset
        """
        return len(subset) / len(data) if len(data) > 0 else 0

    def _round_to_fraction(value: float, m: int) -> float:
        """Round a value to the nearest fraction determined by m.
        
        Args:
            value: The value to round
            m: The denominator for fractions
            
        Returns:
            float: The rounded value
        """
        return round(value * m) / m

    def _find_disagreement_set(
        f1_preds: np.ndarray,
        f2_preds: np.ndarray,
        X: pd.DataFrame,
        epsilon: float
    ) -> Tuple[pd.Index, pd.Index, pd.Index]:
        diff = np.abs(f1_preds - f2_preds)
        mask_epsilon = diff > epsilon
        mask_greater = f1_preds > f2_preds
        mask_smaller = f1_preds < f2_preds

        u_idx = X.index[mask_epsilon]
        u_greater_idx = X.index[mask_epsilon & mask_greater]
        u_smaller_idx = X.index[mask_epsilon & mask_smaller]

        return u_idx, u_greater_idx, u_smaller_idx

    def _find_candidate_for_update(
        u_greater_idx: pd.Index,
        u_smaller_idx: pd.Index,
        X: pd.DataFrame,
        y: pd.Series,
        f1_preds: np.ndarray,
        f2_preds: np.ndarray
    ) -> Tuple[int, int]:
        u = [u_greater_idx, u_smaller_idx]

        v_star = [
            np.mean(y.loc[u_greater_idx]),
            np.mean(y.loc[u_smaller_idx])
        ]

        v = [
            [np.mean(f1_preds[u[0]]), np.mean(f1_preds[u[1]])],
            [np.mean(f2_preds[u[0]]), np.mean(f2_preds[u[1]])]
        ]

        violations = []
        for subscript, i in itertools.product([Subscript.GREATER, Subscript.SMALLER], [0, 1]):
            prob_mass = len(u[subscript]) / len(X)
            violation = prob_mass * (v_star[subscript] - v[i][subscript]) ** 2
            violations.append((violation, subscript, i))

        max_violation = max(violations, key=lambda x: x[0])
        return max_violation[1], max_violation[2]

    def _patch(predictions: np.ndarray, indices: pd.Index, delta: float, all_indices: pd.Index) -> np.ndarray:
        new_predictions = predictions.copy()
        mask = all_indices.isin(indices)
        new_predictions[mask] += delta
        return np.clip(new_predictions, 0, 1)

    @staticmethod
    def reconcile(
        f1_preds: Any,
        f2_preds: Any,
        X: pd.DataFrame,
        y: pd.Series,
        alpha: float = 0.1,
        epsilon: float = 0.2,
        max_iterations: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:

        m = round(2 / (np.sqrt(alpha) * epsilon))

        all_indices = X.index

        u_idx, u_greater_idx, u_smaller_idx = _find_disagreement_set(f1_preds, f2_preds, X, epsilon)
        t = 0

        while calculate_probability_mass(X, X.loc[u_idx]) >= alpha and t < max_iterations:
            subscript, i = _find_candidate_for_update(u_greater_idx, u_smaller_idx, X, y, f1_preds, f2_preds)

            g_idx = u_greater_idx if subscript == Subscript.GREATER else u_smaller_idx
            true_labels_subset = y.loc[g_idx]

            if i == 0:
                predictions_subset = f1_preds[g_idx]
                delta = np.mean(true_labels_subset) - np.mean(predictions_subset)
                delta = round_to_fraction(delta, m)
                f1_preds = _patch(f1_preds, g_idx, delta, all_indices)
            else:
                predictions_subset = f2_preds[g_idx]
                delta = np.mean(true_labels_subset) - np.mean(predictions_subset)
                delta = round_to_fraction(delta, m)
                f2_preds = _patch(f2_preds, g_idx, delta, all_indices)

            t += 1
            u_idx, u_greater_idx, u_smaller_idx = _find_disagreement_set(f1_preds, f2_preds, X, epsilon)

        return f1_preds, f2_preds
