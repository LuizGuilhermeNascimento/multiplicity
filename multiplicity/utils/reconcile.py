from typing import Tuple, Dict, Any, Optional, Callable
import numpy as np
import pandas as pd
import itertools
from datetime import datetime
from numpy.typing import NDArray
from enum import Enum

# Type variables for better type hints
FloatArray = NDArray[np.float64]

class Subscript(Enum):
    GREATER = 0
    SMALLER = 1

def calculate_probability_mass(data: pd.DataFrame, subset: pd.DataFrame) -> float:
    """Calculate the probability mass of a subset in relation to the full dataset.
    
    Args:
        data: The complete dataset
        subset: A subset of the data
        
    Returns:
        float: The probability mass (proportion) of the subset
    """
    return len(subset) / len(data) if len(data) > 0 else 0

def round_to_fraction(value: float, m: int) -> float:
    """Round a value to the nearest fraction determined by m.
    
    Args:
        value: The value to round
        m: The denominator for fractions
        
    Returns:
        float: The rounded value
    """
    return round(value * m) / m

def initialize_predictions(
    model1: Any,
    model2: Any,
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, float, float, list]:
    """Generate initial predictions and calculate metrics."""
    model1_predictions = model1.predict(X)
    model2_predictions = model2.predict(X)

    predictions_df = pd.DataFrame({
        'f1_predictions': model1_predictions,
        'f2_predictions': model2_predictions
    }, index=X.index)

    brier1 = np.mean(np.square(model1_predictions - y))
    brier2 = np.mean(np.square(model2_predictions - y))
    brier_scores = [[brier1, brier2]]

    return predictions_df, brier1, brier2, brier_scores


def find_disagreement_set(
    predictions_df: pd.DataFrame,
    X: pd.DataFrame,
    epsilon: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find sets where models disagree significantly."""
    f1_preds = predictions_df['f1_predictions']
    f2_preds = predictions_df['f2_predictions']

    diff = np.abs(f1_preds - f2_preds)
    mask_epsilon = diff > epsilon
    u_epsilon = X[mask_epsilon]

    mask_greater = f1_preds > f2_preds
    mask_smaller = f1_preds < f2_preds
    u_greater = u_epsilon[mask_greater[mask_epsilon]]
    u_smaller = u_epsilon[mask_smaller[mask_epsilon]]

    return u_epsilon, u_greater, u_smaller


def find_candidate_for_update(
    u_greater: pd.DataFrame,
    u_smaller: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    predictions_df: pd.DataFrame
) -> Tuple[int, int]:
    """Find the best candidate for update."""
    u = [u_greater, u_smaller]
    f1_preds = predictions_df['f1_predictions']
    f2_preds = predictions_df['f2_predictions']

    greater_idx = u_greater.index
    smaller_idx = u_smaller.index

    v_star = [
        np.mean(y.loc[greater_idx]),
        np.mean(y.loc[smaller_idx])
    ]

    v = [
        [np.mean(f1_preds.loc[greater_idx]), np.mean(f1_preds.loc[smaller_idx])],
        [np.mean(f2_preds.loc[greater_idx]), np.mean(f2_preds.loc[smaller_idx])]
    ]

    violations = []
    for subscript, i in itertools.product([Subscript.GREATER, Subscript.SMALLER], [0, 1]):
        prob_mass = calculate_probability_mass(X, u[subscript])
        violation = prob_mass * (v_star[subscript] - v[i][subscript]) ** 2
        violations.append((violation, subscript, i))

    max_violation = max(violations, key=lambda x: x[0])
    return max_violation[1], max_violation[2]


def patch(predictions: np.ndarray, indices: pd.Index, delta: float, prediction_index: pd.Index) -> np.ndarray:
    """Update predictions for a subset of data points."""
    new_predictions = predictions.copy()
    mask = prediction_index.isin(indices)
    new_predictions[mask] += delta
    return np.clip(new_predictions, 0, 1)


def reconcile(
    model1: Any,
    model2: Any,
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.1,
    epsilon: float = 0.2,
    max_iterations: int = 1000
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run the reconciliation algorithm."""
    start_time = datetime.now()
    m = round(2 / (np.sqrt(alpha) * epsilon))

    predictions_df, b1_init, b2_init, brier_scores = initialize_predictions(
        model1, model2, X, y
    )

    t = t1 = t2 = 0

    u, u_greater, u_smaller = find_disagreement_set(predictions_df, X, epsilon)
    initial_disagreement = calculate_probability_mass(X, u)

    while calculate_probability_mass(X, u) >= alpha and t < max_iterations:
        subscript, i = find_candidate_for_update(u_greater, u_smaller, X, y, predictions_df)

        col_name = f'f{"1" if i == 0 else "2"}_predictions'
        selected_predictions = predictions_df[col_name]
        g = u_greater if subscript == Subscript.GREATER else u_smaller
        g_indices = g.index

        true_labels_subset = y.loc[g_indices]
        predictions_subset = selected_predictions.loc[g_indices]

        delta = np.mean(true_labels_subset) - np.mean(predictions_subset)
        delta = round_to_fraction(delta, m)

        patched = patch(selected_predictions.values, g_indices, delta, predictions_df.index)
        predictions_df[col_name] = patched

        t += 1
        if i == 0:
            t1 += 1
        else:
            t2 += 1

        brier = [
            np.mean(np.square(predictions_df['f1_predictions'] - y)),
            np.mean(np.square(predictions_df['f2_predictions'] - y))
        ]
        brier_scores.append(brier)

        u, u_greater, u_smaller = find_disagreement_set(predictions_df, X, epsilon)

    final_predictions = (
        predictions_df['f1_predictions'].values,
        predictions_df['f2_predictions'].values
    )
    final_diff = np.abs(final_predictions[0] - final_predictions[1])
    final_disagreement = calculate_probability_mass(X, X[final_diff > epsilon])

    metrics = {
        "initial_disagreement": initial_disagreement,
        "final_disagreement": final_disagreement,
        "initial_brier": brier_scores[0],
        "final_brier": brier_scores[-1],
        "total_rounds": t,
        "model1_updates": t1,
        "model2_updates": t2,
        "runtime_seconds": (datetime.now() - start_time).seconds,
        "brier_scores_history": brier_scores
    }

    return final_predictions[0], final_predictions[1], metrics