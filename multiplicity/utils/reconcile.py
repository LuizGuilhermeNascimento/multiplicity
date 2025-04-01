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

class Reconcile:
    def __init__(
        self,
        model1: Any,
        model2: Any,
        X: pd.DataFrame,
        y: FloatArray,
        predict_fn: Optional[Callable[[Any, pd.DataFrame], FloatArray]] = None,
        alpha: float = 0.1,
        epsilon: float = 0.2
    ) -> None:
        """Initialize the Reconcile algorithm.
        
        Args:
            model1: First model with predict_proba or custom prediction method
            model2: Second model with predict_proba or custom prediction method
            X: Feature matrix for reconciliation and evaluation
            y: Target labels for reconciliation and evaluation
            predict_fn: Optional custom prediction function that takes (model, data) and returns probabilities
                       If None, will try to use model.predict_proba(X)[:, 1]
            alpha: Approximate group conditional mean consistency parameter
            epsilon: Disagreement threshold
        """
        self.model1 = model1
        self.model2 = model2
        self.alpha = alpha
        self.epsilon = epsilon
        self.X = X
        self.y = y
        
        # Set prediction function
        self.predict_fn = predict_fn if predict_fn is not None else self._default_predict_fn
        
        # Initialize tracking variables
        self.t = self.t1 = self.t2 = 0  # Rounds counters
        self.m = round(2 / (np.sqrt(alpha) * epsilon))  # Discretization parameter
        
        # Generate initial predictions and metrics
        self._initialize_predictions()
        
    def _default_predict_fn(self, model: Any, data: pd.DataFrame) -> FloatArray:
        """Default prediction function using predict_proba."""
        try:
            return model.predict_proba(data)[:, 1]
        except (AttributeError, IndexError) as e:
            raise ValueError(
                "Model doesn't support predict_proba or doesn't return probability scores. "
                "Please provide a custom predict_fn."
            ) from e
    
    def _initialize_predictions(self) -> None:
        """Generate initial predictions and calculate metrics."""
        # Generate predictions efficiently using vectorized operations
        model1_predictions = self.predict_fn(self.model1, self.X)
        model2_predictions = self.predict_fn(self.model2, self.X)
        
        # Create predictions DataFrame
        self.predictions_df = pd.DataFrame({
            'f1_predictions': model1_predictions,
            'f2_predictions': model2_predictions
        }, index=self.X.index)
        
        # Calculate initial metrics using vectorized operations
        self.initial_brier_score_1 = np.mean(np.square(model1_predictions - self.y))
        self.initial_brier_score_2 = np.mean(np.square(model2_predictions - self.y))
        self.brier_scores = [[self.initial_brier_score_1, self.initial_brier_score_2]]

    def find_disagreement_set(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Find sets where models disagree significantly."""
        f1_preds = self.predictions_df['f1_predictions']
        f2_preds = self.predictions_df['f2_predictions']
        
        # Vectorized operations for finding disagreements
        diff = np.abs(f1_preds - f2_preds)
        mask_epsilon = diff > self.epsilon
        u_epsilon = self.X[mask_epsilon]
        
        # Use boolean indexing for better performance
        mask_greater = f1_preds > f2_preds
        mask_smaller = f1_preds < f2_preds
        u_greater = u_epsilon[mask_greater[mask_epsilon]]
        u_smaller = u_epsilon[mask_smaller[mask_epsilon]]
        
        return u_epsilon, u_greater, u_smaller
    
    def find_candidate_for_update(self, u_greater: pd.DataFrame, u_smaller: pd.DataFrame) -> Tuple[int, int]:
        """Find the best candidate for update."""
        u = [u_greater, u_smaller]
        f1_preds = self.predictions_df['f1_predictions']
        f2_preds = self.predictions_df['f2_predictions']
        
        # Pre-calculate indices for better performance
        greater_idx = u_greater.index
        smaller_idx = u_smaller.index
        
        # Calculate means using vectorized operations
        v_star = [
            np.mean(self.y[self.X.index.isin(greater_idx)]),
            np.mean(self.y[self.X.index.isin(smaller_idx)])
        ]
        
        v = [
            [np.mean(f1_preds[f1_preds.index.isin(greater_idx)]),
             np.mean(f1_preds[f1_preds.index.isin(smaller_idx)])],
            [np.mean(f2_preds[f2_preds.index.isin(greater_idx)]),
             np.mean(f2_preds[f2_preds.index.isin(smaller_idx)])]
        ]
        
        # Find best candidate efficiently
        violations = []
        for subscript, i in itertools.product([Subscript.GREATER.value, Subscript.SMALLER.value], [0, 1]):
            prob_mass = calculate_probability_mass(self.X, u[subscript])
            violation = prob_mass * (v_star[subscript] - v[i][subscript]) ** 2
            violations.append((violation, subscript, i))
        
        max_violation = max(violations, key=lambda x: x[0])
        return max_violation[1], max_violation[2]
    
    def patch(self, predictions: FloatArray, indices: pd.Index, delta: float) -> FloatArray:
        """Update predictions for a subset of data points."""
        predictions = predictions.copy()
        mask = self.predictions_df.index.isin(indices)
        predictions[mask] += delta
        return np.clip(predictions, 0, 1)
    
    def reconcile(self) -> Tuple[FloatArray, FloatArray, Dict[str, Any]]:
        """Run the reconciliation algorithm.
        
        Returns:
            Tuple containing:
            - Final predictions from model 1
            - Final predictions from model 2
            - Dictionary with metrics and statistics
        """
        start_time = datetime.now()
        
        # Initial disagreement calculation
        u, u_greater, u_smaller = self.find_disagreement_set()
        diff = np.abs(self.predictions_df['f1_predictions'] - self.predictions_df['f2_predictions'])
        initial_disagreement = calculate_probability_mass(self.X, self.X[diff > self.epsilon])
        
        # Main reconciliation loop with optimized operations
        while calculate_probability_mass(self.X, u) >= self.alpha:
            subscript, i = self.find_candidate_for_update(u_greater, u_smaller)
            
            selected_predictions = self.predictions_df[f'f{"1" if i == 0 else "2"}_predictions']
            g = u_greater if subscript == Subscript.GREATER.value else u_smaller
            
            # Calculate updates using vectorized operations
            g_indices = g.index
            predictions_subset = selected_predictions[selected_predictions.index.isin(g_indices)]
            true_labels_subset = self.y[self.X.index.isin(g_indices)]
            
            delta = np.mean(true_labels_subset) - np.mean(predictions_subset)
            delta = round_to_fraction(delta, self.m)
            
            # Apply updates efficiently
            col_name = f'f{"1" if i == 0 else "2"}_predictions'
            self.predictions_df[col_name] = self.patch(self.predictions_df[col_name].values, g_indices, delta)
            
            if i == 0:
                self.t1 += 1
            else:
                self.t2 += 1
            
            # Update metrics using vectorized operations
            self.t += 1
            current_brier = [
                np.mean(np.square(self.predictions_df['f1_predictions'] - self.y)),
                np.mean(np.square(self.predictions_df['f2_predictions'] - self.y))
            ]
            self.brier_scores.append(current_brier)
            
            u, u_greater, u_smaller = self.find_disagreement_set()
        
        # Calculate final metrics
        final_predictions = (
            self.predictions_df['f1_predictions'].values,
            self.predictions_df['f2_predictions'].values
        )
        final_diff = np.abs(final_predictions[0] - final_predictions[1])
        final_disagreement = calculate_probability_mass(self.X, self.X[final_diff > self.epsilon])
        
        metrics = {
            "initial_disagreement": initial_disagreement,
            "final_disagreement": final_disagreement,
            "initial_brier": self.brier_scores[0],
            "final_brier": self.brier_scores[-1],
            "total_rounds": self.t,
            "model1_updates": self.t1,
            "model2_updates": self.t2,
            "runtime_seconds": (datetime.now() - start_time).seconds,
            "brier_scores_history": self.brier_scores
        }
        
        return final_predictions[0], final_predictions[1], metrics 