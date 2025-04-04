import numpy as np
from typing import List, Tuple, Callable
from .base import BasePredictor, Event

class BinaryReDCal:
    """Reconcile Decision Calibration (ReDCal) algorithm for binary classification
    
    Args:
        predictor1: First binary classifier instance
        predictor2: Second binary classifier instance
        loss_fns: List of binary classification loss functions
        alpha: Loss margin threshold
        eta: Disagreement region mass threshold
        beta: Decision calibration tolerance
    """
    def __init__(
        self,
        predictor1: BasePredictor,
        predictor2: BasePredictor,
        loss_fns: List[Callable],
        alpha: float = 0.001,
        eta: float = 0.01,
        beta: float = 0.00001
    ):
        self.predictor1 = predictor1
        self.predictor2 = predictor2
        self.loss_fns = loss_fns
        self.alpha = alpha  # Loss margin
        self.eta = eta  # Disagreement region mass threshold
        self.beta = beta  # Decision calibration tolerance
        
    def _find_largest_disagreement(
        self, 
        x: np.ndarray,
        preds1: np.ndarray,
        preds2: np.ndarray
    ) -> Tuple[Event, np.ndarray]:
        """Find the largest disagreement event between predictors
        
        Args:
            x: Input features of shape (n_samples, n_features)
            preds1: Predictions from first predictor of shape (n_samples, 1)
            preds2: Predictions from second predictor of shape (n_samples, 1)
            
        Returns:
            Tuple of (Event, disagreement mask array)
        """
        max_mass = 0
        max_event = None
        max_mask = None
        
        # For binary classification, we only need to check disagreement between
        # predicting class 0 vs class 1
        for loss_fn in self.loss_fns:
            # Calculate loss differences for predicting 0 vs 1
            loss_diff1 = loss_fn(preds1, 1) - loss_fn(preds1, 0)
            loss_diff2 = loss_fn(preds2, 0) - loss_fn(preds2, 1)
            
            # Create disagreement mask
            mask = ((loss_diff1 > self.alpha) | (loss_diff2 > self.alpha)).astype(float)
            mass = np.mean(mask)
            
            if mass > max_mass:
                max_mass = mass
                max_event = Event(loss_fn, 0, 1)
                max_mask = mask
                        
        return max_event, max_mask
    
    def _calculate_update(
        self,
        x: np.ndarray,
        y: np.ndarray,
        event_mask: np.ndarray,
        preds1: np.ndarray,
        preds2: np.ndarray,
        event: Event
    ) -> Tuple[int, np.ndarray]:
        """Calculate which predictor to update and the update direction
        
        Args:
            x: Input features of shape (n_samples, n_features)
            y: Binary target values of shape (n_samples, 1)
            event_mask: Binary mask indicating disagreement region
            preds1: Predictions from first predictor of shape (n_samples, 1)
            preds2: Predictions from second predictor of shape (n_samples, 1)
            event: Current disagreement event
            
        Returns:
            Tuple of (predictor index to update, update direction array)
        """
        # Calculate expected loss differences
        loss_diff1 = np.mean(
            (event.loss_fn(y, event.action1) - event.loss_fn(y, event.action2)) * event_mask
        )
        loss_diff2 = np.mean(
            (event.loss_fn(preds1, event.action1) - event.loss_fn(preds1, event.action2)) * event_mask
        )
        
        # Pick predictor with larger error
        if abs(loss_diff1 - loss_diff2) > abs(loss_diff1):
            predictor_idx = 0
            preds = preds1
        else:
            predictor_idx = 1
            preds = preds2
            
        # Calculate update direction
        event_samples = event_mask > 0
        if np.sum(event_samples) == 0:
            # If no samples in event region, return zero update
            return predictor_idx, np.zeros(1)
            
        # Calculate mean difference for binary predictions
        phi = np.mean(y[event_samples] - preds[event_samples])
        
        return predictor_idx, np.array([phi])
        
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_iterations: int = 1000
    ) -> Tuple[List[float], List[float]]:
        """Run the ReDCal algorithm for binary classification
        
        Args:
            x: Input features of shape (n_samples, n_features)
            y: Binary target values of shape (n_samples, 1)
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (predictor1 log loss scores, predictor2 log loss scores)
        """
        log_loss_scores1 = []
        log_loss_scores2 = []
        
        for iteration in range(max_iterations):
            # Get current predictions
            preds1 = self.predictor1.predict(x)
            preds2 = self.predictor2.predict(x)
            
            # Find largest disagreement
            event, event_mask = self._find_largest_disagreement(x, preds1, preds2)
            
            # Check termination condition
            if event is None or np.mean(event_mask) < self.eta:
                break
                
            # Calculate update
            predictor_idx, phi = self._calculate_update(
                x, y, event_mask, preds1, preds2, event
            )
            
            # Apply update
            if predictor_idx == 0:
                self.predictor1.update(x, y, event_mask, phi)
            else:
                self.predictor2.update(x, y, event_mask, phi)
                
            # Record log loss scores
            eps = 1e-15  # Small constant to avoid log(0)
            log_loss1 = -np.mean(y * np.log(preds1 + eps) + (1 - y) * np.log(1 - preds1 + eps))
            log_loss2 = -np.mean(y * np.log(preds2 + eps) + (1 - y) * np.log(1 - preds2 + eps))
            log_loss_scores1.append(log_loss1)
            log_loss_scores2.append(log_loss2)
            
        return log_loss_scores1, log_loss_scores2 