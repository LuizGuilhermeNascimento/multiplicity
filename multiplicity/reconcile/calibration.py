from typing import Tuple, Callable
import numpy as np
from ..models.neural import NeuralPredictor

class ReconcileCalibration:

    @staticmethod
    def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate the Brier score for binary classification per sample.
        
        Args:
            y_true: True binary labels (0 or 1)
            y_pred: Predicted probabilities for the positive class (1)
            
        Returns:
            np.ndarray: Brier score for each sample
        """
        return np.square(y_true - y_pred).reshape(-1)

    @staticmethod
    def find_disagreement_mask(
        preds1: np.ndarray,
        preds2: np.ndarray,
        loss_fn: Callable,
        alpha: float
    ) -> Tuple[bool, np.ndarray]:
        """Find the disagreement mask between predictors using a single loss function
        
        Args:
            preds1: Predictions from first model
            preds2: Predictions from second model
            loss_fn: Loss function to evaluate predictions (should return per-sample losses)
            alpha: Threshold for disagreement
            
        Returns:
            Tuple containing:
            - Boolean indicating if disagreement exists
            - Mask indicating which samples have disagreement
        """
        positive_label = np.ones_like(preds1)
        negative_label = np.zeros_like(preds1)

        loss_diff1 = loss_fn(preds1, positive_label) - loss_fn(preds1, negative_label)
        loss_diff2 = loss_fn(preds2, negative_label) - loss_fn(preds2, positive_label)

        mask = ((loss_diff1 > alpha) | (loss_diff2 > alpha)).astype(float)
        has_disagreement = np.mean(mask) > 0

        return has_disagreement, mask

    @staticmethod
    def calculate_update(
        y: np.ndarray,
        event_mask: np.ndarray,
        preds1: np.ndarray,
        preds2: np.ndarray,
        loss_fn: Callable
    ) -> Tuple[int, np.ndarray]:
        """Calculate which predictor to update and the update direction
        
        Args:
            y: True labels
            event_mask: Mask indicating disagreement samples
            preds1: Predictions from first model
            preds2: Predictions from second model
            loss_fn: Loss function to evaluate predictions (should return per-sample losses)
            
        Returns:
            Tuple containing:
            - Index of predictor to update (0 or 1)
            - Direction and magnitude of update
        """
        # For binary classification, action1=0, action2=1 for loss calculation
        positive_label = np.ones_like(preds1)
        negative_label = np.zeros_like(preds1)
        
        # Calculate the mean of per-sample loss differences weighted by event mask
        loss_diff1 = np.mean(
            (loss_fn(y, positive_label) - loss_fn(y, negative_label)) * event_mask
        )
        loss_diff2 = np.mean(
            (loss_fn(preds1, positive_label) - loss_fn(preds1, negative_label)) * event_mask
        )

        if abs(loss_diff1 - loss_diff2) > abs(loss_diff1):
            predictor_idx = 0
            preds = preds1
        else:
            predictor_idx = 1
            preds = preds2

        event_samples = event_mask > 0
        if np.sum(event_samples) == 0:
            return predictor_idx, np.zeros(1)

        phi = np.mean(y[event_samples] - preds[event_samples])
        return predictor_idx, np.array([phi])

    @staticmethod
    def reconcile_calibration(
        predictor1: NeuralPredictor,
        predictor2: NeuralPredictor,
        x: np.ndarray,
        y: np.ndarray,
        loss_fn: Callable = None,
        alpha: float = 0.001,
        eta: float = 0.01,
        max_iterations: int = 1000,
        verbose: bool = False
    ) -> Tuple[NeuralPredictor, NeuralPredictor]:
        """Run the ReDCal algorithm for binary classification
        
        Args:
            predictor1: First neural predictor
            predictor2: Second neural predictor
            x: Input features
            y: True labels
            loss_fn: Loss function (defaults to Brier score if None)
            alpha: Threshold for disagreement
            eta: Minimum fraction of disagreement samples
            max_iterations: Maximum number of iterations
            verbose: Whether to print progress
            
        Returns:
            Tuple containing the updated predictors
        """
        if loss_fn is None:
            loss_fn = ReconcileCalibration.brier_score
            
        for iteration in range(max_iterations):
            preds1 = predictor1.predict(x)
            preds2 = predictor2.predict(x)

            has_disagreement, event_mask = ReconcileCalibration.find_disagreement_mask(
                preds1, preds2, loss_fn, alpha
            )

            if not has_disagreement or np.mean(event_mask) < eta:
                break

            predictor_idx, phi = ReconcileCalibration.calculate_update(
                y, event_mask, preds1, preds2, loss_fn
            )

            if predictor_idx == 0:
                predictor1.update(x, y, event_mask, phi)
            else:
                predictor2.update(x, y, event_mask, phi)

            # Log loss calculation for monitoring (remains scalar)
            eps = 1e-15
            log_loss1 = -np.mean(y * np.log(preds1 + eps) + (1 - y) * np.log(1 - preds1 + eps))
            log_loss2 = -np.mean(y * np.log(preds2 + eps) + (1 - y) * np.log(1 - preds2 + eps))
            
            if verbose:
                print(f"Iteration {iteration}: Loss1 = {log_loss1}, Loss2 = {log_loss2}")

        return predictor1, predictor2