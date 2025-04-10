from typing import Tuple, Dict, Any, Optional, Callable
import numpy as np
import pandas as pd
import itertools
from datetime import datetime
from numpy.typing import NDArray
from enum import Enum
from ..models.neural import NeuralPredictor

class ReconcileCalibration:

    def _brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the Brier score for binary classification.
        
        Args:
            y_true: True binary labels (0 or 1)
            y_pred: Predicted probabilities for the positive class (1)
            
        Returns:
            float: Brier score
        """
        return np.mean(np.square(y_true - y_pred))

    def _find_disagreement_event(
        preds1: np.ndarray,
        preds2: np.ndarray,
        loss_fn: Callable,
        alpha: float
    ) -> Tuple[Event, np.ndarray]:
        """Find the disagreement event between predictors using a single loss function"""
        positive_label = np.ones_like(preds1)
        negative_label = np.zeros_like(preds1)

        loss_diff1 = loss_fn(preds1, positive_label) - loss_fn(preds1, negative_label)
        loss_diff2 = loss_fn(preds2, negative_label) - loss_fn(preds2, positive_label)

        mask = ((loss_diff1 > alpha) | (loss_diff2 > alpha)).astype(float)
        event = Event(loss_fn, 0, 1) if np.mean(mask) > 0 else None

        return event, mask

    def _calculate_update(
        y: np.ndarray,
        event_mask: np.ndarray,
        preds1: np.ndarray,
        preds2: np.ndarray,
        event: Event
    ) -> Tuple[int, np.ndarray]:
        """Calculate which predictor to update and the update direction"""
        loss_diff1 = np.mean(
            (event.loss_fn(y, event.action1) - event.loss_fn(y, event.action2)) * event_mask
        )
        loss_diff2 = np.mean(
            (event.loss_fn(preds1, event.action1) - event.loss_fn(preds1, event.action2)) * event_mask
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
        self,
        predictor1: NeuralPredictor,
        predictor2: NeuralPredictor,
        x: np.ndarray,
        y: np.ndarray,
        loss_fn: Callable = self._brier_score,
        alpha: float = 0.001,
        eta: float = 0.01,
        beta: float = 0.00001,
        max_iterations: int = 1000,
        verbose: bool = False
    ) -> Tuple[list, list]:
        """Run the ReDCal algorithm for binary classification using a single loss function"""

        for iteration in range(max_iterations):
            preds1 = predictor1.predict(x)
            preds2 = predictor2.predict(x)

            event, event_mask = self._find_disagreement_event(preds1, preds2, loss_fn, alpha)

            if event is None or np.mean(event_mask) < eta:
                break

            predictor_idx, phi = self._calculate_update(y, event_mask, preds1, preds2, event)

            if predictor_idx == 0:
                predictor1.update(x, y, event_mask, phi)
            else:
                predictor2.update(x, y, event_mask, phi)

            eps = 1e-15
            log_loss1 = -np.mean(y * np.log(preds1 + eps) + (1 - y) * np.log(1 - preds1 + eps))
            log_loss2 = -np.mean(y * np.log(preds2 + eps) + (1 - y) * np.log(1 - preds2 + eps))
            
            if verbose:
                print(f"Iteration {iteration}: Loss1 = {log_loss1}, Loss2 = {log_loss2}")

        return predictor1, predictor2