import numpy as np
from typing import List, Tuple, Callable
from .base import BasePredictor, Event

## A loss_fn deve receber as preds e um vetor de labels, não um valor fixo como 0 ou 1
def find_disagreement_event(
    preds1: np.ndarray,
    preds2: np.ndarray,
    loss_fn: Callable,
    alpha: float
) -> Tuple[Event, np.ndarray]:
    """Find the disagreement event between predictors using a single loss function"""
    loss_diff1 = loss_fn(preds1, 1) - loss_fn(preds1, 0)
    loss_diff2 = loss_fn(preds2, 0) - loss_fn(preds2, 1)

    mask = ((loss_diff1 > alpha) | (loss_diff2 > alpha)).astype(float)
    event = Event(loss_fn, 0, 1) if np.mean(mask) > 0 else None

    return event, mask

def calculate_update(
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

def reconcile_binary_predictors(
    predictor1,
    predictor2,
    x: np.ndarray,
    y: np.ndarray,
    loss_fn: Callable,
    alpha: float = 0.001,
    eta: float = 0.01,
    beta: float = 0.00001,
    max_iterations: int = 1000
) -> Tuple[list, list]:
    """Run the ReDCal algorithm for binary classification using a single loss function"""
    predictor1.fit(x, y)
    predictor2.fit(x, y)

    log_loss_scores1 = []
    log_loss_scores2 = []

    for iteration in range(max_iterations):
        preds1 = predictor1.predict(x)
        preds2 = predictor2.predict(x)

        event, event_mask = find_disagreement_event(preds1, preds2, loss_fn, alpha)

        if event is None or np.mean(event_mask) < eta:
            break

        predictor_idx, phi = calculate_update(y, event_mask, preds1, preds2, event)

        if predictor_idx == 0:
            predictor1.update(x, y, event_mask, phi)
        else:
            predictor2.update(x, y, event_mask, phi)

        eps = 1e-15
        log_loss1 = -np.mean(y * np.log(preds1 + eps) + (1 - y) * np.log(1 - preds1 + eps))
        log_loss2 = -np.mean(y * np.log(preds2 + eps) + (1 - y) * np.log(1 - preds2 + eps))
        log_loss_scores1.append(log_loss1)
        log_loss_scores2.append(log_loss2)

    return log_loss_scores1, log_loss_scores2