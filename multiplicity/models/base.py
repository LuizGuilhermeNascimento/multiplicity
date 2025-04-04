from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List, Callable, Set

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from dataclasses import dataclass


class BaseMultiplicityModel(BaseEstimator, ABC):
    """Base class for multiplicity models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseMultiplicityModel":
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability estimates."""
        pass
    
    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate model performance score."""
        pass
    
    @abstractmethod
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        pass
    
    @abstractmethod
    def set_params(self, **params: Any) -> "BaseMultiplicityModel":
        """Set model parameters."""
        pass

    @abstractmethod
    def get_reference_clf(self) -> ClassifierMixin:
        """Get the reference classifier."""
        pass

    @abstractmethod
    def get_reference_reg(self) -> RegressorMixin:
        """Get the reference regressor."""
        pass

class Event:
    """Event class representing a disagreement between predictors
    
    Args:
        loss_fn: Loss function used to evaluate predictions
        action1: First action (0 or 1)
        action2: Second action (0 or 1)
    """
    def __init__(self, loss_fn: Callable, action1: int, action2: int):
        self.loss_fn = loss_fn
        self.action1 = action1
        self.action2 = action2

class BasePredictor:
    """Base class for binary classifiers that can be updated
    
    All binary classifiers used with ReDCal should inherit from this class
    and implement the predict and update methods.
    """
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict probabilities for binary classification
        
        Args:
            x: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities of shape (n_samples, 1)
        """
        raise NotImplementedError
        
    def update(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray, direction: np.ndarray) -> None:
        """Update the predictor based on disagreement region
        
        Args:
            x: Input features of shape (n_samples, n_features)
            y: Target values of shape (n_samples, 1)
            mask: Binary mask indicating disagreement region
            direction: Update direction for predictions
        """
        raise NotImplementedError 