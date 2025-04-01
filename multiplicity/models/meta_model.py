from typing import Any, Dict
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array

from .base import BaseMultiplicityModel

class MultiplicityModel(BaseMultiplicityModel):
    """Meta-model that combines classification and regression for predictive multiplicity analysis."""

    def __init__(
        self,
        classifier: ClassifierMixin,
        regressor: RegressorMixin,
        threshold: float = 0.5
    ) -> None:
        """Initialize the MultiplicityModel.

        Args:
            classifier: Any sklearn-compatible classifier
            regressor: Any sklearn-compatible regressor
            threshold: Classification threshold for binary predictions
        """
        self.clf = classifier
        self.reg = regressor
        self.threshold = threshold
        self._validate_models()

    def _validate_models(self) -> None:
        """Validate that the provided models are compatible."""
        if not isinstance(self.clf, (BaseEstimator, ClassifierMixin)):
            raise ValueError("Classifier must be a scikit-learn compatible classifier")
        if not isinstance(self.reg, (BaseEstimator, RegressorMixin)):
            raise ValueError("Regressor must be a scikit-learn compatible regressor")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiplicityModel":
        """Fit both classifier and regressor.

        Args:
            X: Training data
            y: Target values

        Returns:
            self: Returns an instance of self
        """
        X, y = check_X_y(X, y)
        self.clf.fit(X, y)
        
        clf_probs = self.clf.predict_proba(X)
        
        self.reg.fit(X, clf_probs[:, 1])
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the combined model.

        Args:
            X: Input data

        Returns:
            np.ndarray: Predicted values
        """
        X = check_array(X)
        raw_preds = (self.reg.predict(X) > self.threshold).astype(int)
        
        return np.clip(raw_preds, 0, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability estimates.

        Args:
            X: Input data

        Returns:
            np.ndarray: Probability estimates
        """
        X = check_array(X)
        reg_preds = self.reg.predict(X)
        probs = np.clip(reg_preds, 0, 1)
        
        return np.column_stack([1 - probs, probs])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate model performance score.

        Args:
            X: Input data
            y: True target values

        Returns:
            float: Performance score
        """
        X, y = check_X_y(X, y)
        preds = self.predict(X)
        return accuracy_score(y, preds)

    def get_reference_clf(self) -> ClassifierMixin:
        """Get the reference classifier."""
        return self.clf

    def get_reference_reg(self) -> RegressorMixin:
        """Get the reference regressor."""
        return self.reg

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters.

        Args:
            deep: If True, return parameters of nested models

        Returns:
            Dict[str, Any]: Model parameters
        """
        params = {
            "clf": self.clf,
            "reg": self.reg,
            "threshold": self.threshold,
        }
        
        if deep:
            params.update({
                f"clf__{k}": v
                for k, v in self.clf.get_params(deep=True).items()
            })
            params.update({
                f"reg__{k}": v
                for k, v in self.reg.get_params(deep=True).items()
            })
        
        return params

    def set_params(self, **params: Any) -> "MultiplicityModel":
        """Set model parameters.

        Args:
            **params: Parameters to set

        Returns:
            self: Returns an instance of self
        """
        if not params:
            return self
            
        valid_params = self.get_params(deep=True)
        
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key}")
            
            if key in ["clf", "reg", "threshold"]:
                setattr(self, key, value)
            elif key.startswith("clf__"):
                self.clf.set_params(**{key[12:]: value})
            elif key.startswith("reg__"):
                self.reg.set_params(**{key[11:]: value})
        
        return self 