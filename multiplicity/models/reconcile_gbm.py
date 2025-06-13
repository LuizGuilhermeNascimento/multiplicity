from typing import Callable, List, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array
from sklearn.ensemble import RandomForestClassifier

class ReconcileModel:
    """
    Represents a reconciled prediction model built from a base prediction and a sequence of 
    regression adjustments (trees and their respective deltas) to align with a target model.
    
    Attributes:
        base_pred (np.ndarray): The final reconciled base predictions.
        trees (List[Any]): A list of trained regressors used to adjust predictions.
        deltas (List[float]): A list of scaling factors (one per regressor) used during reconciliation.
    """

    def __init__(self, base_f: Callable[[np.ndarray], np.ndarray], trees: List[Any], deltas: List[float], disagreements: List[np.ndarray]) -> None:
        """
        Initializes the ReconcileModel.

        Args:
            base_pred (np.ndarray): Initial prediction vector to be adjusted.
            trees (List[Any]): List of trained regressors.
            deltas (List[float]): List of delta values scaling the contribution of each tree.
        """
        self.base_f = base_f
        self.trees = trees
        self.deltas = deltas
        self.disagreements = disagreements
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates binary class predictions for input data.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Binary predictions (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns class probabilities for input data.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Probabilities for each class, shape (n_samples, 2).
        """
        pred = self.base_f(X)
        for tree, delta in zip(self.trees, self.deltas):
            h_pred = tree.predict(X)
            pred += delta * (h_pred - pred)
        final_pred = np.clip(pred, 0.0, 1.0)
        return np.column_stack([1 - final_pred, final_pred])


class ReconcileGBM(BaseEstimator):
    """
    Scikit-learn compatible estimator that reconciles predictions from a source model
    towards a target function using iterative gradient-like adjustments with regressors.

    Attributes:
        f_target (Callable): The target prediction function (e.g., real labels or external model).
        f_source (Callable): The initial source prediction function (e.g., base classifier).
        alpha (float): Threshold for the minimum disagreement mass to continue iterations.
        epsilon (float): Tolerance for disagreement between source and target predictions.
        max_iterations (int): Maximum number of reconciliation iterations.
        base_classifier_cls (type): Base classifier class to be trained if f_source is not provided.
        base_classifier_params (dict): Parameters for the base classifier.
        base_regressor_cls (type): Regressor class used in reconciliation steps.
        base_regressor_params (dict): Parameters for the base regressor.
    """

    def __init__(
        self,
        f_target: Optional[Callable[[np.ndarray], np.ndarray]],
        f_source: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        max_iterations: int = 50,
        base_classifier_cls: type = RandomForestClassifier,
        base_classifier_params: Optional[dict] = None,
        base_regressor_cls: type = DecisionTreeRegressor,
        base_regressor_params: Optional[dict] = None
    ) -> None:
        """
        Initializes the ReconcileGBM model with the given parameters.
        """
        self.f_target = f_target
        self.f_source = f_source
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.base_classifier_cls = base_classifier_cls
        self.base_classifier_params = base_classifier_params or {}
        self.base_regressor_cls = base_regressor_cls
        self.base_regressor_params = base_regressor_params or {}
        self.reconciled_model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ReconcileGBM':
        """
        Fits the model by either training a base classifier (if f_source is None) and then 
        applying the reconciliation procedure to align predictions with f_target.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.

        Returns:
            self: The fitted ReconcileGBM instance.
        """
        X, y = check_X_y(X, y)

        if self.f_source is None:
            self.fitted_classifier_ = clone(self.base_classifier_cls(**self.base_classifier_params))
            self.fitted_classifier_.fit(X, y)

            def f_source(x): return self.fitted_classifier_.predict_proba(x)[:, 1]
            self.f_source = f_source

        self.reconciled_model_ = self.reconcile(
            X,
            f_source=self.f_source,
            f_target=self.f_target,
            base_regressor_cls=self.base_regressor_cls,
            base_regressor_params=self.base_regressor_params,
            alpha=self.alpha,
            epsilon=self.epsilon,
            max_iterations=self.max_iterations
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for input samples.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        X = check_array(X)
        if self.reconciled_model_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.reconciled_model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for input samples.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted probabilities, shape (n_samples, 2).
        """
        X = check_array(X)
        if self.reconciled_model_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.reconciled_model_.predict_proba(X)

    @staticmethod
    def reconcile(
        X: np.ndarray,
        f_source: Callable[[np.ndarray], np.ndarray],
        f_target: Callable[[np.ndarray], np.ndarray],
        base_regressor_cls: type = DecisionTreeRegressor,
        base_regressor_params: Optional[dict] = None,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        max_iterations: int = 10
    ) -> ReconcileModel:
        """
        Runs the reconciliation procedure to align the source predictions with the target function.

        Args:
            X (np.ndarray): Input data.
            f_source (Callable): Source prediction function.
            f_target (Callable): Target prediction function.
            base_regressor_cls (type): Class of the base regressor.
            base_regressor_params (dict): Parameters for the base regressor.
            alpha (float): Minimum disagreement mass threshold to continue.
            epsilon (float): Disagreement tolerance threshold.
            max_iterations (int): Maximum number of boosting-like steps.

        Returns:
            ReconcileModel: An object representing the reconciled predictor.
        """
        base_regressor_params = base_regressor_params or {}
        f_t_preds = f_source(X)
        trees = []
        deltas = []
        disagreements = []
        t = 0

        while t < max_iterations:
            f_target_preds = f_target(X)
            disagreement = np.abs(f_t_preds - f_target_preds) > epsilon
            disagreements.append(disagreement)
            mass = np.mean(disagreement)

            if mass < alpha:
                break

            X_dis = X[disagreement]
            if X_dis.shape[0] == 0:
                break

            y_dis = f_target(X_dis)
            tree = base_regressor_cls(**base_regressor_params)
            tree.fit(X_dis, y_dis)
            h_t_preds = tree.predict(X)

            delta = np.abs(np.mean(f_t_preds[disagreement]) - np.mean(f_target_preds[disagreement]))
            f_t_preds = f_t_preds + delta * (h_t_preds - f_t_preds)
            f_t_preds = np.clip(f_t_preds, 0.0, 1.0)

            trees.append(tree)
            deltas.append(delta)
            t += 1

        return ReconcileModel(f_source, trees, deltas, disagreements)