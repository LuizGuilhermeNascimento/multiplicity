from typing import Callable, List, Tuple, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array
from sklearn.ensemble import RandomForestClassifier

class ReconcileGBM(BaseEstimator):
    def __init__(
        self,
        f_star: Callable[[np.ndarray], np.ndarray],
        alpha: float = 0.1,
        epsilon: float = 0.1,
        max_iterations: int = 10,
        base_classifier: Optional[Any] = None
    ) -> None:
        self.f_star = f_star
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.base_classifier = base_classifier if base_classifier is not None else RandomForestClassifier()
        self.trees_: List[DecisionTreeRegressor] = []
        self.deltas_: List[float] = []
        self.fitted_classifier_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ReconcileGBM':
        X, y = check_X_y(X, y)
        self.fitted_classifier_ = clone(self.base_classifier)
        self.fitted_classifier_.fit(X, y)
        self._reconcile(X)
        return self

    def _reconcile(self, X: np.ndarray) -> None:

        def f(X_):
            proba = self.fitted_classifier_.predict_proba(X_)
            return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        
        f_star = self.f_star
        t = 0
        f_t_preds = f(X)
        m = int(np.ceil(2 / np.sqrt(self.alpha * self.epsilon)))
        self.trees_ = []
        self.deltas_ = []
        n = X.shape[0]
        while t < self.max_iterations:
            f_star_preds = f_star(X)
            disagreement = np.abs(f_t_preds - f_star_preds) > self.epsilon
            mass = np.mean(disagreement)

            if mass < self.alpha:
                break

            X_disagreement = X[disagreement]
            if X_disagreement.shape[0] == 0:
                break
            y_disagreement = f_star(X_disagreement)

            tree = DecisionTreeRegressor()
            tree.fit(X_disagreement, y_disagreement)
            h_t_preds = tree.predict(X)

            delta = np.abs(np.mean(f_t_preds[disagreement]) - np.mean(f_star_preds[disagreement]))
            f_t_preds = f_t_preds + delta * (h_t_preds - f_t_preds)
            f_t_preds = np.clip(f_t_preds, 0.0, 1.0)

            self.trees_.append(tree)
            self.deltas_.append(delta)
            t += 1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)

        if self.fitted_classifier_ is None:
            raise RuntimeError('You must fit the model before predicting.')
        
        proba = self.fitted_classifier_.predict_proba(X)
        base_pred = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        pred = base_pred.copy()

        for tree, delta in zip(self.trees_, self.deltas_):
            h_pred = tree.predict(X)
            pred = pred + delta * (h_pred - pred)
            pred = np.clip(pred, 0.0, 1.0)
            
        return np.column_stack([1 - pred, pred])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int) 