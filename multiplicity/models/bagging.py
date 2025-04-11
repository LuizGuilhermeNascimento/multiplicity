from typing import List, Optional, Union, Any
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_random_state
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import cpu_count

class BaggingModel:
    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_estimators: int = 10,
        max_samples: Union[int, float] = 1.0,
        bootstrap: bool = True,
        n_jobs: int = 1,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the Bagging model.

        Parameters
        ----------
        base_estimator : BaseEstimator
            The base estimator to fit on random subsets of the dataset
        n_estimators : int, default=10
            The number of base estimators in the ensemble
        max_samples : int or float, default=1.0
            The number of samples to draw from X to train each base estimator
        bootstrap : bool, default=True
            Whether samples are drawn with replacement
        n_jobs : int, default=1
            The number of jobs to run in parallel
        random_state : int, optional
            Controls random number generation
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.random_state = random_state
        self.estimators_: List[BaseEstimator] = []
        self.random_state_ = check_random_state(random_state)

    def _fit_estimator(
        self, 
        estimator: BaseEstimator, 
        X: np.ndarray, 
        y: np.ndarray, 
        sample_indices: np.ndarray
    ) -> BaseEstimator:
        """
        Fit an estimator on a subset of the training data.

        Parameters
        ----------
        estimator : BaseEstimator
            The estimator to fit
        X : np.ndarray
            Training data
        y : np.ndarray
            Target values
        sample_indices : np.ndarray
            Indices of samples to use for training

        Returns
        -------
        BaseEstimator
            The fitted estimator
        """
        X_subset = X[sample_indices]
        y_subset = y[sample_indices]
        return estimator.fit(X_subset, y_subset)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggingModel':
        """
        Fit the bagging model.

        Parameters
        ----------
        X : np.ndarray
            Training data
        y : np.ndarray
            Target values

        Returns
        -------
        self : BaggingModel
            The fitted model
        """
        n_samples = X.shape[0]
        
        if isinstance(self.max_samples, float):
            max_samples = int(self.max_samples * n_samples)
        else:
            max_samples = self.max_samples

        self.estimators_ = []
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            
            for i in range(self.n_estimators):
                estimator = clone(self.base_estimator)
                if self.bootstrap:
                    indices = self.random_state_.randint(0, n_samples, max_samples)
                else:
                    indices = self.random_state_.permutation(n_samples)[:max_samples]
                
                futures.append(
                    executor.submit(self._fit_estimator, estimator, X, y, indices)
                )
            
            self.estimators_ = [future.result() for future in futures]
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the bagging model.

        Parameters
        ----------
        X : np.ndarray
            Input samples

        Returns
        -------
        np.ndarray
            Predicted values
        """
        predictions = np.array([
            estimator.predict(X) for estimator in self.estimators_
        ])
        
        # For classification tasks
        if predictions.dtype.kind in {'U', 'S', 'O', "i"} or len(predictions.shape) > 2:
            # Use mode for classification
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=predictions
            )
        
        # For regression tasks
        return np.mean(predictions, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : np.ndarray
            Input samples

        Returns
        -------
        np.ndarray
            Class probabilities for each sample
        """
        if not hasattr(self.base_estimator, "predict_proba"):
            raise AttributeError("Base estimator doesn't have predict_proba method")
        
        probas = np.array([
            estimator.predict_proba(X) for estimator in self.estimators_
        ])
        return np.mean(probas, axis=0) 