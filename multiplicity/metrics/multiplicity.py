from typing import Optional, Tuple, Union

import numpy as np
from sklearn.metrics import pairwise_distances


def multiplicity_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classifier_probs: np.ndarray,
    threshold: float = 0.1
) -> float:
    """Calculate the predictive multiplicity score.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        classifier_probs: Probability estimates from classifier
        threshold: Threshold for considering predictions as multiple
        
    Returns:
        float: Multiplicity score between 0 and 1
    """
    # Calculate prediction differences
    pred_diff = np.abs(y_pred[:, np.newaxis] - y_pred)
    
    # Calculate probability differences
    prob_diff = np.abs(classifier_probs[:, np.newaxis] - classifier_probs)
    
    # Identify cases with similar probabilities but different predictions
    multiplicity_mask = (prob_diff < threshold) & (pred_diff > threshold)
    
    # Calculate multiplicity score
    return np.mean(multiplicity_mask)


def decision_boundary_analysis(
    model: "MultiplicityModel",
    X: np.ndarray,
    n_samples: int = 1000,
    radius: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Analyze decision boundary regions for multiplicity.
    
    Args:
        model: Trained MultiplicityModel instance
        X: Input data
        n_samples: Number of samples to generate
        radius: Radius for neighborhood analysis
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Boundary points and their multiplicity scores
    """
    # Generate random samples around existing points
    idx = np.random.choice(len(X), n_samples)
    noise = np.random.normal(0, radius, size=(n_samples, X.shape[1]))
    X_boundary = X[idx] + noise
    
    # Get predictions and probabilities
    y_pred = model.predict(X_boundary)
    probs = model.predict_proba(X_boundary)[:, 1]
    
    # Calculate pairwise distances
    distances = pairwise_distances(X_boundary)
    
    # Calculate local multiplicity scores
    local_scores = np.zeros(n_samples)
    for i in range(n_samples):
        neighbors = distances[i] < radius
        if np.sum(neighbors) > 1:
            local_scores[i] = multiplicity_score(
                y_pred[neighbors],
                y_pred[neighbors],
                probs[neighbors]
            )
    
    return X_boundary, local_scores


def stability_score(
    model: "MultiplicityModel",
    X: np.ndarray,
    n_perturbations: int = 10,
    noise_std: float = 0.01
) -> np.ndarray:
    """Evaluate prediction stability across similar instances.
    
    Args:
        model: Trained MultiplicityModel instance
        X: Input data
        n_perturbations: Number of perturbations per instance
        noise_std: Standard deviation of Gaussian noise
        
    Returns:
        np.ndarray: Stability scores for each instance
    """
    stability_scores = np.zeros(len(X))
    
    for i in range(len(X)):
        # Generate perturbed versions of the instance
        X_perturbed = np.tile(X[i], (n_perturbations, 1))
        X_perturbed += np.random.normal(0, noise_std, X_perturbed.shape)
        
        # Get predictions for perturbed instances
        y_pred = model.predict(X_perturbed)
        probs = model.predict_proba(X_perturbed)[:, 1]
        
        # Calculate stability score as inverse of prediction variance
        stability_scores[i] = 1.0 / (np.var(y_pred) + 1e-10)
    
    return stability_scores 