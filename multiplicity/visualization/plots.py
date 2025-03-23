from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA


def plot_decision_boundaries(
    X: np.ndarray,
    scores: np.ndarray,
    title: str = "Decision Boundary Analysis",
    n_components: int = 2
) -> None:
    """Plot decision boundaries with multiplicity scores.
    
    Args:
        X: Input data points
        scores: Multiplicity scores for each point
        title: Plot title
        n_components: Number of PCA components for visualization
    """
    if X.shape[1] > 2:
        pca = PCA(n_components=n_components)
        X_plot = pca.fit_transform(X)
    else:
        X_plot = X

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=scores, cmap='viridis')
    plt.colorbar(scatter, label='Multiplicity Score')
    plt.title(title)
    plt.xlabel('Component 1' if X.shape[1] > 2 else 'Feature 1')
    plt.ylabel('Component 2' if X.shape[1] > 2 else 'Feature 2')
    plt.show()


def plot_regression_curves(
    model: "MultiplicityModel",
    X: np.ndarray,
    feature_idx: int = 0,
    n_points: int = 100
) -> None:
    """Plot regression curves for different classifier probabilities.
    
    Args:
        model: Trained MultiplicityModel instance
        X: Input data
        feature_idx: Index of feature to plot
        n_points: Number of points for plotting
    """
    x_min, x_max = X[:, feature_idx].min(), X[:, feature_idx].max()
    x_plot = np.linspace(x_min, x_max, n_points)
    
    # Create test points
    X_test = np.tile(X.mean(axis=0), (n_points, 1))
    X_test[:, feature_idx] = x_plot
    
    # Get predictions
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_plot, y_pred, c=probs, cmap='viridis')
    plt.colorbar(label='Classifier Probability')
    plt.title('Regression Curves')
    plt.xlabel(f'Feature {feature_idx}')
    plt.ylabel('Prediction')
    plt.show()


def plot_multiplicity_heatmap(
    X: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 50
) -> None:
    """Plot multiplicity heatmap for 2D data.
    
    Args:
        X: 2D input data
        scores: Multiplicity scores
        n_bins: Number of bins for heatmap
    """
    if X.shape[1] != 2:
        raise ValueError("Heatmap plotting requires 2D data")
    
    plt.figure(figsize=(10, 8))
    
    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(
        X[:, 0], X[:, 1],
        bins=n_bins,
        weights=scores
    )
    
    # Plot heatmap
    sns.heatmap(hist.T, cmap='viridis')
    plt.title('Multiplicity Heatmap')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def plot_stability_analysis(
    X: np.ndarray,
    stability_scores: np.ndarray,
    n_components: int = 2
) -> None:
    """Plot stability analysis results.
    
    Args:
        X: Input data
        stability_scores: Stability scores for each instance
        n_components: Number of PCA components for visualization
    """
    if X.shape[1] > 2:
        pca = PCA(n_components=n_components)
        X_plot = pca.fit_transform(X)
    else:
        X_plot = X
    
    plt.figure(figsize=(12, 6))
    
    # Plot stability scores
    plt.subplot(1, 2, 1)
    plt.hist(stability_scores, bins=50)
    plt.title('Stability Score Distribution')
    plt.xlabel('Stability Score')
    plt.ylabel('Count')
    
    # Plot points colored by stability
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=stability_scores, cmap='viridis')
    plt.colorbar(scatter, label='Stability Score')
    plt.title('Stability Analysis')
    plt.xlabel('Component 1' if X.shape[1] > 2 else 'Feature 1')
    plt.ylabel('Component 2' if X.shape[1] > 2 else 'Feature 2')
    
    plt.tight_layout()
    plt.show() 