import numpy as np
from itertools import combinations


def rashomon_set(models_dict: dict, X: np.ndarray, y: np.ndarray, epsilon=0.05) -> dict:
    """Identify the Rashomon set of models.

    Args:
        models_dict: Dictionary of model names and instances
        X: Input data
        y: Target values
        epsilon: Tolerance for error difference

    Returns:
        dict: Rashomon set of models
    """
    loss_dict = {model_name: np.abs(y - model.predict(X)) for model_name, model in models_dict.items()}
    error_ref = sorted(list(loss_dict.values()))[0]

    rashomon_set = {model_name: model for model_name, model in models_dict.items() if loss_dict[model_name] <= (1 + epsilon) * error_ref}
    return rashomon_set


def arbitrariness(models: list, X: np.ndarray) -> float:
    """ Calculate the arbitrariness of a set of models.

    Args:
        models: List of models
        X: Input data

    Returns:
        float: Arbitrariness score
    """
    if (len(models) == 1):
        return 0
    
    preds = np.array([model.predict(X) for model in models]).T
    disagree_samples = np.sum([len(np.unique(sample)) > 1 for sample in preds])

    return disagree_samples / preds.shape[0]


def pairwise_disagreement(models: list, X: np.ndarray) -> float:
    """ Calculate the pairwise disagreement of a set of models.

    Args:
        models: List of models
        X: Input data

    Returns:
        float: Pairwise disagreement score
    """
    if (len(models) == 1):
        return 0

    preds = np.array([model.predict(X) for model in models]).T

    disagree_by_sample = []

    for sample in preds:
        disag = 0
        for (model1, model2) in combinations(sample, 2):
            if model1 != model2:
                disag += 1

        disag *= 2 / (preds.shape[1] * (preds.shape[1] - 1))

        disagree_by_sample.append(disag)

    return np.mean(disagree_by_sample)