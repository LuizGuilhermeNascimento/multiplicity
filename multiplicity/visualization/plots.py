from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


def plot_prob_dist(
    probs: Dict[str, List[float]], 
    title: str = "Probability Distribution",
    reference_scores: Optional[List[float]] = None,
    reference_label: str = "Reference",
    save_path: Optional[str] = None
) -> None:
    """Plot probability distribution for each fold with optional reference distribution.
    
    Args:
        probs: Dictionary mapping fold names to probability lists
        title: Plot title
        reference_scores: Optional list of reference probabilities to compare against
        reference_label: Label for the reference distribution
        save_path: Optional path to save the plot instead of displaying it
    """
    sns.reset_defaults()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot reference distribution first if provided
    if reference_scores is not None:
        sns.kdeplot(
            reference_scores, 
            label=reference_label, 
            ax=ax,
            linestyle='--',
            color='black',
            linewidth=2,
            fill=False
        )
    
    # Plot fold distributions
    for fold, probabilities in probs.items():
        sns.kdeplot(probabilities, label=fold, fill=True, ax=ax, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
    sns.reset_defaults()