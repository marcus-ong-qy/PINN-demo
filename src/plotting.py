"""
Plotting utilities for PINN visualization
"""

import matplotlib.pyplot as plt


def plot_results(t, y_true, y_pred, title='PINN Results', t_samples=None, y_noisy=None):
    """
    Plot comparison between true and predicted positions.

    Args:
        t: Time array (full resolution)
        y_true: True position values
        y_pred: Predicted position values
        title: Plot title
        t_samples: Optional time samples for noisy data points
        y_noisy: Optional noisy data points
    """
    plt.figure(figsize=(8, 5))
    plt.plot(t, y_true, label='True Position', color='blue', linewidth=2)
    plt.plot(t, y_pred, label='Predicted Position', color='red', linewidth=2, linestyle='--')

    # Add noisy data points if provided
    if t_samples is not None and y_noisy is not None:
        plt.scatter(t_samples, y_noisy, color='green', s=30, alpha=0.6, label='Noisy Data', zorder=5)

    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Position (m)', fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comparison(t, y_true, predictions, titles, t_samples=None, y_noisy=None):
    """
    Plot side-by-side comparison of multiple model predictions.

    Args:
        t: Time array (full resolution)
        y_true: True position values
        predictions: List of prediction arrays (one per model)
        titles: List of titles (one per model)
        t_samples: Optional time samples for noisy data points
        y_noisy: Optional noisy data points
    """
    num_plots = len(predictions)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

    # Handle case of single plot
    if num_plots == 1:
        axes = [axes]

    # Plot each model
    for ax, y_pred_model, title in zip(axes, predictions, titles):
        ax.plot(t, y_true, label='True Position', color='blue', linewidth=2)
        ax.plot(t, y_pred_model, label='Predicted Position', color='red', linewidth=2, linestyle='--')

        if t_samples is not None and y_noisy is not None:
            ax.scatter(t_samples, y_noisy, color='green', s=30, alpha=0.6, label='Noisy Data', zorder=5)

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Position (m)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
