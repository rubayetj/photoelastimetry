"""
Matplotlib configuration for publication-quality figures.

This file contains rcParams settings used across all plotting scripts
in the papers directory.
"""

import matplotlib.pyplot as plt

# IOP textwidth: 435.32716pt / 72 = 6.04 inches

# Publication-quality figure settings
PLOT_PARAMS = {
    "figure.figsize": (6.04, 4.0),  # IOP textwidth aspect ratio
    "font.size": 10,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": True,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 12,
    "lines.linewidth": 1,
    "lines.markersize": 4,
    "image.cmap": "magma",
}

# Color scheme for RGB channels
COLORS = ["#E74C3C", "#2ECC71", "#3498DB"]  # Red, Green, Blue
CHANNEL_NAMES = ["Red", "Green", "Blue"]


def configure_plots():
    """Apply publication-quality plot configuration."""
    plt.rcParams.update(PLOT_PARAMS)
