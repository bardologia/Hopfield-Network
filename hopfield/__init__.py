"""Hopfield Network Library - associative memory, energy analysis, and visualisation."""

from hopfield.network import HopfieldNetwork
from hopfield.energy import EnergyAnalyzer
from hopfield.visualization import HopfieldVisualizer
from hopfield.utils import (
    generate_random_patterns,
    add_noise,
    hamming_distance,
    overlap,
    pattern_to_grid,
)

__version__ = "1.0.0"

__all__ = [
    "HopfieldNetwork",
    "EnergyAnalyzer",
    "HopfieldVisualizer",
    "generate_random_patterns",
    "add_noise",
    "hamming_distance",
    "overlap",
    "pattern_to_grid",
]
