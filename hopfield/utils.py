"""Utility functions - pattern generation, noise, and similarity metrics."""

from __future__ import annotations
import numpy as np
from typing import Tuple


def generate_random_patterns(n_patterns: int, n_neurons: int, seed: int | None = None) -> np.ndarray:
    """Generate random bipolar (+/-1) patterns of shape (P, N)."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(n_patterns, n_neurons))


def make_letter_patterns() -> dict[str, np.ndarray]:
    """Hand-crafted 5x5 letter bitmaps as bipolar vectors."""
    letters = {
        "A": [-1,1,1,1,-1, 1,-1,-1,-1,1, 1,1,1,1,1, 1,-1,-1,-1,1, 1,-1,-1,-1,1],
        "C": [1,1,1,1,1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,1,1,1,1],
        "H": [1,-1,-1,-1,1, 1,-1,-1,-1,1, 1,1,1,1,1, 1,-1,-1,-1,1, 1,-1,-1,-1,1],
        "I": [1,1,1,1,1, -1,-1,1,-1,-1, -1,-1,1,-1,-1, -1,-1,1,-1,-1, 1,1,1,1,1],
        "L": [1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,1,1,1,1],
        "T": [1,1,1,1,1, -1,-1,1,-1,-1, -1,-1,1,-1,-1, -1,-1,1,-1,-1, -1,-1,1,-1,-1],
        "X": [1,-1,-1,-1,1, -1,1,-1,1,-1, -1,-1,1,-1,-1, -1,1,-1,1,-1, 1,-1,-1,-1,1],
        "O": [-1,1,1,1,-1, 1,-1,-1,-1,1, 1,-1,-1,-1,1, 1,-1,-1,-1,1, -1,1,1,1,-1],
    }
    return {k: np.array(v, dtype=float) for k, v in letters.items()}


def make_shape_patterns(size: int = 10) -> dict[str, np.ndarray]:
    """Simple geometric shapes on a size x size grid as flat bipolar vectors."""
    patterns = {}

    grid = -np.ones((size, size)); grid[size // 2, :] = 1
    patterns["horizontal_bar"] = grid.flatten()

    grid = -np.ones((size, size)); grid[:, size // 2] = 1
    patterns["vertical_bar"] = grid.flatten()

    grid = -np.ones((size, size)); grid[size // 2, :] = 1; grid[:, size // 2] = 1
    patterns["cross"] = grid.flatten()

    grid = np.array([[-1 if (i+j) % 2 == 0 else 1 for j in range(size)] for i in range(size)], dtype=float)
    patterns["checkerboard"] = grid.flatten()

    grid = -np.ones((size, size))
    for i in range(size): grid[i, i] = 1
    patterns["diagonal"] = grid.flatten()

    grid = -np.ones((size, size))
    grid[0, :] = 1; grid[-1, :] = 1; grid[:, 0] = 1; grid[:, -1] = 1
    patterns["border"] = grid.flatten()

    return patterns


def add_noise(pattern: np.ndarray, noise_level: float = 0.1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Flip a fraction of bits in a bipolar pattern."""
    if rng is None:
        rng = np.random.default_rng()
    noisy = pattern.copy()
    n_flip = int(round(noise_level * len(pattern)))
    noisy[rng.choice(len(pattern), size=n_flip, replace=False)] *= -1
    return noisy


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Number of differing positions between two bipolar vectors."""
    return int(np.sum(a != b))


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised overlap m = (1/N) a.b in [-1, 1]."""
    return float(np.dot(a, b) / len(a))


def pattern_to_grid(pattern: np.ndarray, shape: Tuple[int, int] | None = None) -> np.ndarray:
    """Reshape flat pattern to 2-D grid for display."""
    if shape is None:
        side = int(np.sqrt(len(pattern)))
        if side * side != len(pattern):
            raise ValueError("Pattern length is not a perfect square; provide shape.")
        shape = (side, side)
    return pattern.reshape(shape)
