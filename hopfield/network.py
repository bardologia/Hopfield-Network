"""Core Hopfield Network - Hebbian/Storkey learning, async/sync recall, energy."""

from __future__ import annotations
import numpy as np
from typing import List, Tuple


class HopfieldNetwork:
    """Discrete Hopfield Network for associative memory."""

    def __init__(self, n_neurons: int) -> None:
        if n_neurons < 1:
            raise ValueError("n_neurons must be positive.")
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
        self.stored_patterns: List[np.ndarray] = []

    def train(self, patterns: np.ndarray) -> None:
        """Store patterns using the Hebbian outer-product rule."""
        patterns = np.atleast_2d(patterns)
        p, n = patterns.shape
        if n != self.n_neurons:
            raise ValueError(f"Pattern length {n} != network size {self.n_neurons}.")
        self.weights = (patterns.T @ patterns) / n
        np.fill_diagonal(self.weights, 0)
        self.stored_patterns = [patterns[i] for i in range(p)]

    def train_storkey(self, patterns: np.ndarray) -> None:
        """Store patterns using the Storkey incremental learning rule."""
        patterns = np.atleast_2d(patterns)
        p, n = patterns.shape
        if n != self.n_neurons:
            raise ValueError(f"Pattern length {n} != network size {self.n_neurons}.")
        self.weights = np.zeros((n, n))
        for mu in range(p):
            xi = patterns[mu]
            h = self.weights @ xi
            # H_ij = h_i - w_ij * xi_j  (local field at i excluding j)
            H = h[:, None] - self.weights * xi[None, :]
            self.weights += (np.outer(xi, xi) - xi[:, None] * H.T - H * xi[None, :]) / n
            np.fill_diagonal(self.weights, 0)
        self.stored_patterns = [patterns[i] for i in range(p)]

    def recall(self, state: np.ndarray, mode: str = "async",
               max_steps: int = 100, record_history: bool = False) -> Tuple[np.ndarray, dict]:
        """Run network dynamics. Returns (final_state, info)."""
        state = np.array(state, dtype=np.float64).copy()
        if state.shape[0] != self.n_neurons:
            raise ValueError("State size does not match the network.")

        info: dict = {"steps": 0, "converged": False}
        if record_history:
            info["energy_history"] = [self.energy(state)]
            info["state_history"] = [state.copy()]

        for step in range(1, max_steps + 1):
            prev_state = state.copy()
            if mode == "async":
                for i in np.random.permutation(self.n_neurons):
                    state[i] = 1.0 if self.weights[i] @ state >= 0 else -1.0
            elif mode == "sync":
                state = np.where(self.weights @ state >= 0, 1.0, -1.0)
            else:
                raise ValueError(f"Unknown mode '{mode}'. Use 'async' or 'sync'.")

            info["steps"] = step
            if record_history:
                info["energy_history"].append(self.energy(state))
                info["state_history"].append(state.copy())
            if np.array_equal(state, prev_state):
                info["converged"] = True
                break

        return state, info

    def energy(self, state: np.ndarray) -> float:
        """E(s) = -1/2 s^T W s"""
        return -0.5 * state @ self.weights @ state

    def overlap_with_patterns(self, state: np.ndarray) -> np.ndarray:
        """Overlap m^mu = (1/N) xi^mu . s for each stored pattern."""
        if not self.stored_patterns:
            return np.array([])
        patterns = np.array(self.stored_patterns)
        return (patterns @ state) / self.n_neurons

    @property
    def theoretical_capacity(self) -> float:
        """P_max ~ N / (2 ln N)"""
        n = self.n_neurons
        return n / (2 * np.log(n)) if n > 1 else 0.0

    def __repr__(self) -> str:
        p = len(self.stored_patterns)
        return (f"HopfieldNetwork(n_neurons={self.n_neurons}, "
                f"stored_patterns={p}, capacity~{self.theoretical_capacity:.1f})")
