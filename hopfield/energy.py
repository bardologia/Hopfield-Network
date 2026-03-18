"""Energy landscape analysis for Hopfield Networks."""

from __future__ import annotations
import numpy as np
from itertools import product as iter_product
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from hopfield.network import HopfieldNetwork


class EnergyAnalyzer:
    """Analyse energy landscape, attractors, and basins of a HopfieldNetwork."""

    def __init__(self, network: "HopfieldNetwork") -> None:
        self.network = network

    def compute_energy(self, state: np.ndarray) -> float:
        return self.network.energy(state)

    def energy_along_path(self, state_history: List[np.ndarray]) -> np.ndarray:
        """Energy at each state in a trajectory."""
        return np.array([self.network.energy(s) for s in state_history])

    def stored_pattern_energies(self) -> np.ndarray:
        """Energy of every stored pattern."""
        return np.array([self.network.energy(p) for p in self.network.stored_patterns])

    def is_fixed_point(self, state: np.ndarray) -> bool:
        """True if state is stable under one synchronous update."""
        h = self.network.weights @ state
        return bool(np.array_equal(np.where(h >= 0, 1.0, -1.0), state))

    def find_spurious_states(self, n_probes: int = 500) -> List[np.ndarray]:
        """Search for spurious attractors by probing with random states."""
        spurious: List[np.ndarray] = []
        known = set()
        for p in self.network.stored_patterns:
            known.add(tuple(p.astype(int)))
            known.add(tuple((-p).astype(int)))

        for _ in range(n_probes):
            probe = np.random.choice([-1, 1], size=self.network.n_neurons).astype(float)
            final, _ = self.network.recall(probe, mode="async", max_steps=50)
            key = tuple(final.astype(int))
            if key not in known:
                spurious.append(final.copy())
                known.add(key)
        return spurious

    def enumerate_all_energies(self) -> Tuple[np.ndarray, np.ndarray]:
        """Enumerate all 2^N states and energies (N <= 18 only)."""
        n = self.network.n_neurons
        if n > 18:
            raise RuntimeError(f"N={n} too large for full enumeration.")
        states = np.array(list(iter_product([-1, 1], repeat=n)), dtype=float)
        energies = np.array([self.network.energy(s) for s in states])
        return states, energies

    def find_all_minima(self) -> Tuple[np.ndarray, np.ndarray]:
        """Find all local minima by full enumeration (small N)."""
        states, energies = self.enumerate_all_energies()
        mask = [self.is_fixed_point(s) for s in states]
        return states[mask], energies[mask]

    def estimate_basin_sizes(self, n_probes: int = 1000) -> dict:
        """Estimate basin sizes. Returns pattern_idx -> fraction, plus 'spurious'."""
        counts: dict = {i: 0 for i in range(len(self.network.stored_patterns))}
        counts["spurious"] = 0
        patterns = np.array(self.network.stored_patterns)

        for _ in range(n_probes):
            probe = np.random.choice([-1, 1], size=self.network.n_neurons).astype(float)
            final, _ = self.network.recall(probe, mode="async", max_steps=100)
            overlaps = (patterns @ final) / self.network.n_neurons
            matched = False
            for idx, ov in enumerate(overlaps):
                if abs(ov) > 0.99:
                    counts[idx] += 1
                    matched = True
                    break
            if not matched:
                counts["spurious"] += 1

        return {k: v / n_probes for k, v in counts.items()}

    def __repr__(self) -> str:
        return f"EnergyAnalyzer(network={self.network!r})"
