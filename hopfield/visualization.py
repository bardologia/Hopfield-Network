"""Plotting utilities for Hopfield Network analysis."""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from hopfield.network import HopfieldNetwork

_BIPOLAR_CMAP = mcolors.ListedColormap(["#1a1a1a", "#f5f5f5"])
_BIPOLAR_NORM = mcolors.BoundaryNorm([-1, 0, 1], _BIPOLAR_CMAP.N)

# Grayscale-friendly academic palette
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#7f7f7f", "#9467bd", "#8c564b"]
_MARKERS = ["o", "s", "^", "D", "v", "p"]


class HopfieldVisualizer:
    """Visualisation suite for a HopfieldNetwork."""

    def __init__(self, network: "HopfieldNetwork", grid_shape: Tuple[int, int] | None = None) -> None:
        self.network = network
        if grid_shape is None:
            side = int(np.sqrt(network.n_neurons))
            if side * side == network.n_neurons:
                grid_shape = (side, side)
        self.grid_shape = grid_shape

    def show_patterns(self, patterns, titles=None, suptitle=None, figsize=None) -> plt.Figure:
        """Display bipolar patterns as grid images."""
        if isinstance(patterns, np.ndarray) and patterns.ndim == 1:
            patterns = [patterns]
        patterns = [np.asarray(p) for p in patterns]
        n = len(patterns)

        fig, axes = plt.subplots(1, n, figsize=figsize or (2.2 * n, 2.6))
        if n == 1:
            axes = [axes]
        for idx, (ax, pat) in enumerate(zip(axes, patterns)):
            ax.imshow(self._to_grid(pat), cmap=_BIPOLAR_CMAP, norm=_BIPOLAR_NORM, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_linewidth(0.4); spine.set_color("0.5")
            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=9)
        if suptitle:
            fig.suptitle(suptitle, fontsize=11)
        return fig

    def show_recall_sequence(self, state_history, step_indices=None, suptitle="Recall Dynamics", figsize=None):
        """Show state snapshots at selected steps during recall."""
        total = len(state_history)
        if step_indices is None:
            step_indices = np.linspace(0, total - 1, min(8, total), dtype=int).tolist()
        return self.show_patterns(
            [state_history[i] for i in step_indices],
            titles=[f"$t = {i}$" for i in step_indices],
            suptitle=suptitle, figsize=figsize,
        )

    def show_weights(self, figsize=(5.5, 4.5), cmap="RdBu_r") -> plt.Figure:
        """Weight matrix heatmap."""
        fig, ax = plt.subplots(figsize=figsize)
        vmax = np.abs(self.network.weights).max()
        im = ax.imshow(self.network.weights, cmap=cmap, vmin=-vmax, vmax=vmax,
                       aspect="equal", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$w_{ij}$")
        cbar.ax.tick_params(labelsize=8)
        ax.set_title(r"Weight Matrix $W$", fontsize=11)
        ax.set_xlabel(r"Neuron $j$"); ax.set_ylabel(r"Neuron $i$")
        return fig

    def plot_energy_trajectory(self, energy_history, figsize=(5.5, 3.2)) -> plt.Figure:
        """Energy vs update step."""
        fig, ax = plt.subplots(figsize=figsize)
        steps = np.arange(len(energy_history))
        ax.plot(steps, energy_history, "-", color=_PALETTE[0], marker="o", markersize=3,
                markerfacecolor="white", markeredgewidth=0.8, markeredgecolor=_PALETTE[0])
        ax.set_xlabel("Update step")
        ax.set_ylabel(r"$E(\mathbf{s})$")
        ax.set_title("Energy during recall", fontsize=11)
        return fig

    def plot_overlap_trajectory(self, state_history, pattern_labels=None, figsize=(5.5, 3.2)) -> plt.Figure:
        """Overlap with each stored pattern over time."""
        n_patterns = len(self.network.stored_patterns)
        overlaps = np.array([self.network.overlap_with_patterns(s) for s in state_history])
        if pattern_labels is None:
            pattern_labels = [f"Pattern {i}" for i in range(n_patterns)]

        fig, ax = plt.subplots(figsize=figsize)
        for mu in range(n_patterns):
            c = _PALETTE[mu % len(_PALETTE)]
            mk = _MARKERS[mu % len(_MARKERS)]
            ax.plot(range(len(state_history)), overlaps[:, mu], linestyle="-", marker=mk,
                    color=c, markersize=3, markerfacecolor="white", markeredgewidth=0.7,
                    markeredgecolor=c, label=pattern_labels[mu])
        ax.set_xlabel("Update step")
        ax.set_ylabel(r"Overlap $m^\mu$")
        ax.set_title("Pattern overlap during recall", fontsize=11)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(1.0, color="0.7", ls=":", lw=0.6)
        ax.axhline(-1.0, color="0.7", ls=":", lw=0.6)
        ax.legend(fontsize=8, frameon=True, ncol=min(n_patterns, 3))
        return fig

    def show_recall_comparison(self, original, noisy, recalled, figsize=(7, 2.5)):
        """Side-by-side: original -> noisy -> recalled."""
        return self.show_patterns([original, noisy, recalled],
            titles=["Original", "Noisy probe", "Recalled"],
            suptitle="Pattern recall", figsize=figsize)

    def plot_basin_sizes(self, basin_dict, pattern_labels=None, figsize=(5.5, 3.2)) -> plt.Figure:
        """Bar chart of estimated basin of attraction sizes."""
        fig, ax = plt.subplots(figsize=figsize)
        labels, sizes = [], []
        for k, v in basin_dict.items():
            if k == "spurious":
                labels.append("Spurious")
            elif pattern_labels and k < len(pattern_labels):
                labels.append(pattern_labels[k])
            else:
                labels.append(f"Pattern {k}")
            sizes.append(v)
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]
        bars = ax.bar(labels, sizes, color=colors, edgecolor="white", linewidth=0.5, width=0.6)
        ax.set_ylabel("Fraction of probes")
        ax.set_title("Estimated basin of attraction sizes", fontsize=11)
        ax.set_ylim(0, max(sizes) * 1.15 if sizes else 1)
        for bar, s in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{s:.1%}", ha="center", va="bottom", fontsize=7)
        return fig

    def plot_energy_histogram(self, energies, figsize=(5.5, 3.2)) -> plt.Figure:
        """Histogram of state energies."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(energies, bins=40, color=_PALETTE[0], edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.set_xlabel(r"Energy $E$")
        ax.set_ylabel("Count")
        ax.set_title("Energy distribution", fontsize=11)
        return fig

    def plot_capacity_curve(self, pattern_counts, success_rates, n_neurons, figsize=(5.5, 3.5)) -> plt.Figure:
        """Recall success rate vs number of stored patterns."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(pattern_counts, success_rates, "-o", color=_PALETTE[0], markersize=3,
                markerfacecolor="white", markeredgewidth=0.8, markeredgecolor=_PALETTE[0],
                label="Empirical")
        p_max = n_neurons / (2 * np.log(n_neurons)) if n_neurons > 1 else 0
        ax.axvline(p_max, color="0.4", ls="--", lw=0.8,
                   label=rf"$P_{{\max}} \approx {p_max:.1f}$")
        ax.set_xlabel(r"Stored patterns $P$")
        ax.set_ylabel("Recall success rate")
        ax.set_title(f"Storage capacity ($N = {n_neurons}$)", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=True)
        return fig

    def _to_grid(self, pattern: np.ndarray) -> np.ndarray:
        if self.grid_shape:
            return pattern.reshape(self.grid_shape)
        side = int(np.sqrt(len(pattern)))
        if side * side == len(pattern):
            return pattern.reshape(side, side)
        return pattern.reshape(1, -1)
