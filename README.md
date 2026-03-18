# Discrete Hopfield Networks for Associative Memory: Implementation and Analysis

> **Course Project — Boltzmann Machines & Energy-Based Models**
>
> A modular Python implementation of discrete Hopfield networks with support for Hebbian and Storkey learning rules, asynchronous and synchronous recall dynamics, energy landscape analysis, and publication-quality visualisation.

---

## Table of Contents

1. [Abstract](#abstract)
2. [Model Definition & Pattern Storage](#1-model-definition--pattern-storage)
3. [Recall Dynamics & Noise Robustness](#2-recall-dynamics--noise-robustness)
4. [Energy Function & Trajectory Analysis](#3-energy-function--trajectory-analysis)
5. [Weight Matrix Structure](#4-weight-matrix-structure)
6. [Pattern Overlap Dynamics](#5-pattern-overlap-dynamics)
7. [Asynchronous vs. Synchronous Updates](#6-asynchronous-vs-synchronous-updates)
8. [Storage Capacity](#7-storage-capacity)
9. [Hebbian vs. Storkey Learning](#8-hebbian-vs-storkey-learning)
10. [Basin of Attraction Analysis](#9-basin-of-attraction-analysis)
11. [Spurious States](#10-spurious-states)
12. [Letter Pattern Recognition](#11-letter-pattern-recognition)
13. [Convergence Speed Analysis](#12-convergence-speed-analysis)
14. [Project Structure & Usage](#project-structure--usage)
15. [References](#references)

---

## Abstract

This repository provides a self-contained implementation of the **discrete Hopfield network** (Hopfield, 1982), a canonical energy-based model that functions as a content-addressable (associative) memory. The codebase supports two learning rules—classical Hebbian learning and the Storkey incremental rule—as well as both asynchronous (sequential) and synchronous (parallel) update dynamics. Accompanying analysis tools enable full energy landscape enumeration (for small networks), spurious state detection, basin-of-attraction estimation, and storage capacity benchmarking.

All figures below were generated from the interactive notebook with $N = 50$ neurons, $P = 6$ stored patterns, and seed $= 42$. At this configuration the network operates at $\approx 94\%$ of its theoretical capacity, producing non-trivial dynamics that reveal the network's true behaviour under stress.

---

## 1. Model Definition & Pattern Storage

A Hopfield network consists of $N$ binary neurons with states $s_i \in \{-1, +1\}$, interconnected by a symmetric weight matrix $W$ satisfying $w_{ij} = w_{ji}$ and $w_{ii} = 0$.

### Hebbian Learning Rule

Given $P$ pattern vectors $\boldsymbol{\xi}^{\mu} \in \{-1,+1\}^N$, the weight matrix is constructed via the outer-product rule:

$$
w_{ij} = \frac{1}{N} \sum_{\mu=1}^{P} \xi_i^{\mu}\, \xi_j^{\mu}, \qquad w_{ii} = 0.
$$

In matrix form:

$$
W = \frac{1}{N} \sum_{\mu=1}^{P} \boldsymbol{\xi}^{\mu} \left(\boldsymbol{\xi}^{\mu}\right)^\top, \quad \mathrm{diag}(W) = 0.
$$

Each stored pattern becomes a fixed point (attractor) of the network dynamics — a local minimum of the energy landscape.

<p align="center">
  <img src="figures/01_stored_patterns.png" width="85%" />
</p>

---

## 2. Recall Dynamics & Noise Robustness

Retrieval from a corrupted probe $\tilde{\mathbf{s}}$ (obtained by flipping a fraction $\eta$ of bits) proceeds by iterating the update rule:

$$
s_i(t+1) = \mathrm{sgn}\!\left(\sum_{j=1}^{N} w_{ij}\, s_j(t)\right)
$$

until the state vector $\mathbf{s}(t)$ converges to a fixed point. Recall quality is measured by the **overlap** between the converged state and the target pattern:

$$
m^{\mu} = \frac{1}{N} \sum_{i=1}^{N} \xi_i^{\mu}\, s_i \;\in [-1,\,1].
$$

A value of $m^{\mu} = 1$ indicates perfect recall; $m^{\mu} = -1$ retrieves the complement; $m^{\mu} \approx 0$ means the state is uncorrelated with that pattern.

With the network near capacity ($P/P_{\max} \approx 94\%$), recall from noise levels $\eta = 20\%$–$50\%$ becomes genuinely challenging — cross-talk from other patterns competes with the target signal:

<p align="center">
  <img src="figures/02_recall_noise_levels.png" width="70%" />
</p>

---

## 3. Energy Function & Trajectory Analysis

The Lyapunov (energy) function is a scalar function of the state vector:

$$
E(\mathbf{s}) = -\frac{1}{2} \sum_{i \neq j} w_{ij}\, s_i\, s_j = -\frac{1}{2}\, \mathbf{s}^\top W\, \mathbf{s}.
$$

Under asynchronous updates, $E$ is **non-increasing** at every step. When neuron $i$ flips from $s_i$ to $s_i'$, the energy change is:

$$
\Delta E = -(s_i' - s_i) \sum_{j=1}^{N} w_{ij}\, s_j = -(s_i' - s_i)\, h_i
$$

where $h_i = \sum_j w_{ij} s_j$ is the local field. Since the update rule sets $s_i' = \mathrm{sgn}(h_i)$, we have $(s_i' - s_i) \cdot h_i \geq 0$, hence $\Delta E \leq 0$. This guarantees convergence to a local minimum.

Each stored pattern $\boldsymbol{\xi}^{\mu}$ sits at a local minimum. The trajectory below shows energy decreasing over multiple steps as the network recovers from 35% noise:

<p align="center">
  <img src="figures/03_energy_trajectory.png" width="60%" />
</p>

---

## 4. Weight Matrix Structure

The weight matrix $W \in \mathbb{R}^{N \times N}$ encodes the network's memory. Its key structural properties:

- **Symmetric**: $w_{ij} = w_{ji}$ (undirected connections)
- **Zero diagonal**: $w_{ii} = 0$ (no self-connections)
- **Scale**: Off-diagonal entries scale as $w_{ij} \sim \mathcal{O}(P/N)$

<p align="center">
  <img src="figures/04_weight_matrix.png" width="55%" />
</p>

---

## 5. Pattern Overlap Dynamics

During recall, the overlap vector $\mathbf{m}(t) = (m^1(t), \dots, m^P(t))^\top$ tracks alignment with every stored pattern:

$$
m^{\mu}(t) = \frac{1}{N} \sum_{i=1}^{N} s_i(t)\, \xi_i^{\mu} = \frac{1}{N}\, \mathbf{s}(t) \cdot \boldsymbol{\xi}^{\mu} \quad \in [-1,\,1].
$$

The target pattern's overlap rises toward $+1$ while competing patterns remain near $0$. Near capacity, this separation is gradual — the network must work through cross-talk noise over several update sweeps:

<p align="center">
  <img src="figures/05_overlap_dynamics.png" width="60%" />
</p>

---

## 6. Asynchronous vs. Synchronous Updates

Two update schedules are implemented:

**Asynchronous** — Pick neuron $i$ at random, compute $h_i = \sum_j w_{ij}\, s_j$, update $s_i \leftarrow \mathrm{sgn}(h_i)$. Repeat for all neurons in a random permutation. Guarantees $\Delta E \leq 0$ at every single-neuron flip, so convergence to a fixed point is assured.

**Synchronous** — Update the entire state vector in one step:

$$
\mathbf{s}(t+1) = \mathrm{sgn}\!\left(W\, \mathbf{s}(t)\right)
$$

This is faster per iteration but does **not** guarantee monotonic energy descent and may produce period-2 limit cycles.

<p align="center">
  <img src="figures/06_async_vs_sync_dynamics.png" width="85%" />
</p>

Quantitative comparison across noise levels — overlap, convergence speed, wall time, and convergence rate:

<p align="center">
  <img src="figures/07_async_vs_sync_comparison.png" width="85%" />
</p>

---

## 7. Storage Capacity

The load ratio $\alpha = P/N$ determines whether the network can reliably store and retrieve patterns.

The theoretical capacity under Hebbian learning is:

$$
P_{\max} \approx \frac{N}{2 \ln N}
$$

beyond which cross-talk noise causes catastrophic recall failure (Amit *et al.*, 1985). At the critical load $\alpha_c \approx 0.138$, overlap with all stored patterns collapses to $\approx 0$.

Empirical recall success rate vs. number of stored patterns ($N = 150$). The dashed line marks $P_{\max}$:

<p align="center">
  <img src="figures/08_storage_capacity.png" width="60%" />
</p>

---

## 8. Hebbian vs. Storkey Learning

The **Storkey incremental rule** (1997) updates weights pattern-by-pattern, accounting for cross-talk via local fields:

$$
w_{ij}^{(\mu)} = w_{ij}^{(\mu-1)} + \frac{1}{N}\!\left( \xi_i^{\mu}\xi_j^{\mu} - \xi_i^{\mu}\,h_{j,i}^{(\mu-1)} - h_{i,j}^{(\mu-1)}\,\xi_j^{\mu} \right),
$$

where $h_{i,j}^{(\mu-1)} = \sum_{k \neq j} w_{ik}^{(\mu-1)}\,\xi_k^{\mu}$ is the local field at neuron $i$ excluding the contribution of neuron $j$.

This raises the capacity bound to:

$$
P_{\max}^{\text{Storkey}} \approx \frac{N}{\sqrt{2 \ln N}}
$$

| Rule | Capacity | Mechanism |
|---|---|---|
| **Hebbian** | $\approx 0.138\, N$ | Single outer-product sum |
| **Storkey** | $\approx N / \sqrt{2 \ln N}$ | Incremental, subtracts cross-talk via local fields |

The Storkey rule sustains high recall accuracy well beyond the Hebbian capacity limit:

<p align="center">
  <img src="figures/09_hebbian_vs_storkey.png" width="65%" />
</p>

---

## 9. Basin of Attraction Analysis

Each stored pattern $\boldsymbol{\xi}^{\mu}$ has a **basin of attraction** — the set of all initial states $\mathbf{s}(0)$ that converge to it under the dynamics. Basin fractions are estimated by Monte-Carlo probing ($2\,000$ random initial states):

- Larger basin → more robust recall
- Basin fractions shrink as the load ratio $\alpha = P/N$ increases
- States converging to non-stored vectors are counted as "spurious"

<p align="center">
  <img src="figures/10_basin_of_attraction.png" width="60%" />
</p>

---

## 10. Spurious States

Not all fixed points correspond to stored patterns. The energy landscape contains additional attractors:

| Spurious type | Form |
|---|---|
| **Negatives** $-\boldsymbol{\xi}^{\mu}$ | Bit-flipped stored pattern (also a fixed point by symmetry of $W$) |
| **Mixture states** | $\mathrm{sgn}\!\bigl(\pm\boldsymbol{\xi}^1 \pm \boldsymbol{\xi}^2 \pm \boldsymbol{\xi}^3\bigr)$, odd combinations of stored patterns |
| **Spin-glass states** | Unstructured minima with $m^{\mu} \approx 0\;\;\forall\,\mu$ |

The number of spurious states grows rapidly with $N$ and $P$. Below: stored patterns (top) vs. discovered spurious attractors (bottom) with their overlap vectors:

<p align="center">
  <img src="figures/11_spurious_states.png" width="85%" />
</p>

---

## 11. Letter Pattern Recognition

A practical demonstration: hand-crafted $5 \times 5$ letter bitmaps stored as pattern vectors $\boldsymbol{\xi}^{\mu} \in \{-1,+1\}^{25}$ and recalled from 25% noise corruption. Per-letter energy trajectories confirm convergence.

<p align="center">
  <img src="figures/12_letter_patterns_stored.png" width="50%" />
</p>

<p align="center">
  <img src="figures/13_letter_recall.png" width="85%" />
</p>

---

## 12. Convergence Speed Analysis

The number of update steps $T$ to reach a fixed point depends on both the noise level $\eta$ and the load ratio $\alpha = P/N$:

- Higher noise → probe starts further from any attractor → more steps
- Higher load → shallower basins and more cross-talk → slower or failed convergence

<p align="center">
  <img src="figures/14_convergence_heatmap.png" width="65%" />
</p>

---

## Project Structure & Usage

```
Hopfield Network/
├── hopfield/                     # Python package
│   ├── __init__.py               # Public API exports
│   ├── network.py                # HopfieldNetwork class (Hebbian & Storkey learning, recall)
│   ├── energy.py                 # EnergyAnalyzer — landscape analysis, attractor search
│   ├── visualization.py          # HopfieldVisualizer — publication-quality plots
│   └── utils.py                  # Pattern generators, noise injection, similarity metrics
├── notebooks/
│   └── hopfield_demo.ipynb       # Interactive experiments & figures
├── figures/                      # Pre-rendered experiment figures
├── scripts/
│   └── export_figures.py         # Extract PNGs from executed notebook
├── requirements.txt              # Pinned Python dependencies
└── README.md                     # This document
```

### Implementation Summary

| Class / Module | Key Methods |
|---|---|
| `HopfieldNetwork` | `train()`, `train_storkey()`, `recall()`, `energy()`, `overlap_with_patterns()` |
| `EnergyAnalyzer` | `stored_pattern_energies()`, `find_spurious_states()`, `estimate_basin_sizes()`, `enumerate_all_energies()` |
| `HopfieldVisualizer` | Pattern grids, weight heatmaps, energy/overlap trajectories, capacity curves |
| `utils` | `generate_random_patterns`, `make_letter_patterns`, `add_noise`, `overlap`, `hamming_distance` |

### Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook notebooks/hopfield_demo.ipynb
```

### Programmatic Usage

```python
import numpy as np
from hopfield import HopfieldNetwork, EnergyAnalyzer, HopfieldVisualizer
from hopfield.utils import generate_random_patterns, add_noise

net = HopfieldNetwork(n_neurons=50)
patterns = generate_random_patterns(n_patterns=6, n_neurons=50, seed=42)
net.train(patterns)

probe = add_noise(patterns[0], noise_level=0.35)
recalled, info = net.recall(probe, mode="async", record_history=True)

analyzer = EnergyAnalyzer(net)
print("Pattern energies:", analyzer.stored_pattern_energies())

viz = HopfieldVisualizer(net, grid_shape=(5, 10))
viz.show_recall_comparison(patterns[0], probe, recalled)
viz.plot_energy_trajectory(info["energy_history"])
```

---

## References

1. Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences*, 79(8), 2554–2558.
2. Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1985). Storing infinite numbers of patterns in a spin-glass model of neural networks. *Physical Review Letters*, 55(14), 1530–1533.
3. Storkey, A. J. (1997). Increasing the capacity of a Hopfield network without sacrificing functionality. In *Artificial Neural Networks — ICANN '97* (pp. 451–456). Springer.
4. Hertz, J., Krogh, A., & Palmer, R. G. (1991). *Introduction to the Theory of Neural Computation*. Addison-Wesley.
5. Amit, D. J. (1989). *Modeling Brain Function: The World of Attractor Neural Networks*. Cambridge University Press.

---

<sub>Last updated: March 2026</sub>
