# filepath: /examples/plot_01_synthetic_decomposition.py
#
# Copyright (c) 2026 Nikhil Sunder
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
r"""
A series with a known decomposition
===================================

The model is handed a series built from the parts it is meant to find: a
linear trend, a slow oscillation whose frequency rises through the sample
and whose amplitude steps up two-thirds of the way through, a faster
oscillation at a fixed frequency, and noise. The example fits the default
filtered alignment, reads every path off the result, checks the identity
that ties them together, and then asks the only question that matters
about a decomposition with a known answer: how did the network divide the
series between its trend head and its components?

The answer, on a short schedule, is the behaviour the user guide's
nuances page describes: the trend head reconstructs the series and the
components carry little. The example is therefore less a demonstration
that the model recovers planted sinusoids than a reading of what it does
instead, with the tools that make that reading quick.
"""

# %%
# The series
# ----------
# Four hundred and eighty steps. The slow component's frequency runs from
# 0.08 to 0.20 radians per step (periods of 79 down to 31 steps), its
# amplitude from 0.5 to 1.5 around step 300; the fast component sits at
# 0.60 (a period of about 10 steps).

import matplotlib.pyplot as plt
import numpy as np

from ns_sdn.spectral_decomposition import NSSDN, Stage

rng = np.random.default_rng(7)
n = 480
t = np.arange(n, dtype=float)
omega_slow = 0.08 + 0.00025 * t
amp_slow = 1.0 + 0.5 * np.tanh((t - 300) / 30)
slow = amp_slow * np.sin(np.cumsum(omega_slow))
fast = 0.4 * np.sin(0.60 * t)
trend_true = 0.004 * t
y = trend_true + slow + fast + 0.05 * rng.standard_normal(n)

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(t, y, lw=0.9, label="series")
ax.plot(t, trend_true, lw=1.2, ls="--", label="planted trend")
ax.set_xlabel("step")
ax.legend(loc="upper left")
ax.set_title("A trend, a chirp with a step in amplitude, a fast sinusoid, noise")
plt.tight_layout()

# %%
# Fit
# ---
# Three components, a 32-dimensional state and one stage of three hundred
# Adam steps: a fraction of the published configuration, chosen so the
# page builds in under a minute. ``seed`` fixes the initialization.

res = NSSDN(y, k=3, state_dim=32, schedule=(Stage(5e-3, 300),), seed=0).fit()
print(res.summary())

# %%
# The identity
# ------------
# Trend plus the sum of the component contributions is the emission,
# exactly, and the emission reconstructs the series closely: under the
# filtered alignment the state at ``t`` has seen ``y[t]``.

rebuilt = res.trend + res.components.sum(axis=1)
print("identity holds:", bool(np.allclose(rebuilt, res.emission, atol=1e-5)))
print(f"emission RMSE: {np.sqrt(np.mean(res.reconstruction_resid**2)):.4f}")
print(f"one-step RMSE: {np.sqrt(res.sigma2):.4f}")

# %%
# Where the variance went
# -----------------------
# The question with a known answer. The planted oscillations account for
# most of the series' variance; the decomposition's ``oscillation`` is
# what the components account for.

parts = {
    "series": y,
    "planted oscillation": slow + fast,
    "fitted trend": res.trend,
    "fitted oscillation": res.oscillation,
}
for name, values in parts.items():
    print(f"{name:>22s} variance {np.var(values):8.4f}")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t, y, lw=0.8, label="series")
axes[0].plot(t, res.emission, lw=0.8, label="emission")
axes[0].plot(t, res.trend, lw=1.4, label="trend head $B_t$")
axes[0].legend(loc="upper left")
axes[0].set_title("The trend head tracks the series; the emission is trend plus components")
axes[1].plot(t, slow + fast, lw=0.8, label="planted oscillation")
axes[1].plot(t, res.oscillation, lw=1.2, label="components' contribution")
axes[1].legend(loc="upper left")
axes[1].set_xlabel("step")
plt.tight_layout()

# %%
# The component paths
# -------------------
# Gated amplitude and instantaneous frequency per component, the paper's
# Figures 4 and 5, with the planted frequencies drawn through the second
# panel. The frequencies are read in radians per step.

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for j in range(res.k):
    axes[0].plot(t, res.amplitude[:, j], label=f"component {j + 1}")
    axes[1].plot(t, res.frequency[:, j], label=f"component {j + 1}")
    axes[2].plot(t, res.gate[:, j], label=f"component {j + 1}")
axes[1].plot(t, omega_slow, color="k", ls="--", lw=1, label="planted slow")
axes[1].axhline(0.60, color="k", ls=":", lw=1, label="planted fast")
axes[0].set_ylabel("gated amplitude")
axes[1].set_ylabel("frequency (rad/step)")
axes[2].set_ylabel("gate")
axes[2].set_xlabel("step")
for ax in axes:
    ax.legend(loc="upper right", fontsize=8, ncol=2)
axes[0].set_title("Amplitude, frequency and gate paths against the planted frequencies")
plt.tight_layout()

print("mean frequency per component:", res.frequency.mean(axis=0).round(3))
print("power share per component:   ", res.power_share.round(3))

# %%
# The spectral state
# ------------------
# The same reading in permutation-invariant coordinates: the centroid and
# dispersion of the power-weighted frequency measure, and how many
# components it effectively uses.

state = res.spectral_state()
print(state.summary())

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(t, state.centroid, label="centroid")
axes[0].fill_between(
    t,
    state.centroid - state.dispersion,
    state.centroid + state.dispersion,
    alpha=0.25,
    label="centroid $\\pm$ dispersion",
)
axes[0].set_ylabel("frequency (rad/step)")
axes[0].legend(loc="upper right")
axes[1].plot(t, state.effective_count, label="effective count")
axes[1].plot(t, state.participation_ratio, label="participation ratio")
axes[1].set_ylim(0.9, res.k + 0.1)
axes[1].set_ylabel("components")
axes[1].set_xlabel("step")
axes[1].legend(loc="upper right")
axes[0].set_title("Spectral-state coordinates over the sample")
plt.tight_layout()

# %%
# The predicted alignment
# -----------------------
# Trained to predict ``y[t]`` from the state after ``y[t - 1]`` instead,
# the network is a forecaster and the one-step error is the thing it
# minimizes. The random walk is the benchmark: it predicts each value by
# the one before it, which on a smooth series is already most of the way.

fwd = NSSDN(y, k=3, state_dim=32, alignment="predicted", schedule=(Stage(5e-3, 300),), seed=0).fit()
naive = y[:-1]
target = y[1:]
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))  # noqa: E731
print(f"one-step RMSE, filtered:  {rmse(target, res.fittedvalues):.4f}")
print(f"one-step RMSE, predicted: {rmse(target, fwd.fittedvalues):.4f}")
print(f"one-step RMSE, random walk: {rmse(target, naive):.4f}")
print(f"predicted alignment, fitted oscillation variance: {np.var(fwd.oscillation):.4f}")

# %%
# What this shows
# ---------------
# On this schedule the trend head has done the work under both alignments:
# it reconstructs the series under the filtered one and reproduces the
# previous value under the predicted one, while the components' share of
# the variance stays near zero. One component's frequency does settle close
# to the planted 0.60, which is also near the softplus initialization, but
# its gated amplitude is a few hundredths; nothing finds the 0.08-to-0.20
# chirp, whose frequency the drift penalty would have to be pulled a long
# way to reach. The paths are smooth and the identities hold, which is
# what the package guarantees; whether the components carry the
# oscillation is a property of the fit, not of the API, and the tools
# above are how to check it. The published DGS10 run, with sixteen
# components and three thousand steps, is the configuration the papers
# read structure from; the next example runs a shorter version of it.
