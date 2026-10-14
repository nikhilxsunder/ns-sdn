# filepath: /examples/plot_03_reproduce_orss.py
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
Reproducing the IEEE-ORSS DGS10 run
===================================

The published experiment of Sunder and Leon (2026), *2026 IEEE Opportunity
Research Scholars Symposium (ORSS)*, pp. 1-5,
`doi:10.1109/ORSS71174.2026.11684282 <https://doi.org/10.1109/ORSS71174.2026.11684282>`_,
step for step, through the package: the 10-year
Treasury constant-maturity yield (:fred:`DGS10`) in logs from February
2000 on a calendar-day grid with gaps of up to a week filled forward, the
last 365 observations held out, the default :class:`~ns_sdn.spectral_decomposition.NSSDN`
configuration (sixteen components, a 128-dimensional GRU state, the
filtered alignment, the two-stage schedule with its amplitude and
frequency drift penalties) with seed 1337, then the in-sample
reconstruction, the conditional one-step holdout, the recursive holdout,
the metrics table with the random-walk row the paper did not include, and
the trend, amplitude, frequency and gating paths on the training scale.
The spectral state of the fit, which the paper did not compute, closes the
script.

This is the only example the documentation build does not execute by
default. The published fit is three thousand full-sequence gradient steps
over 9,186 observations, which takes tens of minutes to an hour on a
recent laptop and hours on a CI runner. Run it directly,

.. code-block:: bash

   FRED_API_KEY=... python examples/plot_03_reproduce_orss.py

which writes ``metrics.csv``, ``spectral_state.csv`` and every figure to
``orss_reproduction/`` (or the directory named by ``NSSDN_ORSS_OUTPUT``),
or build the documentation with ``DOCS_RUN_ORSS=1`` to execute it in the
gallery. The shorter :ref:`DGS10 example
<sphx_glr_auto_examples_plot_02_dgs10_spectral_state.py>` runs the same
protocol at a size a build can afford.
"""

# %%
# Data
# ----
# ``fetch_series`` reproduces the paper's preparation: ``asfreq("D")`` and a
# forward fill limited to seven days, so weekends and holidays carry the
# last quote and longer outages are dropped. The key is read from
# ``FRED_API_KEY`` and never written anywhere.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ns_sdn.spectral_decomposition import NSSDN

from _fred import fetch_series

SERIES_ID = "DGS10"
START_DATE = "2000-02-01"
HOLDOUT = 365
SEED = 1337
SHOWN = 4
OUTPUT = Path(os.environ.get("NSSDN_ORSS_OUTPUT", "orss_reproduction"))
OUTPUT.mkdir(parents=True, exist_ok=True)

log_yield_series = fetch_series(SERIES_ID, start=START_DATE, units="log")
dates = log_yield_series.index
log_yield = log_yield_series.to_numpy()
split = log_yield.shape[0] - HOLDOUT
train, holdout = log_yield[:split], log_yield[split:]
print(f"N={log_yield.shape[0]}  train={split}  holdout={HOLDOUT}")
print(f"{dates[0].date()} to {dates[-1].date()}; holdout from {dates[split].date()}")

# %%
# Fit
# ---
# Every default is the published configuration, so the call is the shortest
# one in the library. The parameter count of the published run is 58,849.

result = NSSDN(train, seed=SEED).fit()
print(f"trainable parameters: {result.n_params}")
print(result.summary())


# %%
# Metrics
# -------
# RMSE, MAE and MAPE as the validation notebook defines them, on the
# training reconstruction, on the conditional one-step holdout in scaled,
# log and level units, on the recursive holdout, and for the random walk on
# the same holdout.


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and MAPE in percent, as the validation notebook defines them."""
    error = predicted - actual
    denominator = np.where(np.abs(actual) < 1e-12, np.nan, actual)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape_pct": float(np.nanmean(np.abs(error / denominator)) * 100.0),
    }


fit_log, fit_level = result.emission, np.exp(result.emission)
one_log = result.predict(log_yield)[-HOLDOUT:]
one_level = np.exp(one_log)
recursive_log = result.forecast(HOLDOUT)
naive_log = log_yield[split - 1 : -1]
scaled = result.scale.transform

table = pd.DataFrame(
    {
        "train_log_scale": metrics(train, fit_log),
        "train_level_scale": metrics(np.exp(train), fit_level),
        "holdout_1step_scaled": metrics(scaled(holdout), scaled(one_log)),
        "holdout_1step_log": metrics(holdout, one_log),
        "holdout_1step_level": metrics(np.exp(holdout), one_level),
        "holdout_recursive_log": metrics(holdout, recursive_log),
        "holdout_recursive_level": metrics(np.exp(holdout), np.exp(recursive_log)),
        "holdout_random_walk_log": metrics(holdout, naive_log),
        "holdout_random_walk_level": metrics(np.exp(holdout), np.exp(naive_log)),
    }
).T
print(table.to_string())
table.to_csv(OUTPUT / "metrics.csv")

# %%
# Figure 1: in-sample fit on the training segment
# -----------------------------------------------
# The emission under the filtered alignment reconstructs the observed
# series; the figure is in levels.

holdout_start = dates[split]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates[:split], np.exp(train), lw=0.8, label="historical actual")
ax.plot(dates[:split], fit_level, lw=0.8, label="historical fitted")
ax.set_ylabel("percent")
ax.set_title("In-sample fit on training segment")
ax.legend(loc="upper right")
plt.tight_layout()
fig.savefig(OUTPUT / "in_sample_fit.png", dpi=100)

# %%
# Figure 2: the holdout, conditional one-step against recursive
# -------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates[split:], np.exp(holdout), lw=0.9, label="actual (holdout)")
ax.plot(dates[split:], one_level, lw=0.9, label="conditional 1-step")
ax.plot(dates[split:], np.exp(recursive_log), lw=0.9, label="recursive multi-step")
ax.set_ylabel("percent")
ax.set_title("Holdout: conditional 1-step vs recursive forecast")
ax.legend(loc="upper right")
plt.tight_layout()
fig.savefig(OUTPUT / "holdout_forecasts.png", dpi=100)

# %%
# Figures 3 to 6: the paths on the training scale
# -----------------------------------------------
# ``decompose(log_yield, scaled=True)`` runs the trained network over the
# full sample, training and holdout together, and returns the paths in
# training units, which is how the paper draws them. The dashed line is
# the holdout boundary. Four of the sixteen components are shown, as in
# the paper, by index.

paths = result.decompose(log_yield, scaled=True)
shown = range(min(SHOWN, result.k))


def figure(name: str, title: str, series: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, values in series.items():
        ax.plot(dates, values, lw=0.8, label=label)
    ax.axvline(holdout_start, color="k", linestyle="--", lw=0.8)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT / f"{name}.png", dpi=100)


figure("learned_trend", "Learned trend component $B_t$", {"$B_t$": paths["trend"]})
figure(
    "amplitude_paths",
    "Selected amplitude paths",
    {f"amplitude[{j}]": paths["amplitude"][:, j] for j in shown},
)
figure(
    "frequency_paths",
    "Selected instantaneous frequency paths (radians per day)",
    {f"frequency[{j}]": paths["frequency"][:, j] for j in shown},
)
figure(
    "gating_paths",
    "Selected gating paths",
    {f"gate[{j}]": paths["gate"][:, j] for j in shown},
)

# %%
# The spectral state
# ------------------
# Not in the paper: the decomposition of the training segment read as a
# time-local measure on the frequency axis, with the 2020 window shaded.
# A joint contraction of dispersion and effective count across the window
# is the compression reading; the disentanglement score says how far the
# per-component paths behind it can be trusted.

state = result.spectral_state()
coords = state.to_pandas(index=dates[: state.nobs])
coords.to_csv(OUTPUT / "spectral_state.csv")
print(state.summary())

window = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-08-31"))
before = coords.loc["2019-09-01":"2020-02-29"].mean()
after = coords.loc[window[0] : window[1]].mean()
comparison = pd.DataFrame({"before": before, "after": after})
comparison["change"] = comparison["after"] - comparison["before"]
rows = ["centroid", "dispersion", "effective_count", "velocity"]
print(comparison.loc[rows].round(4).to_string())

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(coords.index, coords["centroid"], lw=0.8, label="centroid")
axes[0].fill_between(
    coords.index,
    coords["centroid"] - coords["dispersion"],
    coords["centroid"] + coords["dispersion"],
    alpha=0.25,
    label="centroid $\\pm$ dispersion",
)
axes[0].set_ylabel("rad/day")
axes[1].plot(coords.index, coords["effective_count"], lw=0.8, label="effective count")
axes[1].plot(coords.index, coords["participation_ratio"], lw=0.8, label="participation ratio")
axes[1].set_ylabel("components")
axes[2].plot(coords.index, coords["velocity"], lw=0.6, label="velocity")
axes[2].plot(coords.index, coords["velocity_frequency"], lw=0.6, label="frequency part")
axes[2].plot(coords.index, coords["velocity_mass"], lw=0.6, label="mass part")
axes[2].set_ylabel("$W_1$ per day")
for ax in axes:
    ax.axvspan(*window, color="0.85")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
axes[0].set_title("Spectral state of the training segment, 2020 window shaded")
plt.tight_layout()
fig.savefig(OUTPUT / "spectral_state.png", dpi=100)

print(f"wrote metrics.csv, spectral_state.csv and the figures to {OUTPUT.resolve()}")
