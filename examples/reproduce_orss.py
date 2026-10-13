# Copyright (c) 2026 Nikhil Sunder
# SPDX-License-Identifier: MIT
"""Reproduce the IEEE-ORSS DGS10 run and its figures through the package.

Mirrors ``notebooks/tests.ipynb`` step for step: log DGS10 from FRED on a
daily grid with short gaps filled, the last 365 observations held out, the
default ``NSSDN`` configuration (k=16, d=128, GRU, filtered alignment, the
two-stage schedule) with seed 1337, then the in-sample reconstruction, the
conditional one-step holdout, the recursive holdout, and the trend,
amplitude, frequency and gating paths on the training scale.

Usage::

    FRED_API_KEY=... python examples/reproduce_orss.py [output_dir]

The key is read from the environment and never written anywhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import fedfred as fd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ns_sdn.spectral_decomposition import NSSDN

SERIES_ID = "DGS10"
START_DATE = "2000-02-01"
HOLDOUT = 365
SEED = 1337
SHOWN = 4


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and MAPE in percent, as the validation notebook defines them."""
    error = predicted - actual
    denominator = np.where(np.abs(actual) < 1e-12, np.nan, actual)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape_pct": float(np.nanmean(np.abs(error / denominator)) * 100.0),
    }


def main(output: Path) -> None:
    """Fetch, fit, evaluate, and write the metrics table and figures to ``output``."""
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise SystemExit("set FRED_API_KEY in the environment.")
    output.mkdir(parents=True, exist_ok=True)

    frame = fd.FredAPI(api_key=key).get_series_observations(
        series_id=SERIES_ID, observation_start=START_DATE, units="log"
    )
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"]).asfreq("D")
    frame["value"] = frame["value"].ffill(limit=7)
    frame = frame.dropna(subset=["value"])
    dates = frame.index
    log_yield = frame["value"].to_numpy(dtype=np.float64)
    split = log_yield.shape[0] - HOLDOUT
    train, holdout = log_yield[:split], log_yield[split:]
    print(f"N={log_yield.shape[0]}  train={split}  holdout={HOLDOUT}")

    result = NSSDN(train, seed=SEED).fit()
    print(f"trainable parameters: {result.n_params}")

    # Fig. 1: in-sample reconstruction (the emission under the filtered alignment).
    fit_log, fit_level = result.emission, np.exp(result.emission)
    table = pd.DataFrame(
        [metrics(train, fit_log), metrics(np.exp(train), fit_level)],
        index=["train_log_scale", "train_level_scale"],
    )

    # Fig. 2: conditional one-step holdout, the published protocol.
    one_log = result.predict(log_yield)[-HOLDOUT:]
    one_level = np.exp(one_log)
    scaled = result.scale.transform
    table = pd.concat(
        [
            table,
            pd.DataFrame(
                [
                    metrics(scaled(holdout), scaled(one_log)),
                    metrics(holdout, one_log),
                    metrics(np.exp(holdout), one_level),
                ],
                index=["holdout_1step_scaled", "holdout_1step_log", "holdout_1step_level"],
            ),
        ]
    )

    # Recursive holdout, the validation notebook's stricter test.
    recursive_log = result.forecast(HOLDOUT)
    table = pd.concat(
        [
            table,
            pd.DataFrame(
                [metrics(holdout, recursive_log), metrics(np.exp(holdout), np.exp(recursive_log))],
                index=["holdout_recursive_log", "holdout_recursive_level"],
            ),
        ]
    )
    # Random walk on the same holdout, for the comparison the paper did not make.
    table.loc["holdout_random_walk_level"] = metrics(
        np.exp(holdout), np.exp(log_yield[split - 1 : -1])
    )
    print(table.to_string())
    table.to_csv(output / "metrics.csv")
    print(result.summary())

    # Figures 3-6 on the training scale, over the full sample, as in the paper.
    paths = result.decompose(log_yield, scaled=True)
    holdout_start = dates[split]

    def figure(name: str, title: str, series: dict[str, np.ndarray]) -> None:
        plt.figure(figsize=(12, 5))
        for label, values in series.items():
            plt.plot(dates, values, label=label)
        plt.axvline(holdout_start, linestyle="--")
        plt.title(title)
        if len(series) > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"{name}.png", dpi=100)
        plt.close()

    figure("learned_trend", "Learned trend component B_t", {"B_t": paths["trend"]})
    for name, key, title in (
        ("amplitude_paths", "amplitude", "Selected amplitude paths"),
        ("frequency_paths", "frequency", "Selected instantaneous frequency paths"),
        ("gating_paths", "gate", "Selected gating paths"),
    ):
        figure(name, title, {f"{key}[{j}]": paths[key][:, j] for j in range(min(SHOWN, result.k))})

    plt.figure(figsize=(12, 6))
    plt.plot(dates[:split], np.exp(train), label="historical actual")
    plt.plot(dates[:split], fit_level, label="historical fitted")
    plt.title("In-sample fit on training segment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "in_sample_fit.png", dpi=100)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(dates[split:], np.exp(holdout), label="actual (holdout)")
    plt.plot(dates[split:], one_level, label="conditional 1-step")
    plt.plot(dates[split:], np.exp(recursive_log), label="recursive multi-step")
    plt.title("Holdout: conditional 1-step vs recursive forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "holdout_forecasts.png", dpi=100)
    plt.close()

    state = result.spectral_state()
    state.to_pandas(index=dates[: state.nobs]).to_csv(output / "spectral_state.csv")
    print(state.summary())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("orss_reproduction"))
