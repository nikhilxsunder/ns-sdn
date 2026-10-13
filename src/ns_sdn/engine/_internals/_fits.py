# Copyright (c) 2026 Nikhil Sunder
# SPDX-License-Identifier: MIT
"""Fit records: the raw output of an estimator, before result assembly.

A fit carries what training produced and nothing about how; the model that
owns the specification produces it, and the public result is assembled
from the fit and the specification together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._predictors import _SpectralPaths, _SpectralPredictor
from ._trainers import _StageRecord

__all__ = ["_SpectralFit"]


@dataclass(frozen=True, kw_only=True, slots=True)
class _SpectralFit:
    """What one training run produced.

    Attributes:
        paths: The teacher-forced pass over the sample: ``n`` rows under the
            filtered alignment, ``n - 1`` under the predicted one.
        fittedvalues: One-step conditional predictions aligned with
            ``endog[1:]``: the emission rows under the predicted alignment,
            the emissions through ``n - 2`` under the filtered one.
        resid: ``endog[1:] - fittedvalues``.
        llf: Concentrated Gaussian log-likelihood of ``resid``.
        nobs: ``n - 1``.
        n_params: Trainable parameter count.
        sigma2: ``resid @ resid / nobs``.
        history: Recorded losses by stage and step.
        predictor: The trained network, callable on new data.
    """

    paths: _SpectralPaths
    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int
    sigma2: float
    history: tuple[_StageRecord, ...]
    predictor: _SpectralPredictor
