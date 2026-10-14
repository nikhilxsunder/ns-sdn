# filepath: /src/ns_sdn/engine/_internals/_fits.py
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
