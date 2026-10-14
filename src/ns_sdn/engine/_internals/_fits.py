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
r"""Fit records: the raw output of an estimator, before result assembly.

A fit carries what training produced and nothing about how; the model
that owns the specification produces it (``_models._fit_family``), and
the public result is assembled from the fit and the specification
together (``NSSDNResult._from_fit``). Keeping the two apart is what lets
the result be a frozen record of plain arrays while the mechanism that
produced it stays private.

Example:
    >>> _SpectralFit.__dataclass_params__.frozen, len(_SpectralFit.__slots__)
    (True, 9)
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
    r"""What one training run produced.

    Every array is on the original scale and the likelihood is the
    concentrated Gaussian one,

    .. math::

       \hat\sigma^2 = \frac{1}{m}\sum_{t=1}^{n-1} e_t^2,
       \qquad
       \ell = -\frac{m}{2}\bigl(\log 2\pi + \log\hat\sigma^2 + 1\bigr),
       \qquad m = n - 1,

    with :math:`e_t` the one-step residuals.

    Attributes:
        paths: The teacher-forced pass over the sample: :math:`n` rows
            under the filtered alignment, :math:`n - 1` under the
            predicted one.
        fittedvalues: ``(n - 1,)`` one-step conditional predictions
            aligned with ``endog[1:]``: the emission rows under the
            predicted alignment, the emissions through :math:`n - 2` under
            the filtered one.
        resid: ``(n - 1,)`` ``endog[1:] - fittedvalues``.
        llf: Concentrated Gaussian log-likelihood :math:`\ell` of ``resid``.
        nobs: The effective sample size :math:`m = n - 1`.
        n_params: Trainable parameter count.
        sigma2: :math:`\hat\sigma^2`, floored at :math:`10^{-12}`.
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
