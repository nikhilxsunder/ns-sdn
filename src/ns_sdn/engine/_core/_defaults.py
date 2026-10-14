# filepath: /src/ns_sdn/engine/_core/_defaults.py
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
r"""Package-wide numeric constants and estimator defaults.

Every magic number resolves here. Changing one changes fitted output, so
these are implementation detail rather than public API; the public
constructors expose the ones a user may legitimately move (the Huber
threshold, the schedule, the active-weight threshold) as arguments with
these as their defaults. The training defaults are the two-stage
schedule of the IEEE-ORSS DGS10 run.

Example:
    >>> _STAGE_A[1] + _STAGE_B[1]
    3000
    >>> _FREQUENCY_MIN < _FREQUENCY_FLOOR * 100 and _TINY < _AMPLITUDE_FLOOR
    True
"""

from __future__ import annotations

import math

__all__ = [
    "_ACTIVE_WEIGHT",
    "_AMPLITUDE_FLOOR",
    "_FREQUENCY_FLOOR",
    "_FREQUENCY_MIN",
    "_HISTORY_EVERY",
    "_HUBER_DELTA",
    "_LOG_2PI",
    "_SCHEMA_VERSION",
    "_SPECTRUM_NFFT",
    "_STAGE_A",
    "_STAGE_B",
    "_TINY",
]

_LOG_2PI: float = math.log(2.0 * math.pi)
r""":math:`\log 2\pi`, the constant of the Gaussian log-likelihood.

Enters the concentrated likelihood
:math:`\ell = -\tfrac{m}{2}(\log 2\pi + \log\hat\sigma^2 + 1)`.
"""

_TINY: float = 1e-12
r"""Division guard, :math:`10^{-12}`.

The floor on every denominator and logarithm argument that can reach
zero: the residual variance, the power row sums behind the spectral
weights, the weights inside the entropy, the step sizes behind the
velocities, and the correlation denominators.
"""

_AMPLITUDE_FLOOR: float = 1e-4
r""":math:`\varepsilon_A`, added to the softplus amplitude so no component is exactly silent.

:math:`A_{t,k} = \operatorname{softplus}(\cdot) + \varepsilon_A`, in
training units. A strictly positive amplitude keeps the frequency and
phase heads of every component on the gradient path.
"""

_FREQUENCY_FLOOR: float = 1e-4
r""":math:`\varepsilon_\omega`, added to the softplus frequency so phase always advances.

:math:`\omega_{t,k} = \operatorname{softplus}(\cdot) + \varepsilon_\omega`,
in radians per unit time, under ``positive_frequency``.
"""

_FREQUENCY_MIN: float = 1e-3
r""":math:`\omega_{\min}`, lower edge of the ordered-frequency band, in radians per unit time.

Under ``ordered_frequency`` the frequencies are cumulative softmax gaps
inside :math:`[\omega_{\min}, \pi / \operatorname{median}\Delta t]`; the
upper edge is the Nyquist rate of the grid.
"""

_HUBER_DELTA: float = 1.0
r"""Huber threshold :math:`\delta` on the scaled residual.

Under min-max scaling to :math:`[-1, 1]` this is half the sample range:
residuals beyond it are penalized linearly rather than quadratically.
"""

_SPECTRUM_NFFT: int = 256
"""Points in the real FFT used by the spectrum penalty.

The target and the prediction are zero-padded or truncated to this
length before their power spectra are compared.
"""

_HISTORY_EVERY: int = 100
"""Steps between recorded losses in the training history.

Each stage records step 0, every multiple of this, and its last step.
"""

_ACTIVE_WEIGHT: float = 1e-2
r"""Power-weight threshold :math:`w_{t,k} \ge 10^{-2}` below which a component is inactive.

A component carrying less than one percent of the power at a row has an
unidentified frequency there; the spectral-state ``active`` mask uses
this floor unless the caller passes another.
"""

_SCHEMA_VERSION: int = 1
"""Reserved: the version a serialized result will carry once the format is frozen.

Not yet written into pickles; bump it when the predictor's ``__getstate__``
layout changes so a loader can tell the layouts apart.
"""

_STAGE_A: tuple[float, int, float, float] = (1e-3, 1200, 2e-4, 8e-4)
r"""Stage A of the published schedule.

:math:`(\eta, \text{steps}, \lambda_A, \lambda_\omega)`: Adam learning
rate, full-sequence steps, amplitude-drift weight, frequency-drift
weight.
"""

_STAGE_B: tuple[float, int, float, float] = (4e-4, 1800, 3e-4, 1.5e-3)
r"""Stage B of the published schedule, same layout as :data:`_STAGE_A`.

A lower rate with heavier drift penalties, run after stage A on the same
network with a fresh optimizer.
"""
