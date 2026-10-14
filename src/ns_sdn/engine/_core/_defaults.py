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
"""Package-wide numeric constants and estimator defaults.

Every magic number resolves here. Changing one changes fitted output, so
these are implementation detail rather than public API. The training
defaults are the two-stage schedule of the IEEE-ORSS DGS10 run.
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
"""Log of 2π."""

_TINY: float = 1e-12
"""Division guard."""

_AMPLITUDE_FLOOR: float = 1e-4
"""Added to the softplus amplitude so no component is exactly silent."""

_FREQUENCY_FLOOR: float = 1e-4
"""Added to the softplus frequency so phase always advances."""

_FREQUENCY_MIN: float = 1e-3
"""Lower edge of the ordered-frequency band, in radians per unit time."""

_HUBER_DELTA: float = 1.0
"""Huber threshold on the scaled residual."""

_SPECTRUM_NFFT: int = 256
"""Points in the rFFT used by the spectrum penalty."""

_HISTORY_EVERY: int = 100
"""Steps between recorded losses in the training history."""

_ACTIVE_WEIGHT: float = 1e-2
"""Power-weight threshold below which a component is inactive for diagnostics."""

_SCHEMA_VERSION: int = 1

_STAGE_A: tuple[float, int, float, float] = (1e-3, 1200, 2e-4, 8e-4)
"""Stage A of the published schedule: (learning rate, steps, amplitude, frequency)."""

_STAGE_B: tuple[float, int, float, float] = (4e-4, 1800, 3e-4, 1.5e-3)
"""Stage B of the published schedule."""
