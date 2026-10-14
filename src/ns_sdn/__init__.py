# filepath: /src/ns_sdn/__init__.py
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
"""ns-sdn: the Non-Stationary Spectral Decomposition Network.

A neural state-space model that represents a scalar time series as a
trend plus a small number of sinusoids whose amplitude, instantaneous
frequency, phase offset and participation evolve with a latent state,
estimated as a one-step forecaster and read as an interpretable
time-varying spectral decomposition.

The package keeps the rule its sibling ``cultivars`` keeps: a fitted
result is a record, not a handle. ``fit`` returns a frozen dataclass
carrying the one-step fit, the complete decomposition on the original
scale, and a ``summary`` whose notes say what the numbers assume; the
model object holds only the specification. Names are imported from the
leaf module that defines them:
``from ns_sdn.spectral_decomposition import NSSDN``.

Layout. :mod:`~ns_sdn.spectral_decomposition` holds the specification
and the result, :mod:`~ns_sdn.spectral_state` the measure-valued
coordinates that read a decomposition, :mod:`~ns_sdn.summary` the
reporting records, :mod:`~ns_sdn.typing` the categorical option aliases,
:mod:`~ns_sdn.exceptions` the error hierarchy, and :mod:`~ns_sdn.engine`
the private ``_core`` and ``_internals`` layers for anyone extending the
package.

Example:
    >>> import numpy as np
    >>> from ns_sdn.spectral_decomposition import NSSDN, Stage
    >>> y = np.sin(0.25 * np.arange(160.0))
    >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 30),), seed=0).fit()
    >>> res.fittedvalues.shape, res.forecast(3).shape
    ((159,), (3,))
"""

from __future__ import annotations

from . import engine, exceptions, spectral_decomposition, spectral_state, summary, typing

__all__ = [
    "engine",
    "exceptions",
    "spectral_decomposition",
    "spectral_state",
    "summary",
    "typing",
]
