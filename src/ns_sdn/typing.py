# filepath: /src/ns_sdn/typing.py
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
r"""Categorical option aliases for the public constructors.

Each name is a PEP 695 alias over a ``Literal``: an annotation for the
static checker and the argument the runtime validator reads, so the type
and the check cannot drift apart. A name here is a specification choice
with modeling content, never a display option:

* :data:`Alignment` -- what the emission at :math:`t` conditions on
  (``"filtered"`` on :math:`y_t`, ``"predicted"`` on :math:`y_{t-1}`).
* :data:`Transition` -- the latent update :math:`F_\psi`
  (``"gru"`` or ``"timeaware"``).
* :data:`Objective` -- the prediction term of the loss
  (``"mse"`` or ``"huber"``).
* :data:`Scaling` -- how the sample is mapped before training
  (``"minmax"`` or ``"none"``).

The aliases are defined in ``engine._core._types`` and re-exported here
so that user code imports them from a public module; the docstring of
each alias says what every value does to the fit.

See Also:
    * :class:`ns_sdn.spectral_decomposition.NSSDN` -- the constructor
      these annotate.
    * :class:`ns_sdn.exceptions.SpecificationError` -- what a value
      outside an alias raises.

Example:
    >>> from ns_sdn.typing import Alignment, Transition
    >>> from typing import get_args
    >>> get_args(Alignment.__value__), get_args(Transition.__value__)
    (('filtered', 'predicted'), ('gru', 'timeaware'))

    A misspelled option is refused at construction with the admissible
    set in the message:

    >>> import numpy as np
    >>> from ns_sdn.spectral_decomposition import NSSDN
    >>> NSSDN(np.arange(10.0), scaling="standard")
    Traceback (most recent call last):
        ...
    ns_sdn.exceptions.SpecificationError: scaling must be one of 'minmax', 'none'; got 'standard'.
"""

from __future__ import annotations

from .engine._core._types import Alignment, Objective, Scaling, Transition

__all__ = ["Alignment", "Objective", "Scaling", "Transition"]
