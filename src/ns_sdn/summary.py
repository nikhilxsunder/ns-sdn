# filepath: /src/ns_sdn/summary.py
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
r"""The reporting surface: the records every fitted result renders itself through.

``summary()`` on a result does not format anything; it returns a
:class:`SummaryTable`, a record that renders on demand as fixed-width
text (``print``, ``str``, the bare repr), as HTML in a notebook
(``_repr_html_``) or as a :class:`pandas.DataFrame` (``to_pandas``).
:class:`InformationCriteria` is the record behind every
``information_criteria`` property, the triple

.. math::

   \mathrm{AIC} = -2\ell + 2p,
   \qquad
   \mathrm{BIC} = -2\ell + p\log m,
   \qquad
   \mathrm{HQIC} = -2\ell + 2p\log\log m,

for a log-likelihood :math:`\ell`, :math:`p` parameters and an effective
sample of :math:`m` observations. Both are defined once in ``_core`` and
bound to this module so pickles, annotations, and the API reference name
them here.

Holding the structure rather than the string is what lets the same
summary carry its notes into every rendering: a table that says
"compare against a random walk before reading these as skill" says it in
the notebook, in the terminal and in the frame's ``attrs`` alike.

See Also:
    * :meth:`ns_sdn.spectral_decomposition.NSSDNResult.summary` and
      :meth:`ns_sdn.spectral_state.SpectralState.summary` -- the two
      producers of a :class:`SummaryTable`.
    * :meth:`ns_sdn.spectral_decomposition.NSSDNResult.compare` -- a
      comparison table across results by one criterion.

Example:
    >>> from ns_sdn.summary import SummaryTable
    >>> table = SummaryTable(title="T", metadata=(("a", "1"),), notes=("n",))
    >>> table.to_text().splitlines()[-1], type(table).__module__
    ('n', 'ns_sdn.summary')

    The criteria record is a named tuple with a constructor from the
    likelihood:

    >>> from ns_sdn.summary import InformationCriteria
    >>> crit = InformationCriteria.from_likelihood(-100.0, 200, 3)
    >>> round(crit.aic, 1), round(crit.bic, 2), crit._fields
    (206.0, 215.89, ('aic', 'bic', 'hqic'))
"""

from __future__ import annotations

from .engine._core._containers import _InformationCriteria as InformationCriteria
from .engine._core._containers import _SummaryTable as SummaryTable

__all__ = ["InformationCriteria", "SummaryTable"]

for _record, _public in (
    (InformationCriteria, "InformationCriteria"),
    (SummaryTable, "SummaryTable"),
):
    _record.__module__ = __name__
    _record.__name__ = _record.__qualname__ = _public
del _record, _public
