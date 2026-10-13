# Copyright (c) 2026 Nikhil Sunder
# SPDX-License-Identifier: MIT
"""The reporting surface: the records every fitted result renders itself through.

``summary()`` on a result does not format anything; it returns a
:class:`SummaryTable`, a record that renders on demand as text, HTML, or a
dataframe. :class:`InformationCriteria` is the record behind every
``information_criteria`` property. Both are defined once in ``_core`` and
bound to this module so pickles, annotations, and the API reference name
them here.

Example:
    >>> from ns_sdn.summary import SummaryTable
    >>> table = SummaryTable(title="T", metadata=(("a", "1"),), notes=("n",))
    >>> table.to_text().splitlines()[-1], type(table).__module__
    ('n', 'ns_sdn.summary')
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
