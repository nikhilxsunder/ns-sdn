# filepath: /src/ns_sdn/engine/_internals/_mixins.py
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
"""Behavior mixins: capabilities a frozen result opts into.

Every class declares ``__slots__ = ()`` and no fields. A mixin annotates
the attributes it reads and the concrete result declares them, which is
what lets several compose onto one slotted dataclass. The surface and the
names match ``cultivars``' mixins so a result from this package reads the
same way to a user of that one.

Three capabilities: rendering (:class:`_SummaryMixin`, which turns a
``_summary_table`` into ``summary()``, ``repr``, ``str`` and the notebook
display), frame interop (:class:`_SeriesMixin`, ``to_dict`` and
``to_pandas`` over the aligned per-observation outputs), and
criterion-named comparison (:class:`_ComparisonMixin`, the information
criteria and ``compare``).

Example:
    >>> import numpy as np
    >>> from ns_sdn.spectral_decomposition import NSSDN, Stage
    >>> y = np.sin(0.3 * np.arange(80.0))
    >>> res = NSSDN(y, k=1, state_dim=4, schedule=(Stage(1e-2, 5),), seed=0).fit()
    >>> [m.__name__ for m in type(res).__mro__ if m.__name__.endswith("Mixin")]
    ['_SummaryMixin', '_SeriesMixin', '_ComparisonMixin']
    >>> res.summary().title, res.to_dict()["observed"].shape, res.aic > res.llf
    ('NSSDN(k=1) Results', (80,), True)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ...exceptions import SpecificationError
from .._core._containers import _InformationCriteria, _SummaryTable

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["_ComparisonMixin", "_SeriesMixin", "_SummaryMixin"]


class _SummaryMixin:
    """Text, notebook, and dataframe rendering for a fitted result.

    The concrete result supplies :meth:`_summary_table`; everything a user
    touches derives from it: ``summary()`` returns the table, ``repr`` and
    ``str`` render it as text, and ``_repr_html_`` renders it in a
    notebook. Results mixing this in must be declared
    ``@dataclass(..., repr=False)`` or the generated field dump shadows
    the summary.
    """

    __slots__ = ()

    def _summary_table(self) -> _SummaryTable:
        """Build the structured summary; every concrete result overrides this.

        Returns:
            The table.

        Raises:
            NotImplementedError: Always, on the mixin itself.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _summary_table().")

    def summary(self) -> _SummaryTable:
        """The estimation summary, renderable as text, HTML, or a dataframe.

        Returns:
            The :class:`~ns_sdn.summary.SummaryTable`; ``print`` it,
            display it in a notebook, or call ``to_pandas()`` on it.
        """
        return self._summary_table()

    def __repr__(self) -> str:
        """Render the summary as text, so a bare result shows its table."""
        return self._summary_table().to_text()

    def __str__(self) -> str:
        """Render the summary as text."""
        return self._summary_table().to_text()

    def _repr_html_(self) -> str:
        """Render the summary as HTML for Jupyter.

        Returns:
            The fragment :meth:`~ns_sdn.summary.SummaryTable._repr_html_`
            produces.
        """
        return self._summary_table()._repr_html_()


class _SeriesMixin:
    """Dataframe interop for the aligned per-observation outputs of a fit.

    Reads ``endog``, ``fittedvalues`` and ``resid``; a result with more
    per-observation paths overrides :meth:`_series` to add them as
    columns, and ``to_dict`` and ``to_pandas`` follow.
    """

    __slots__ = ()

    endog: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """The aligned per-observation outputs, keyed by column name.

        Returns:
            ``observed`` (the slice of ``endog`` the fitted values align
            with), ``fitted`` and ``resid``, each of ``nobs`` rows.
        """
        fitted = self.fittedvalues
        return {
            "observed": self.endog[self.endog.shape[0] - fitted.shape[0] :],
            "fitted": fitted,
            "resid": self.resid,
        }

    def to_dict(self) -> dict[str, npt.NDArray[np.float64]]:
        """The aligned outputs as plain arrays.

        Returns:
            What :meth:`_series` produces, with no pandas involved.
        """
        return self._series()

    def to_pandas(self, *, index: npt.ArrayLike | None = None) -> pd.DataFrame:
        """The aligned outputs as a :class:`pandas.DataFrame`, one column per series.

        Args:
            index: Optional row index of the right length, typically the
                dates of the rows; the index is named ``obs`` either way.

        Returns:
            The frame.

        Raises:
            ImportError: If pandas is not installed; install
                ``ns-sdn[pandas]``.
        """
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise ImportError("to_pandas requires pandas: pip install 'ns-sdn[pandas]'") from error
        frame = pd.DataFrame(self._series())
        if index is not None:
            frame.index = pd.Index(np.asarray(index))
        frame.index.name = "obs"
        return frame


class _ComparisonMixin:
    r"""Criterion-named comparison between fitted results.

    Reads ``llf``, ``nobs`` and ``n_params`` and exposes the criteria
    :math:`\mathrm{AIC} = -2\ell + 2p`,
    :math:`\mathrm{BIC} = -2\ell + p\log m` and
    :math:`\mathrm{HQIC} = -2\ell + 2p\log\log m`. No ordering dunders:
    AIC and BIC routinely disagree, so a comparison always names its
    criterion.
    """

    __slots__ = ()

    llf: float
    nobs: int
    n_params: float

    def _comparison_label(self) -> str:
        """The row label a comparison table prints; results override it.

        Returns:
            The class name by default.
        """
        return type(self).__name__

    @property
    def information_criteria(self) -> _InformationCriteria:
        """AIC, BIC, and HQIC from ``llf``, ``nobs``, ``n_params``, as one record."""
        return _InformationCriteria.from_likelihood(self.llf, self.nobs, self.n_params)

    @property
    def aic(self) -> float:
        r"""Akaike information criterion, :math:`-2\ell + 2p`."""
        return self.information_criteria.aic

    @property
    def bic(self) -> float:
        r"""Bayesian information criterion, :math:`-2\ell + p\log m`."""
        return self.information_criteria.bic

    @property
    def hqic(self) -> float:
        r"""Hannan-Quinn information criterion, :math:`-2\ell + 2p\log\log m`."""
        return self.information_criteria.hqic

    def compare(self, *others: _ComparisonMixin, criterion: str = "aic") -> _SummaryTable:
        """Rank this result and ``others`` by one criterion, best first.

        Args:
            others: Results fitted on the same effective sample (the same
                ``nobs``); any mixture of result types that carry the
                three attributes.
            criterion: ``"aic"``, ``"bic"`` or ``"hqic"``.

        Returns:
            A table with one row per result: the label, ``llf``, the
            parameter count under ``k``, the criterion and its difference
            from the best (zero for the first row).

        Raises:
            SpecificationError: If the criterion is unknown or the results
                do not share ``nobs``.

        Example:
            >>> import numpy as np
            >>> from ns_sdn.spectral_decomposition import NSSDN, Stage
            >>> y = np.sin(0.3 * np.arange(80.0))
            >>> one = NSSDN(y, k=1, state_dim=4, schedule=(Stage(1e-2, 5),), seed=0).fit()
            >>> two = NSSDN(y, k=2, state_dim=4, schedule=(Stage(1e-2, 5),), seed=0).fit()
            >>> table = one.compare(two, criterion="bic")
            >>> table.columns, table.rows[0][-1]
            (('model', 'llf', 'k', 'BIC', 'dBIC'), '0.000')
            >>> one.compare(two, criterion="r2")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ns_sdn.exceptions.SpecificationError: criterion must be 'aic', 'bic' or 'hqic'; got ...
        """
        if criterion not in ("aic", "bic", "hqic"):
            raise SpecificationError(
                f"criterion must be 'aic', 'bic' or 'hqic'; got {criterion!r}."
            )
        results = (self, *others)
        if len({r.nobs for r in results}) != 1:
            raise SpecificationError("results with different nobs are not comparable by criterion.")
        scored = sorted(results, key=lambda r: getattr(r.information_criteria, criterion))
        best = getattr(scored[0].information_criteria, criterion)
        name = criterion.upper()
        rows = tuple(
            (
                r._comparison_label(),
                f"{r.llf:.3f}",
                f"{r.n_params:g}",
                f"{getattr(r.information_criteria, criterion):.3f}",
                f"{getattr(r.information_criteria, criterion) - best:.3f}",
            )
            for r in scored
        )
        return _SummaryTable(
            title=f"Model comparison by {name}",
            columns=("model", "llf", "k", name, f"d{name}"),
            rows=rows,
            notes=("Criteria are comparable only across fits sharing an engine configuration.",),
        )
