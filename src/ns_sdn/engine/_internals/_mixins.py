# Copyright (c) 2026 Nikhil Sunder
# SPDX-License-Identifier: MIT
"""Behavior mixins: capabilities a frozen result opts into.

Every class declares ``__slots__ = ()`` and no fields. A mixin annotates the
attributes it reads and the concrete result declares them, which is what
lets several compose onto one slotted dataclass. The surface and the names
match ``cultivars``' mixins so a result from this package reads the same
way to a user of that one.
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
    touches derives from it. Results mixing this in must be declared
    ``@dataclass(..., repr=False)`` or the generated field dump shadows it.
    """

    __slots__ = ()

    def _summary_table(self) -> _SummaryTable:
        raise NotImplementedError(f"{type(self).__name__} must implement _summary_table().")

    def summary(self) -> _SummaryTable:
        """The estimation summary, renderable as text, HTML, or a dataframe."""
        return self._summary_table()

    def __repr__(self) -> str:
        return self._summary_table().to_text()

    def __str__(self) -> str:
        return self._summary_table().to_text()

    def _repr_html_(self) -> str:
        return self._summary_table()._repr_html_()


class _SeriesMixin:
    """Dataframe interop for the aligned per-observation outputs of a fit."""

    __slots__ = ()

    endog: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """The aligned per-observation outputs, keyed by column name."""
        fitted = self.fittedvalues
        return {
            "observed": self.endog[self.endog.shape[0] - fitted.shape[0] :],
            "fitted": fitted,
            "resid": self.resid,
        }

    def to_dict(self) -> dict[str, npt.NDArray[np.float64]]:
        """The aligned outputs as plain arrays."""
        return self._series()

    def to_pandas(self, *, index: npt.ArrayLike | None = None) -> pd.DataFrame:
        """The aligned outputs as a :class:`pandas.DataFrame`, one column per series.

        Args:
            index: Optional row index, typically the dates of the effective sample.

        Raises:
            ImportError: If pandas is not installed.
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
    """Criterion-named comparison between fitted results.

    No ordering dunders: AIC and BIC routinely disagree, so a comparison
    always names its criterion.
    """

    __slots__ = ()

    llf: float
    nobs: int
    n_params: float

    def _comparison_label(self) -> str:
        return type(self).__name__

    @property
    def information_criteria(self) -> _InformationCriteria:
        """AIC, BIC, and HQIC from ``llf``, ``nobs``, ``n_params``."""
        return _InformationCriteria.from_likelihood(self.llf, self.nobs, self.n_params)

    @property
    def aic(self) -> float:
        """Akaike information criterion."""
        return self.information_criteria.aic

    @property
    def bic(self) -> float:
        """Bayesian information criterion."""
        return self.information_criteria.bic

    @property
    def hqic(self) -> float:
        """Hannan-Quinn information criterion."""
        return self.information_criteria.hqic

    def compare(self, *others: _ComparisonMixin, criterion: str = "aic") -> _SummaryTable:
        """Rank this result and ``others`` by one criterion, best first.

        Args:
            others: Results fitted on the same effective sample.
            criterion: ``"aic"``, ``"bic"`` or ``"hqic"``.

        Returns:
            A table with the label, ``llf``, parameter count, the criterion
            and its difference from the best.

        Raises:
            SpecificationError: If the criterion is unknown or the results
                do not share ``nobs``.
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
