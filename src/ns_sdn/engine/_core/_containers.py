# filepath: /src/ns_sdn/engine/_core/_containers.py
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
"""Reusable value objects shared by internal mechanisms and public results."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import numpy as np
import numpy.typing as npt

from ...exceptions import SpecificationError
from ._defaults import _TINY
from ._validators import _validate_nonnegative, _validate_positive_int

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["_InformationCriteria", "_MinMaxScale", "_Penalties", "_Stage", "_SummaryTable"]


@dataclass(frozen=True, slots=True)
class _Penalties:
    """Weights on the drift and sparsity penalties added to the prediction loss.

    Each is a non-negative multiplier on a term evaluated on the scaled
    series; zero removes the term. The defaults are stage A of the
    published DGS10 schedule.

    Attributes:
        amplitude: Squared first differences of the effective amplitude paths.
        frequency: Squared first differences of the instantaneous frequencies.
        phase_curvature: Squared second differences of the cumulative phase.
        trend: Squared first differences of the trend.
        sparsity: Group penalty ``sum_k sqrt(mean_t amplitude_{t,k}^2)``,
            which silences whole components rather than shrinking all of them.
        spectrum: Mean squared difference between the rFFT power spectra of
            the target and the prediction.

    Raises:
        SpecificationError: If any weight is negative or non-finite.

    Example:
        >>> from ns_sdn.spectral_decomposition import Penalties
        >>> Penalties(amplitude=3e-4, frequency=1.5e-3).sparsity
        0.0
    """

    amplitude: float = 2e-4
    frequency: float = 8e-4
    phase_curvature: float = 0.0
    trend: float = 0.0
    sparsity: float = 0.0
    spectrum: float = 0.0

    def __post_init__(self) -> None:
        for name in ("amplitude", "frequency", "phase_curvature", "trend", "sparsity", "spectrum"):
            object.__setattr__(self, name, _validate_nonnegative(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class _Stage:
    """One stage of the training schedule: a fresh Adam run at one learning rate.

    Attributes:
        learning_rate: Adam step size, strictly positive.
        steps: Full-sequence gradient steps, at least one.
        penalties: Weights for this stage, or ``None`` to use the model's.

    Raises:
        SpecificationError: If the rate is not positive or the steps not a
            positive integer.

    Example:
        >>> from ns_sdn.spectral_decomposition import Stage
        >>> Stage(learning_rate=1e-3, steps=1200).penalties is None
        True
    """

    learning_rate: float
    steps: int
    penalties: _Penalties | None = None

    def __post_init__(self) -> None:
        rate = _validate_nonnegative(self.learning_rate, "learning_rate")
        if rate == 0.0:
            raise SpecificationError("learning_rate must be strictly positive; got 0.")
        object.__setattr__(self, "learning_rate", rate)
        object.__setattr__(self, "steps", _validate_positive_int(self.steps, "steps"))
        if self.penalties is not None and not isinstance(self.penalties, _Penalties):
            raise SpecificationError(
                "penalties must be a Penalties record or None; "
                f"got {type(self.penalties).__name__}."
            )


class _InformationCriteria(NamedTuple):
    """Model-selection criteria computed from a fitted log-likelihood.

    Attributes:
        aic: ``-2 llf + 2 k``.
        bic: ``-2 llf + k log n``.
        hqic: ``-2 llf + 2 k log log n``.
    """

    aic: float
    bic: float
    hqic: float

    @classmethod
    def from_likelihood(cls, llf: float, nobs: int, n_params: float) -> _InformationCriteria:
        """Compute all three criteria.

        Raises:
            SpecificationError: If ``nobs < 3`` or ``n_params`` is negative.

        Example:
            >>> round(_InformationCriteria.from_likelihood(-100.0, 200, 3).aic, 1)
            206.0
        """
        if nobs < 3:
            raise SpecificationError(f"nobs must be >= 3 to form criteria; got {nobs}.")
        if n_params < 0:
            raise SpecificationError(f"n_params must be non-negative; got {n_params}.")
        penalty = 2.0 * n_params
        return cls(
            aic=float(-2.0 * llf + penalty),
            bic=float(-2.0 * llf + n_params * np.log(nobs)),
            hqic=float(-2.0 * llf + penalty * np.log(np.log(nobs))),
        )


class _MinMaxScale(NamedTuple):
    """An affine map ``y -> (y - low) / (high - low) * (t_high - t_low) + t_low`` and its inverse.

    Attributes:
        low: Sample minimum.
        high: Sample maximum.
        target_low: Lower edge of the target range.
        target_high: Upper edge of the target range.
    """

    low: float
    high: float
    target_low: float = -1.0
    target_high: float = 1.0

    @classmethod
    def fit(cls, y: npt.NDArray[np.float64]) -> _MinMaxScale:
        """Read the range off a sample.

        Example:
            >>> _MinMaxScale.fit(np.array([2.0, 4.0]))
            _MinMaxScale(low=2.0, high=4.0, target_low=-1.0, target_high=1.0)
        """
        return cls(low=float(np.min(y)), high=float(np.max(y)))

    @classmethod
    def identity(cls) -> _MinMaxScale:
        """The map that changes nothing."""
        return cls(low=0.0, high=1.0, target_low=0.0, target_high=1.0)

    @property
    def slope(self) -> float:
        """Original units per scaled unit."""
        return max(self.high - self.low, _TINY) / (self.target_high - self.target_low)

    @property
    def offset(self) -> float:
        """Original value at scaled zero."""
        return self.low - self.target_low * self.slope

    def transform(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Original scale to target range."""
        return (np.asarray(y, dtype=np.float64) - self.offset) / self.slope

    def inverse(self, ys: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Target range to original scale."""
        return np.asarray(ys, dtype=np.float64) * self.slope + self.offset


@dataclass(frozen=True, slots=True)
class _SummaryTable:
    """A rendered-on-demand estimation summary.

    Holds the structure of a summary rather than a formatted string: a
    title, label/value metadata pairs rendered two per line, one table, and
    closing notes. Renders as fixed-width text, as HTML in a notebook, or as
    a dataframe, and imports nothing optional until a frame is asked for.

    Attributes:
        title: Headline.
        metadata: Label/value pairs describing the fit.
        columns: Column headers; the first names the row key.
        rows: Table rows; the first entry of each is the row key.
        notes: Closing lines stating what the numbers assume.
    """

    title: str
    metadata: tuple[tuple[str, str], ...] = ()
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    notes: tuple[str, ...] = ()

    _WIDTH: ClassVar[int] = 80

    def __str__(self) -> str:
        """Render as fixed-width text."""
        return self.to_text()

    def __repr__(self) -> str:
        """Render as fixed-width text, so a bare expression shows the summary."""
        return self.to_text()

    def to_text(self) -> str:
        """Render the summary as fixed-width text."""
        meta = self._metadata_lines()
        table = self._table_lines() if self.rows else []
        width = max([len(line) for line in (*meta, *table)] + [len(self.title), self._WIDTH])
        rule = "=" * width
        lines: list[str] = [self.title.center(width).rstrip(), rule, *meta]
        if table:
            lines.extend(["-" * width, *table])
        lines.append(rule)
        for note in self.notes:
            lines.extend(textwrap.wrap(note, width=width) or [""])
        return "\n".join(lines)

    def _metadata_lines(self) -> list[str]:
        if not self.metadata:
            return []
        label_width = max(len(label) for label, _ in self.metadata) + 2
        value_width = max(len(value) for _, value in self.metadata) + 2
        cell = max(label_width + value_width, self._WIDTH // 2)
        lines: list[str] = []
        for i in range(0, len(self.metadata), 2):
            cells = [
                f"{label + ':':<{cell - value_width}}{value:>{value_width - 2}}  "
                for label, value in self.metadata[i : i + 2]
            ]
            lines.append("".join(cells).rstrip())
        return lines

    def _table_lines(self) -> list[str]:
        name_width = max([len(self.columns[0])] + [len(row[0]) for row in self.rows]) + 2
        widths = [
            max([len(self.columns[j])] + [len(row[j]) for row in self.rows]) + 3
            for j in range(1, len(self.columns))
        ]
        header = self.columns[0].ljust(name_width) + "".join(
            head.rjust(w) for head, w in zip(self.columns[1:], widths, strict=True)
        )
        body = [
            row[0].ljust(name_width)
            + "".join(value.rjust(w) for value, w in zip(row[1:], widths, strict=True))
            for row in self.rows
        ]
        return [header.rstrip(), *(line.rstrip() for line in body)]

    def _repr_html_(self) -> str:
        """Render as a self-contained ``<div>`` for Jupyter."""
        esc = self._escape
        border = "border-bottom:1px solid #999;"
        parts = [
            '<div style="font-family:ui-monospace,Menlo,monospace;font-size:0.85em">',
            f'<div style="font-weight:600;padding:2px 0;{border}">{esc(self.title)}</div>',
        ]
        if self.metadata:
            cells = "".join(
                f'<tr><td style="padding:1px 12px 1px 0;color:#555">{esc(label)}</td>'
                f'<td style="padding:1px 24px 1px 0;text-align:right">{esc(value)}</td></tr>'
                for label, value in self.metadata
            )
            parts.append(f'<table style="border-collapse:collapse">{cells}</table>')
        if self.rows:
            head = "".join(
                f'<th style="padding:2px 10px;text-align:right;{border}">{esc(c)}</th>'
                for c in self.columns
            )
            body = "".join(
                "<tr>"
                + "".join(
                    f'<td style="padding:1px 10px;text-align:{"left" if i == 0 else "right"}">'
                    f"{esc(v)}</td>"
                    for i, v in enumerate(row)
                )
                + "</tr>"
                for row in self.rows
            )
            parts.append(
                f'<table style="border-collapse:collapse;margin-top:4px">'
                f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
            )
        for note in self.notes:
            parts.append(f'<div style="color:#555;padding-top:3px">{esc(note)}</div>')
        parts.append("</div>")
        return "".join(parts)

    def to_pandas(self) -> pd.DataFrame:
        """The table as a :class:`pandas.DataFrame`, metadata in ``attrs``.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise ImportError("to_pandas requires pandas: pip install 'ns-sdn[pandas]'") from error
        frame = pd.DataFrame(
            [row[1:] for row in self.rows],
            index=[row[0] for row in self.rows],
            columns=list(self.columns[1:]),
        )
        frame.index.name = self.columns[0] if self.columns else None
        frame.attrs["title"] = self.title
        frame.attrs["metadata"] = dict(self.metadata)
        frame.attrs["notes"] = list(self.notes)
        return frame

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
