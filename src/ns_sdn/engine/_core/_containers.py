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
r"""Reusable value objects shared by internal mechanisms and public results.

Five records, none of which knows what a model is: the penalty weights
and the training stage that make up a schedule, the information-criteria
triple, the affine scaling map, and the summary table. Each is a frozen
dataclass or a named tuple, validated at construction where it carries
numbers, and importable from here by the engine while the public modules
re-bind the user-facing ones (:class:`~ns_sdn.spectral_decomposition.Penalties`,
:class:`~ns_sdn.spectral_decomposition.Stage`,
:class:`~ns_sdn.summary.InformationCriteria`,
:class:`~ns_sdn.summary.SummaryTable`) under their public names.

Example:
    >>> stage = _Stage(1e-3, 10, _Penalties(amplitude=0.0))
    >>> stage.penalties.frequency, _MinMaxScale.fit(np.array([0.0, 4.0])).slope
    (0.0008, 2.0)
"""

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
    r"""Weights on the drift and sparsity penalties added to the prediction loss.

    Each is a non-negative multiplier :math:`\lambda` on a term evaluated
    on the scaled series over the :math:`r` emission rows and :math:`K`
    components; zero removes the term from the graph entirely. With
    :math:`\bar A_{t,k} = g_{t,k} A_{t,k}` the gated amplitude, the terms
    are

    .. math::

       \mathcal{P}_A = \frac{1}{(r-1)K}\sum_{t,k} (\bar A_{t,k} - \bar A_{t-1,k})^2,
       \qquad
       \mathcal{P}_\omega = \frac{1}{(r-1)K}\sum_{t,k} (\omega_{t,k} - \omega_{t-1,k})^2,

    .. math::

       \mathcal{P}_\theta = \frac{1}{(r-2)K}\sum_{t,k}
       (\theta_{t,k} - 2\theta_{t-1,k} + \theta_{t-2,k})^2,
       \qquad
       \mathcal{P}_B = \frac{1}{r-1}\sum_{t} (B_t - B_{t-1})^2,

    .. math::

       \mathcal{P}_{\mathrm{sp}} = \sum_{k}\Bigl(\frac{1}{r}\sum_t \bar A_{t,k}^2\Bigr)^{1/2},
       \qquad
       \mathcal{P}_{\mathrm{spec}} = \frac{1}{F}\sum_{f}
       \Bigl(|\mathcal{F}\hat y|_f^2 - |\mathcal{F}y|_f^2\Bigr)^2,

    and the total loss is the prediction term plus
    :math:`\lambda_A \mathcal{P}_A + \lambda_\omega \mathcal{P}_\omega
    + \lambda_\theta \mathcal{P}_\theta + \lambda_B \mathcal{P}_B
    + \lambda_{\mathrm{sp}} \mathcal{P}_{\mathrm{sp}}
    + \lambda_{\mathrm{spec}} \mathcal{P}_{\mathrm{spec}}`. The first two
    are the TVP-VAR intuition that spectral parameters move slowly and are
    the only ones the published run used; the defaults are its stage-A
    weights.

    Attributes:
        amplitude: :math:`\lambda_A`, on the squared first differences of
            the gated amplitude paths.
        frequency: :math:`\lambda_\omega`, on the squared first differences
            of the instantaneous frequencies.
        phase_curvature: :math:`\lambda_\theta`, on the squared second
            differences of the cumulative phase; a smoothness prior on the
            frequency itself.
        trend: :math:`\lambda_B`, on the squared first differences of the
            trend, which keeps the trend head from absorbing the
            oscillation.
        sparsity: :math:`\lambda_{\mathrm{sp}}`, the group
            (:math:`\ell_{2,1}`) penalty that silences whole components
            rather than shrinking all of them.
        spectrum: :math:`\lambda_{\mathrm{spec}}`, on the mean squared
            difference between the rFFT power spectra of the target and the
            prediction over :math:`F` frequency bins.

    Raises:
        SpecificationError: If any weight is negative, non-finite or not a
            number.

    Example:
        >>> from ns_sdn.spectral_decomposition import Penalties
        >>> Penalties(amplitude=3e-4, frequency=1.5e-3).sparsity
        0.0
        >>> Penalties()  # doctest: +ELLIPSIS
        Penalties(amplitude=0.0002, frequency=0.0008, phase_curvature=0.0, trend=0.0, ...)
        >>> Penalties(trend=-1.0)
        Traceback (most recent call last):
            ...
        ns_sdn.exceptions.SpecificationError: trend must be non-negative and finite; got -1.0.
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
    r"""One stage of the training schedule: a fresh Adam run at one learning rate.

    A schedule is a tuple of stages run in order on the same network;
    each constructs its own optimizer (so Adam's moment estimates restart)
    and may carry its own penalty weights. The published schedule is two
    stages, :math:`(\eta, \text{steps}) = (10^{-3}, 1200)` then
    :math:`(4 \times 10^{-4}, 1800)`, with heavier drift penalties in the
    second.

    Attributes:
        learning_rate: Adam step size :math:`\eta`, strictly positive.
        steps: Full-sequence gradient steps, at least one; each step is
            one forward pass over the whole sample and one backward pass.
        penalties: Weights for this stage, or ``None`` to use the model's
            own ``penalties``.

    Raises:
        SpecificationError: If the rate is not a positive finite number,
            the steps are not a positive integer, or ``penalties`` is
            neither a record nor ``None``.

    Example:
        >>> from ns_sdn.spectral_decomposition import Stage, Penalties
        >>> Stage(learning_rate=1e-3, steps=1200).penalties is None
        True
        >>> Stage(1e-3, 10, Penalties(frequency=0.0)).penalties.frequency
        0.0
        >>> Stage(0.0, 10)
        Traceback (most recent call last):
            ...
        ns_sdn.exceptions.SpecificationError: learning_rate must be strictly positive; got 0.
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
    r"""Model-selection criteria computed from a fitted log-likelihood.

    For a log-likelihood :math:`\ell`, :math:`p` parameters and an
    effective sample of :math:`m` observations,

    .. math::

       \mathrm{AIC} = -2\ell + 2p,
       \qquad
       \mathrm{BIC} = -2\ell + p\log m,
       \qquad
       \mathrm{HQIC} = -2\ell + 2p\log\log m.

    Lower is better for each. For a network, :math:`p` is the nominal
    weight count and :math:`\ell` a concentrated likelihood at a
    penalized optimum, so the criteria order fits that share an engine
    configuration and differ in :math:`K`; they do not compare a network
    against an ARIMA.

    Attributes:
        aic: Akaike information criterion.
        bic: Bayesian (Schwarz) information criterion.
        hqic: Hannan-Quinn information criterion.
    """

    aic: float
    bic: float
    hqic: float

    @classmethod
    def from_likelihood(cls, llf: float, nobs: int, n_params: float) -> _InformationCriteria:
        r"""Compute all three criteria.

        Args:
            llf: The log-likelihood :math:`\ell`.
            nobs: The effective sample size :math:`m`, at least three so
                that :math:`\log\log m` is defined and positive.
            n_params: The parameter count :math:`p`, non-negative.

        Returns:
            The triple.

        Raises:
            SpecificationError: If ``nobs < 3`` or ``n_params`` is negative.

        Example:
            >>> from ns_sdn.summary import InformationCriteria
            >>> crit = InformationCriteria.from_likelihood(-100.0, 200, 3)
            >>> round(crit.aic, 1), round(crit.bic, 3), round(crit.hqic, 3)
            (206.0, 215.895, 210.004)
            >>> InformationCriteria.from_likelihood(-100.0, 2, 3)
            Traceback (most recent call last):
                ...
            ns_sdn.exceptions.SpecificationError: nobs must be >= 3 to form criteria; got 2.
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
    r"""An affine map between the original scale and a target range, and its inverse.

    .. math::

       \tilde y = \frac{y - \ell}{h - \ell}\,(\tau_h - \tau_\ell) + \tau_\ell
       = \frac{y - c}{a},
       \qquad
       a = \frac{h - \ell}{\tau_h - \tau_\ell}, \quad c = \ell - \tau_\ell\, a,

    with :math:`(\ell, h)` the sample range and :math:`(\tau_\ell, \tau_h)`
    the target range, :math:`[-1, 1]` by default. ``slope`` is :math:`a`
    (original units per scaled unit) and ``offset`` is :math:`c` (the
    original value at scaled zero), which is the form the predictor uses to
    undo the scaling on every path: trend and emission get both, amplitudes
    and components the slope alone. A degenerate sample (``high == low``)
    gets a :math:`10^{-12}` floor on the range rather than a division by
    zero.

    Attributes:
        low: Sample minimum :math:`\ell`.
        high: Sample maximum :math:`h`.
        target_low: Lower edge of the target range :math:`\tau_\ell`.
        target_high: Upper edge of the target range :math:`\tau_h`.

    Example:
        >>> scale = _MinMaxScale.fit(np.array([2.0, 4.0, 6.0]))
        >>> scale.transform(np.array([2.0, 4.0, 6.0]))
        array([-1.,  0.,  1.])
        >>> scale.inverse(scale.transform(np.array([3.0])))
        array([3.])
    """

    low: float
    high: float
    target_low: float = -1.0
    target_high: float = 1.0

    @classmethod
    def fit(cls, y: npt.NDArray[np.float64]) -> _MinMaxScale:
        """Read the range off a sample, targeting ``[-1, 1]``.

        Args:
            y: The sample, any shape, finite.

        Returns:
            The map with ``low = min(y)`` and ``high = max(y)``.

        Example:
            >>> _MinMaxScale.fit(np.array([2.0, 4.0]))
            _MinMaxScale(low=2.0, high=4.0, target_low=-1.0, target_high=1.0)
        """
        return cls(low=float(np.min(y)), high=float(np.max(y)))

    @classmethod
    def identity(cls) -> _MinMaxScale:
        """The map that changes nothing: unit slope, zero offset.

        What a model under ``scaling="none"`` carries, so the predictor's
        inverse scaling is uniform across both choices.

        Example:
            >>> _MinMaxScale.identity().slope, _MinMaxScale.identity().offset
            (1.0, 0.0)
        """
        return cls(low=0.0, high=1.0, target_low=0.0, target_high=1.0)

    @property
    def slope(self) -> float:
        r"""Original units per scaled unit, :math:`a = (h - \ell)/(\tau_h - \tau_\ell)`."""
        return max(self.high - self.low, _TINY) / (self.target_high - self.target_low)

    @property
    def offset(self) -> float:
        r"""Original value at scaled zero, :math:`c = \ell - \tau_\ell\, a`."""
        return self.low - self.target_low * self.slope

    def transform(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Original scale to target range, :math:`\tilde y = (y - c)/a`.

        Args:
            y: Values on the original scale, any shape.

        Returns:
            The same shape in training units.
        """
        return (np.asarray(y, dtype=np.float64) - self.offset) / self.slope

    def inverse(self, ys: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Target range to original scale, :math:`y = a\tilde y + c`.

        Args:
            ys: Values in training units, any shape.

        Returns:
            The same shape on the original scale.
        """
        return np.asarray(ys, dtype=np.float64) * self.slope + self.offset


@dataclass(frozen=True, slots=True)
class _SummaryTable:
    """A rendered-on-demand estimation summary.

    Holds the structure of a summary rather than a formatted string: a
    title, label/value metadata pairs rendered two per line, one table, and
    closing notes. Renders as fixed-width text (:meth:`to_text`, which is
    also ``str`` and ``repr``), as HTML in a notebook (``_repr_html_``),
    or as a dataframe (:meth:`to_pandas`), and imports nothing optional
    until a frame is asked for. Every value is already a string: the
    producer decides the number formatting, the table decides the layout.

    The text layout is the econometric convention: a centred title, a
    rule, the metadata in two columns of ``label: value`` pairs, a rule,
    the table with its first column left-aligned and the rest
    right-aligned, a rule, and the notes word-wrapped to the table width
    (at least 80 characters).

    Attributes:
        title: Headline.
        metadata: Label/value pairs describing the fit, laid out two per
            line in the order given.
        columns: Column headers; the first names the row key and may be
            empty.
        rows: Table rows; the first entry of each is the row key and the
            rest align with ``columns[1:]``.
        notes: Closing lines stating what the numbers assume.

    Example:
        >>> from ns_sdn.summary import SummaryTable
        >>> table = SummaryTable(
        ...     title="Demo",
        ...     metadata=(("Observations", "99"), ("Parameters", "7")),
        ...     columns=("", "mean", "share"),
        ...     rows=(("component 1", "0.50", "0.800"), ("component 2", "0.10", "0.200")),
        ...     notes=("Shares sum to one.",),
        ... )
        >>> print(table)  # doctest: +NORMALIZE_WHITESPACE
                                               Demo
        ================================================================================
        Observations:   99  Parameters:    7
        --------------------------------------------------------------------------------
                       mean   share
        component 1    0.50   0.800
        component 2    0.10   0.200
        ================================================================================
        Shares sum to one.
        >>> table.to_pandas().loc["component 2", "share"]
        '0.200'
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
        """Render the summary as fixed-width text.

        Returns:
            The lines joined with newlines, no trailing newline; the width
            is the widest line, the title or 80, whichever is largest.
        """
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
        """Lay the metadata pairs out two per line, labels left and values right.

        Returns:
            One string per line, right-stripped; empty when there is no
            metadata.
        """
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
        """Lay the table out with a left-aligned key column and right-aligned values.

        Returns:
            The header line followed by one line per row, right-stripped.
        """
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
        """Render as a self-contained ``<div>`` for Jupyter.

        Inline styles only, so the rendering does not depend on the
        notebook's stylesheet; every string is HTML-escaped.

        Returns:
            The HTML fragment.
        """
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

        Rows are indexed by their key, columns are ``columns[1:]``, every
        cell is the string the producer formatted, and ``frame.attrs``
        carries ``title``, ``metadata`` (as a dict) and ``notes``.

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
        """Escape the three characters that would break the HTML rendering.

        Args:
            text: Any cell, label or note.

        Returns:
            The text with ``&``, ``<`` and ``>`` replaced by entities.
        """
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
