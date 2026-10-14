# filepath: /src/ns_sdn/spectral_state.py
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
r"""Spectral-state coordinates: the decomposition read as a time-local measure.

At each step the fitted network places power ``p_{t,k} = (|w_k| g_{t,k}
A_{t,k})^2``, the squared amplitude of its contribution, at frequency
``\omega_{t,k}``. Normalizing gives a probability measure on
the frequency axis,

.. math::

   \mu_t = \sum_{k=1}^{K} w_{t,k}\,\delta(\omega - \omega_{t,k}),
   \qquad w_{t,k} = p_{t,k} \Big/ \sum_j p_{t,j},

and every coordinate here is a functional of ``\mu_t``: its position
(the power-weighted centroid), its spread (the weighted dispersion), its
effective rank (the exponential of the spectral entropy, and the inverse
participation ratio), and its motion (the 1-Wasserstein distance between
consecutive measures per unit time, split into the part due to
frequencies moving under frozen weights and the part due to weights
moving under frozen frequencies). The two-axis reading of spectral
compression is a joint contraction of dispersion and effective count. A
contribution cross-correlation and the disentanglement score it implies
are reported alongside, because the measure alone cannot tell converged
frequencies from redundant components.

These are diagnostics of a model trained on the prediction criterion
alone; nothing here enters a loss. Frequencies and weights of a component
whose weight falls below ``active_threshold`` are unidentified, and the
active mask says which rows a per-component reading should trust.

Example:
    >>> import numpy as np
    >>> from ns_sdn.spectral_decomposition import NSSDN, Stage
    >>> y = np.sin(0.3 * np.arange(240.0)) + 0.5 * np.sin(0.07 * np.arange(240.0))
    >>> res = NSSDN(y, k=3, state_dim=8, schedule=(Stage(1e-2, 40),), seed=0).fit()
    >>> state = res.spectral_state()
    >>> state.weight.shape, state.velocity.shape, bool(np.all(state.effective_count <= 3.0))
    ((240, 3), (239,), True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .engine._core._defaults import _ACTIVE_WEIGHT, _TINY
from .engine._core._validators import _validate_unit_interval
from .engine._internals._mixins import _SummaryMixin
from .summary import SummaryTable

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from .spectral_decomposition import NSSDNResult

__all__ = ["SpectralState"]


def _wasserstein_1d(
    support_a: npt.NDArray[np.float64],
    weight_a: npt.NDArray[np.float64],
    support_b: npt.NDArray[np.float64],
    weight_b: npt.NDArray[np.float64],
) -> float:
    """``W_1`` between two discrete measures on the line, by the CDF integral."""
    grid = np.sort(np.concatenate([support_a, support_b]))
    order_a = np.argsort(support_a)
    order_b = np.argsort(support_b)
    cdf_a = np.cumsum(weight_a[order_a])[
        np.searchsorted(support_a[order_a], grid, side="right") - 1
    ]
    cdf_b = np.cumsum(weight_b[order_b])[
        np.searchsorted(support_b[order_b], grid, side="right") - 1
    ]
    return float(np.sum(np.abs(cdf_a - cdf_b)[:-1] * np.diff(grid)))


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SpectralState(_SummaryMixin):
    """The spectral-state coordinates of a fitted decomposition.

    Rows follow the result's path rows (``r = n`` under the filtered
    alignment, ``n - 1`` under the predicted one). Velocities have one
    fewer row and sit between consecutive rows.

    Attributes:
        time: ``(nobs,)`` the times of the rows.
        frequency: ``(nobs, k)`` instantaneous frequencies.
        weight: ``(nobs, k)`` normalized power ``w_{t,k}``.
        centroid: ``(nobs,)`` power-weighted mean frequency.
        dispersion: ``(nobs,)`` power-weighted frequency standard deviation.
        entropy: ``(nobs,)`` spectral entropy ``-sum_k w log w``.
        effective_count: ``(nobs,)`` ``exp(entropy)`` in ``[1, k]``.
        participation_ratio: ``(nobs,)`` inverse participation ratio ``1 / sum_k w^2``.
        velocity: ``(nobs - 1,)`` ``W_1(mu_t, mu_{t-1}) / dt``.
        velocity_frequency: ``(nobs - 1,)`` the part from frequencies moving under frozen weights.
        velocity_mass: ``(nobs - 1,)`` the part from weights moving under frozen frequencies.
        active: ``(nobs, k)`` whether ``w_{t,k}`` is at least ``active_threshold``.
        contribution_correlation: ``(k, k)`` correlation of the component contributions.
        disentanglement_score: One minus the off-diagonal share of squared correlation,
            in ``[0, 1]``.
        active_threshold: The weight floor used for ``active``.
    """

    time: npt.NDArray[np.float64] = field(repr=False)
    frequency: npt.NDArray[np.float64] = field(repr=False)
    weight: npt.NDArray[np.float64] = field(repr=False)
    centroid: npt.NDArray[np.float64] = field(repr=False)
    dispersion: npt.NDArray[np.float64] = field(repr=False)
    entropy: npt.NDArray[np.float64] = field(repr=False)
    effective_count: npt.NDArray[np.float64] = field(repr=False)
    participation_ratio: npt.NDArray[np.float64] = field(repr=False)
    velocity: npt.NDArray[np.float64] = field(repr=False)
    velocity_frequency: npt.NDArray[np.float64] = field(repr=False)
    velocity_mass: npt.NDArray[np.float64] = field(repr=False)
    active: npt.NDArray[np.bool_] = field(repr=False)
    contribution_correlation: npt.NDArray[np.float64] = field(repr=False)
    disentanglement_score: float
    active_threshold: float

    @classmethod
    def from_result(
        cls, result: NSSDNResult, *, active_threshold: float | None = None
    ) -> SpectralState:
        """Compute every coordinate from a fitted result.

        Args:
            result: A fitted :class:`~ns_sdn.spectral_decomposition.NSSDNResult`.
            active_threshold: Weight floor in ``[0, 1]``; the package default when ``None``.

        Returns:
            The populated state.

        Raises:
            SpecificationError: If the threshold lies outside ``[0, 1]``.
        """
        threshold = (
            _ACTIVE_WEIGHT
            if active_threshold is None
            else _validate_unit_interval(active_threshold, "active_threshold")
        )
        frequency = np.asarray(result.frequency, dtype=np.float64)
        power = np.asarray(result.effective_amplitude, dtype=np.float64) ** 2
        weight = power / np.maximum(power.sum(axis=1, keepdims=True), _TINY)
        centroid = np.sum(weight * frequency, axis=1)
        dispersion = np.sqrt(np.sum(weight * (frequency - centroid[:, None]) ** 2, axis=1))
        entropy = -np.sum(weight * np.log(np.maximum(weight, _TINY)), axis=1)
        effective_count = np.exp(entropy)
        participation_ratio = 1.0 / np.maximum(np.sum(weight**2, axis=1), _TINY)

        dt = np.diff(result.time[-frequency.shape[0] :]) if frequency.shape[0] > 1 else np.zeros(0)
        n = frequency.shape[0]
        velocity = np.zeros(max(n - 1, 0))
        velocity_frequency = np.zeros(max(n - 1, 0))
        velocity_mass = np.zeros(max(n - 1, 0))
        for i in range(n - 1):
            step = max(float(dt[i]), _TINY)
            velocity[i] = (
                _wasserstein_1d(frequency[i + 1], weight[i + 1], frequency[i], weight[i]) / step
            )
            velocity_frequency[i] = (
                _wasserstein_1d(frequency[i + 1], weight[i], frequency[i], weight[i]) / step
            )
            velocity_mass[i] = (
                _wasserstein_1d(frequency[i], weight[i + 1], frequency[i], weight[i]) / step
            )

        contributions = np.asarray(result.components, dtype=np.float64)
        spread = contributions.std(axis=0)
        correlation = np.eye(contributions.shape[1])
        if n > 1:
            centered = contributions - contributions.mean(axis=0)
            cov = centered.T @ centered / n
            denominator = np.outer(spread, spread)
            with np.errstate(invalid="ignore", divide="ignore"):
                correlation = np.where(
                    denominator > _TINY, cov / np.maximum(denominator, _TINY), 0.0
                )
            np.fill_diagonal(correlation, 1.0)
        squared = correlation**2
        total = float(squared.sum())
        off_diagonal = total - float(np.trace(squared))
        score = 1.0 - off_diagonal / total if total > 0.0 else 1.0

        return cls(
            time=np.asarray(result.time[-n:], dtype=np.float64),
            frequency=frequency,
            weight=weight,
            centroid=centroid,
            dispersion=dispersion,
            entropy=entropy,
            effective_count=effective_count,
            participation_ratio=participation_ratio,
            velocity=velocity,
            velocity_frequency=velocity_frequency,
            velocity_mass=velocity_mass,
            active=weight >= threshold,
            contribution_correlation=correlation,
            disentanglement_score=float(score),
            active_threshold=float(threshold),
        )

    @property
    def nobs(self) -> int:
        """Rows."""
        return int(self.frequency.shape[0])

    @property
    def k(self) -> int:
        """Components."""
        return int(self.frequency.shape[1])

    def to_dict(self) -> dict[str, npt.NDArray[np.float64]]:
        """The scalar coordinates as aligned arrays; velocities padded with ``nan`` in row 0."""
        pad = np.full(1, np.nan)
        return {
            "centroid": self.centroid,
            "dispersion": self.dispersion,
            "entropy": self.entropy,
            "effective_count": self.effective_count,
            "participation_ratio": self.participation_ratio,
            "velocity": np.concatenate([pad, self.velocity]),
            "velocity_frequency": np.concatenate([pad, self.velocity_frequency]),
            "velocity_mass": np.concatenate([pad, self.velocity_mass]),
            "n_active": self.active.sum(axis=1).astype(np.float64),
        }

    def to_pandas(self, *, index: npt.ArrayLike | None = None) -> pd.DataFrame:
        """The scalar coordinates as a :class:`pandas.DataFrame`.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise ImportError("to_pandas requires pandas: pip install 'ns-sdn[pandas]'") from error
        frame = pd.DataFrame(self.to_dict())
        if index is not None:
            frame.index = pd.Index(np.asarray(index))
        frame.index.name = "obs"
        return frame

    def _summary_table(self) -> SummaryTable:
        def span(values: npt.NDArray[np.float64]) -> str:
            return f"{values.min():.4g} / {values.mean():.4g} / {values.max():.4g}"

        rows = (
            ("centroid", span(self.centroid)),
            ("dispersion", span(self.dispersion)),
            ("effective count", span(self.effective_count)),
            ("participation ratio", span(self.participation_ratio)),
            ("velocity", span(self.velocity) if self.velocity.size else "n/a"),
            (
                "velocity (frequency)",
                span(self.velocity_frequency) if self.velocity.size else "n/a",
            ),
            ("velocity (mass)", span(self.velocity_mass) if self.velocity.size else "n/a"),
        )
        return SummaryTable(
            title=f"Spectral state (k={self.k})",
            metadata=(
                ("Rows", f"{self.nobs}"),
                ("Disentanglement", f"{self.disentanglement_score:.3f}"),
                ("Active threshold", f"{self.active_threshold:g}"),
                ("Mean active", f"{self.active.sum(axis=1).mean():.2f}"),
            ),
            columns=("coordinate", "min / mean / max"),
            rows=rows,
            notes=(
                "Diagnostics of a model trained on the prediction criterion alone; nothing here "
                "enters a loss.",
                "Identifiability degrades as frequencies converge: compression is also the regime "
                "of weakest "
                "disentanglement, and per-component readings of inactive rows are not meaningful.",
            ),
        )
