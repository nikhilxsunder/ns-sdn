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

At each step the fitted network places power

.. math::

   p_{t,k} = \bigl(|w_k|\, g_{t,k}\, A_{t,k}\bigr)^2,

the squared amplitude of its contribution, at frequency
:math:`\omega_{t,k}`. Normalizing gives a probability measure on the
frequency axis,

.. math::

   \mu_t = \sum_{k=1}^{K} w_{t,k}\,\delta_{\omega_{t,k}},
   \qquad
   w_{t,k} = \frac{p_{t,k}}{\sum_{j=1}^{K} p_{t,j}},

and every coordinate here is a functional of :math:`\mu_t`:

* its position, the power-weighted centroid
  :math:`\bar\omega_t = \sum_k w_{t,k}\,\omega_{t,k}`;
* its spread, the power-weighted dispersion
  :math:`s_t = \bigl(\sum_k w_{t,k}(\omega_{t,k} - \bar\omega_t)^2\bigr)^{1/2}`;
* its effective rank, the exponential of the spectral entropy
  :math:`N_t = \exp H_t` with :math:`H_t = -\sum_k w_{t,k}\log w_{t,k}`,
  and the inverse participation ratio
  :math:`P_t = 1 / \sum_k w_{t,k}^2`, both in :math:`[1, K]`;
* its motion, the 1-Wasserstein distance between consecutive measures
  per unit time, :math:`v_t = W_1(\mu_t, \mu_{t-1}) / \Delta t_t`, split
  into the distance due to frequencies moving under frozen weights and
  the distance due to weights moving under frozen frequencies.

The two-axis reading of spectral compression is a joint contraction of
dispersion and effective count. A contribution cross-correlation and the
disentanglement score it implies are reported alongside, because the
measure alone cannot tell converged frequencies from redundant
components: two components at the same frequency with the same phase are
one component counted twice, and the score is what says so.

These are diagnostics of a model trained on the prediction criterion
alone; nothing here enters a loss. Frequencies and weights of a component
whose weight falls below ``active_threshold`` are unidentified (a silent
component's frequency head receives no gradient from the prediction
term), and the active mask says which rows a per-component reading
should trust.

References:
    Villani, C. (2009). *Optimal Transport: Old and New*. Springer.
    Chapter 6, for the one-dimensional Wasserstein distance as the
    :math:`L^1` distance between distribution functions.

    Huang, N. E. et al. (1998). The empirical mode decomposition and the
    Hilbert spectrum for nonlinear and non-stationary time series
    analysis. *Proc. R. Soc. A*, 454, 903-995. For the instantaneous
    frequency as the object of a time-local spectral reading.

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
    r""":math:`W_1` between two discrete measures on the line, by the CDF integral.

    On :math:`\mathbb{R}` the 1-Wasserstein distance has the closed form

    .. math::

       W_1(\mu, \nu) = \int_{-\infty}^{\infty} \bigl|F_\mu(\omega) - F_\nu(\omega)\bigr|\,
       d\omega,

    with :math:`F_\mu` and :math:`F_\nu` the distribution functions. For
    two measures supported on finitely many atoms the integrand is
    piecewise constant between consecutive atoms of the pooled support,
    so the integral is a finite sum of rectangle areas: the pooled support
    is sorted, each measure's CDF is evaluated on it by a right-continuous
    step, and the absolute differences are weighted by the gaps between
    consecutive support points.

    Args:
        support_a: ``(k,)`` the atoms of the first measure, any order.
        weight_a: ``(k,)`` their masses, summing to one.
        support_b: ``(m,)`` the atoms of the second measure.
        weight_b: ``(m,)`` their masses, summing to one.

    Returns:
        The distance, in the units of the supports.

    Example:
        Two unit masses a distance :math:`0.5` apart are :math:`0.5` apart;
        moving half of one of them costs half as much:

        >>> _wasserstein_1d(np.array([1.0]), np.array([1.0]), np.array([1.5]), np.array([1.0]))
        0.5
        >>> _wasserstein_1d(
        ...     np.array([1.0, 1.5]), np.array([0.5, 0.5]), np.array([1.0]), np.array([1.0])
        ... )
        0.25
    """
    grid = np.sort(np.concatenate([support_a, support_b]))
    order_a = np.argsort(support_a)
    order_b = np.argsort(support_b)
    # Index 0 of each padded CDF is the mass below the smallest atom, zero.
    cdf_a = np.concatenate([[0.0], np.cumsum(weight_a[order_a])])[
        np.searchsorted(support_a[order_a], grid, side="right")
    ]
    cdf_b = np.concatenate([[0.0], np.cumsum(weight_b[order_b])])[
        np.searchsorted(support_b[order_b], grid, side="right")
    ]
    return float(np.sum(np.abs(cdf_a - cdf_b)[:-1] * np.diff(grid)))


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SpectralState(_SummaryMixin):
    r"""The spectral-state coordinates of a fitted decomposition.

    Rows follow the result's path rows (:math:`r = n` under the filtered
    alignment, :math:`n - 1` under the predicted one). Velocities have one
    fewer row and sit between consecutive rows: ``velocity[i]`` is the
    distance from row :math:`i` to row :math:`i + 1` divided by the time
    between them.

    The velocity split is two counterfactual distances rather than an
    additive decomposition,

    .. math::

       v^{\omega}_t = \frac{1}{\Delta t_t}\,
       W_1\!\Bigl(\sum_k w_{t-1,k}\,\delta_{\omega_{t,k}},\; \mu_{t-1}\Bigr),
       \qquad
       v^{m}_t = \frac{1}{\Delta t_t}\,
       W_1\!\Bigl(\sum_k w_{t,k}\,\delta_{\omega_{t-1,k}},\; \mu_{t-1}\Bigr),

    so :math:`v^{\omega}_t` is what the measure would have moved had only
    the frequencies changed and :math:`v^{m}_t` had only the weights
    changed. They need not sum to :math:`v_t`; read them as attributions.

    The disentanglement score is formed from the correlation matrix
    :math:`\rho` of the per-component contributions
    :math:`c_{t,k} = w_k g_{t,k} A_{t,k}\sin(\theta_{t,k} + \phi_{t,k})`
    over the rows,

    .. math::

       D = 1 - \frac{\sum_{j \neq k} \rho_{jk}^2}{\sum_{j,k} \rho_{jk}^2}
       \in [0, 1],

    which is one when the contributions are mutually uncorrelated and
    falls toward zero as components duplicate one another. A component
    with no variance over the sample has its off-diagonal correlations set
    to zero.

    Attributes:
        time: ``(nobs,)`` the times of the rows.
        frequency: ``(nobs, k)`` instantaneous frequencies :math:`\omega_{t,k}`,
            radians per unit time.
        weight: ``(nobs, k)`` normalized power :math:`w_{t,k}`, each row
            summing to one.
        centroid: ``(nobs,)`` power-weighted mean frequency :math:`\bar\omega_t`.
        dispersion: ``(nobs,)`` power-weighted frequency standard deviation
            :math:`s_t`.
        entropy: ``(nobs,)`` spectral entropy :math:`H_t = -\sum_k w_{t,k}\log w_{t,k}`,
            in nats, in :math:`[0, \log K]`.
        effective_count: ``(nobs,)`` :math:`N_t = \exp H_t`, in :math:`[1, K]`.
        participation_ratio: ``(nobs,)`` inverse participation ratio
            :math:`P_t = 1 / \sum_k w_{t,k}^2`, in :math:`[1, K]`.
        velocity: ``(nobs - 1,)`` :math:`v_t = W_1(\mu_t, \mu_{t-1}) / \Delta t_t`.
        velocity_frequency: ``(nobs - 1,)`` :math:`v^{\omega}_t`, the part from
            frequencies moving under frozen weights.
        velocity_mass: ``(nobs - 1,)`` :math:`v^{m}_t`, the part from weights
            moving under frozen frequencies.
        active: ``(nobs, k)`` whether :math:`w_{t,k} \geq` ``active_threshold``.
        contribution_correlation: ``(k, k)`` correlation :math:`\rho` of the
            component contributions over the rows, unit diagonal.
        disentanglement_score: :math:`D`, one minus the off-diagonal share of
            squared correlation, in :math:`[0, 1]`.
        active_threshold: The weight floor used for ``active``.

    Note:
        Dispersion is reported in the units of frequency (radians per unit
        time), effective count and participation ratio in components, and
        velocity in radians per unit time per unit time. Entropy uses the
        natural logarithm, so ``effective_count`` is ``exp(entropy)``
        exactly and reaches :math:`K` only when the power is spread
        uniformly.

    See Also:
        * :meth:`ns_sdn.spectral_decomposition.NSSDNResult.spectral_state`
          -- the usual way to obtain one.
        * :class:`~ns_sdn.summary.SummaryTable` -- what :meth:`summary`
          returns.

    Example:
        A single pure sinusoid fitted with one component has a degenerate
        measure: one atom, zero dispersion, effective count one:

        >>> import numpy as np
        >>> from ns_sdn.spectral_decomposition import NSSDN, Stage
        >>> y = np.sin(0.3 * np.arange(160.0))
        >>> res = NSSDN(y, k=1, state_dim=8, schedule=(Stage(1e-2, 20),), seed=0).fit()
        >>> state = res.spectral_state()
        >>> bool(np.allclose(state.weight, 1.0)), bool(np.allclose(state.dispersion, 0.0))
        (True, True)
        >>> bool(np.allclose(state.effective_count, 1.0)), state.disentanglement_score
        (True, 1.0)
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
        r"""Compute every coordinate from a fitted result.

        The weights are formed from the result's ``effective_amplitude``
        squared, with a :math:`10^{-12}` floor on the row sums so a row of
        silent components gives a uniform-in-nothing zero weight rather
        than a division by zero; the entropy floors each weight the same
        way inside the logarithm. The velocity loop is :math:`O(r K \log K)`
        and the correlation :math:`O(r K^2)`.

        Args:
            result: A fitted :class:`~ns_sdn.spectral_decomposition.NSSDNResult`.
            active_threshold: Weight floor in :math:`[0, 1]`; the package
                default (:math:`10^{-2}`) when ``None``.

        Returns:
            The populated state, with ``nobs`` equal to the result's path
            rows.

        Raises:
            SpecificationError: If the threshold lies outside :math:`[0, 1]`.

        Example:
            >>> import numpy as np
            >>> from ns_sdn.spectral_decomposition import NSSDN, Stage
            >>> y = np.sin(0.3 * np.arange(160.0))
            >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 20),), seed=0).fit()
            >>> state = SpectralState.from_result(res, active_threshold=0.05)
            >>> state.active_threshold, state.active.shape, state.active.dtype
            (0.05, (160, 2), dtype('bool'))
            >>> SpectralState.from_result(res, active_threshold=2.0)
            Traceback (most recent call last):
                ...
            ns_sdn.exceptions.SpecificationError: active_threshold must lie in [0, 1]; got 2.0.
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
        """Rows: the number of path rows the coordinates are computed at."""
        return int(self.frequency.shape[0])

    @property
    def k(self) -> int:
        r"""Components: the number of atoms in every :math:`\mu_t`."""
        return int(self.frequency.shape[1])

    def to_dict(self) -> dict[str, npt.NDArray[np.float64]]:
        """The scalar coordinates as aligned arrays of ``nobs`` rows.

        The three velocities are padded with ``nan`` in row 0 so every
        column has ``nobs`` entries and row :math:`i` of a velocity is the
        motion into row :math:`i`. ``n_active`` is the number of active
        components per row as a float.

        Returns:
            Arrays keyed ``centroid, dispersion, entropy, effective_count,
            participation_ratio, velocity, velocity_frequency, velocity_mass,
            n_active``.

        Example:
            >>> import numpy as np
            >>> from ns_sdn.spectral_decomposition import NSSDN, Stage
            >>> y = np.sin(0.3 * np.arange(160.0))
            >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 20),), seed=0).fit()
            >>> out = res.spectral_state().to_dict()
            >>> len(out), "n_active" in out, bool(np.isnan(out["velocity"][0]))
            (9, True, True)
            >>> out["velocity"].shape, bool(out["n_active"].max() <= 2.0)
            ((160,), True)
        """
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

        One column per key of :meth:`to_dict`, ``nobs`` rows, the index
        named ``obs``.

        Args:
            index: Optional row index of length ``nobs``, typically the
                dates of the path rows (``dates[-state.nobs:]``).

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
        frame = pd.DataFrame(self.to_dict())
        if index is not None:
            frame.index = pd.Index(np.asarray(index))
        frame.index.name = "obs"
        return frame

    def _summary_table(self) -> SummaryTable:
        """Structured summary: the range of every scalar coordinate over the rows.

        The metadata block carries the row count, the disentanglement
        score, the active threshold and the mean number of active
        components; the table gives ``min / mean / max`` of each
        coordinate; the notes say these are diagnostics and where their
        identifiability breaks down.

        Returns:
            The :class:`~ns_sdn.summary.SummaryTable`.
        """

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
