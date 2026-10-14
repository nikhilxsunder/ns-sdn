# filepath: /src/ns_sdn/spectral_decomposition.py
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
r"""The Non-Stationary Spectral Decomposition Network: specification and result.

The model is a neural state-space representation of a scalar series
:math:`y_t` as a trend plus :math:`K` sinusoids whose amplitude,
instantaneous frequency, phase offset and participation evolve with a
latent state (IEEE-ORSS equations 1-5),

.. math::

   \hat y_t = B_t + \sum_{k=1}^{K} w_k\, g_{t,k}\, A_{t,k}\,
   \sin\!\bigl(\theta_{t,k} + \phi_{t,k}\bigr),
   \qquad
   \theta_{t,k} = \theta_{t-1,k} + \omega_{t,k}\,\Delta t_t,
   \qquad
   h_t = F_\psi(h_{t-1}, x_t, \Delta t_t),

where :math:`x_t` stacks the series input and any covariates and
:math:`\Delta t_t = t_t - t_{t-1}` is the step of the time grid. Every
spectral quantity is read off :math:`h_t \in \mathbb{R}^d` by one linear
head followed by the link that gives it its range,

.. math::

   B_t = \beta_B^\top h_t + c_B,
   \qquad
   A_{t,k} = \operatorname{softplus}\!\bigl(\beta_{A,k}^\top h_t + c_{A,k}\bigr)
   + \varepsilon_A,
   \qquad
   \omega_{t,k} = \operatorname{softplus}\!\bigl(\beta_{\omega,k}^\top h_t
   + c_{\omega,k}\bigr) + \varepsilon_\omega,

.. math::

   \phi_{t,k} = \beta_{\phi,k}^\top h_t + c_{\phi,k},
   \qquad
   g_{t,k} = \sigma\!\bigl(\beta_{g,k}^\top h_t + c_{g,k}\bigr),

with :math:`\varepsilon_A, \varepsilon_\omega > 0` small floors that keep
every component audible and every phase advancing, :math:`\sigma` the
logistic function, and :math:`w \in \mathbb{R}^K` the learned mixing
vector of the paper's implementation note. Under ``ordered_frequency``
the frequency head instead emits :math:`K + 1` logits whose softmax gaps
are accumulated inside the Nyquist band, so
:math:`\omega_{t,1} < \cdots < \omega_{t,K}` at every :math:`t`. The phase
integrates the frequency, so each component is a chirp whose frequency the
state can move; the sum of the :math:`K` chirps and the trend is the
emission.

Held under the three-object discipline: :class:`NSSDN` is the immutable,
validated specification; training belongs to the family base it
inherits; :class:`NSSDNResult` is the frozen record the user handles.
The decomposition is the point of the model, so the result carries every
path the figures of the paper are drawn from -- ``trend`` (Fig. 3),
``amplitude`` (Fig. 4, the gated :math:`g_{t,k} A_{t,k}`), ``frequency``
(Fig. 5), ``gate`` (Fig. 6), ``phase``, ``phase_offset``, ``state``, and
the per-component ``components`` whose sum with ``trend`` is the
``emission`` (Fig. 1) -- on the original scale by default and in training
units through ``decompose(scaled=True)``, with one-step ``fittedvalues``
(Fig. 2) aligned with ``endog[1:]`` whatever the alignment.

Two alignments are available. ``"filtered"``, the default and the
published run, conditions the emission at :math:`t` on :math:`y_t`: the
decomposition is a filtered reading of the observed series, the emission
reconstructs it, and the one-step forecast of :math:`y_{t+1}` is the
emission at :math:`t` (the paper's holdout protocol). ``"predicted"``
conditions the emission at :math:`t` on :math:`y_{t-1}`, so every path is
a one-step-ahead prediction and the network trains as a forecaster. The
summary says which one a result is and what that means for its numbers.

Layout. :class:`NSSDN` validates its arguments through ``_validators``
against the :mod:`~ns_sdn.typing` aliases and inherits ``_fit_family``
from ``_SpectralDecompositionModel`` in ``engine._internals._models``,
which builds the ``_SpectralNetwork`` (``_modules``), runs the staged
trainer (``_trainers``) on the ``_criterion`` (``_objectives``), and packs
a ``_SpectralFit`` whose ``predictor`` is the trained network behind a
numpy interface (``_predictors``). :class:`NSSDNResult` is assembled by
``_from_fit`` and composes ``_SummaryMixin``, ``_SeriesMixin`` and
``_ComparisonMixin`` from ``_internals._mixins``. The coordinates that
read the decomposition as a time-local spectral measure live in
:mod:`~ns_sdn.spectral_state`.

References:
    Sunder, N. & Leon, M. (2026). Non-Stationary Spectral Decomposition
    Network: Adaptive Spectral Emission Heads and Frequency Modulation.
    *2026 IEEE Opportunity Research Scholars Symposium (ORSS)*, Atlanta,
    GA, USA, pp. 1-5. https://doi.org/10.1109/ORSS71174.2026.11684282

    Sunder, N. (2026). Non-Stationary Spectral Decomposition Network for
    Econometric Time Series Forecasting. *The International FLAIRS
    Conference Proceedings*, 39(1).
    https://doi.org/10.32473/flairs.39.1.141588

    Huang, N. E. et al. (1998). The empirical mode decomposition and the
    Hilbert spectrum for nonlinear and non-stationary time series
    analysis. *Proc. R. Soc. A*, 454, 903-995.

    Sitzmann, V. et al. (2020). Implicit neural representations with
    periodic activation functions. *Advances in Neural Information
    Processing Systems*, 33.

    Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State
    Space Methods* (2nd ed.). Oxford University Press.

Example:
    A chirp with a drifting amplitude and a linear trend, fitted with a
    short schedule; the paths have one row per observation under the
    default alignment, the one-step fit one fewer:

    >>> import numpy as np
    >>> t = np.arange(300.0)
    >>> amplitude = 0.4 + 0.3 * (t > 150)
    >>> phase = np.cumsum(0.15 + 0.0004 * t)
    >>> noise = 0.02 * np.random.default_rng(0).standard_normal(300)
    >>> y = 0.002 * t + amplitude * np.sin(phase) + noise
    >>> res = NSSDN(y, k=3, state_dim=16, schedule=(Stage(1e-2, 150),), seed=0).fit()
    >>> res.emission.shape, res.amplitude.shape, res.fittedvalues.shape, res.nobs
    ((300,), (300, 3), (299,), 299)
    >>> bool(np.allclose(res.trend + res.components.sum(axis=1), res.emission, atol=1e-5))
    True
    >>> res.forecast(5).shape
    (5,)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .engine._core._containers import _MinMaxScale
from .engine._core._containers import _Penalties as Penalties
from .engine._core._containers import _Stage as Stage
from .engine._core._validators import _validate_endog, _validate_exog, _validate_time
from .engine._internals._fits import _SpectralFit
from .engine._internals._mixins import _ComparisonMixin, _SeriesMixin, _SummaryMixin
from .engine._internals._models import _SpectralDecompositionModel
from .engine._internals._predictors import _SpectralPaths, _SpectralPredictor
from .engine._internals._trainers import _StageRecord
from .exceptions import SpecificationError
from .summary import SummaryTable

if TYPE_CHECKING:  # pragma: no cover
    from .spectral_state import SpectralState

__all__ = ["NSSDN", "NSSDNResult", "Penalties", "Stage"]

for _record, _public in ((Penalties, "Penalties"), (Stage, "Stage")):
    _record.__module__ = __name__
    _record.__name__ = _record.__qualname__ = _public
del _record, _public


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class NSSDNResult(_SummaryMixin, _SeriesMixin, _ComparisonMixin):
    r"""A fitted spectral decomposition network.

    The complete decomposition the network emitted over the sample, the
    one-step conditional fit derived from it, the concentrated Gaussian
    likelihood of the one-step residuals, and the trained network as a
    callable ``predictor``. Composes rendering (``summary``, which is what
    a bare result prints), frame interop (``to_pandas``, ``to_dict`` with
    every path as a column) and criterion-named comparison
    (``information_criteria``, ``compare``).

    **Rows and alignment.** Every path has :math:`r` rows. Under the
    filtered alignment :math:`r = n`, and row :math:`t` is the emission
    that reconstructs :math:`y_t` from the state after :math:`y_t`,

    .. math::

       \hat y_t = \hat y\bigl(h_t\bigr), \qquad h_t = F_\psi(h_{t-1}, y_t),
       \qquad t = 0, \ldots, n - 1.

    Under the predicted alignment :math:`r = n - 1`, and row :math:`i` is
    the emission that predicts :math:`y_{i+1}` from the state after
    :math:`y_i`,

    .. math::

       \hat y_{i+1} = \hat y\bigl(h_{i}\bigr), \qquad h_i = F_\psi(h_{i-1}, y_i),
       \qquad i = 0, \ldots, n - 2.

    ``observed`` is the slice of ``endog`` the rows align with.
    ``fittedvalues`` always has :math:`n - 1` rows aligned with
    ``endog[1:]`` and is the one-step conditional prediction of each
    observation: the emission rows under the predicted alignment, the
    emissions through :math:`n - 2` under the filtered one (the paper's
    ``yhat[split - 1 : -1]`` against ``y[split:]``).

    **Likelihood.** With :math:`e_t = y_t - \hat y_{t \mid t-1}` the
    one-step residuals over the effective sample of :math:`m = n - 1`
    observations, the reported likelihood is the Gaussian likelihood
    concentrated at the residual variance,

    .. math::

       \hat\sigma^2 = \frac{1}{m}\sum_{t=1}^{n-1} e_t^2,
       \qquad
       \ell = -\frac{m}{2}\bigl(\log 2\pi + \log \hat\sigma^2 + 1\bigr),

    which is what the information criteria are formed from.

    Attributes:
        endog: ``(n,)`` the full observed series on its original scale.
        time: ``(n,)`` the time grid, the integers unless one was given.
        exog: ``(n, p)`` the covariates used, or ``None``.
        k: Number of components :math:`K`.
        state_dim: Latent dimension :math:`d`.
        alignment: ``"filtered"`` or ``"predicted"``.
        transition: Transition kind, ``"gru"`` or ``"timeaware"``.
        ordered_frequency: Whether frequencies are strictly increasing in
            :math:`k` at every step.
        gating: Whether a gate head was fitted; when not, ``gate`` is ones.
        objective: Prediction loss, ``"mse"`` or ``"huber"``.
        scaling: Scaling choice, ``"minmax"`` or ``"none"``.
        scale: The affine map between training units and the original scale.
        emission: ``(r,)`` the synthesized series :math:`\hat y_t`,
            equal to ``trend + components.sum(axis=1)``.
        fittedvalues: ``(n - 1,)`` one-step conditional predictions of
            ``endog[1:]``.
        resid: ``(n - 1,)`` ``endog[1:] - fittedvalues``.
        llf: Concentrated Gaussian log-likelihood :math:`\ell` of ``resid``.
        nobs: Effective sample size :math:`m = n - 1`.
        n_params: Trainable parameter count of the network.
        sigma2: Residual variance :math:`\hat\sigma^2`.
        trend: ``(r,)`` trend head :math:`B_t` (Fig. 3).
        amplitude_raw: ``(r, k)`` softplus amplitude :math:`A_{t,k}` before
            the gate.
        amplitude: ``(r, k)`` gated amplitude :math:`g_{t,k} A_{t,k}` (Fig. 4).
        frequency: ``(r, k)`` instantaneous angular frequency
            :math:`\omega_{t,k}`, radians per unit of ``time`` (Fig. 5).
        phase: ``(r, k)`` cumulative phase :math:`\theta_{t,k}`.
        phase_offset: ``(r, k)`` phase offset :math:`\phi_{t,k}`.
        gate: ``(r, k)`` gate :math:`g_{t,k} \in (0, 1)` (Fig. 6); ones when
            gating is off.
        state: ``(r, state_dim)`` latent state :math:`h_t`.
        mix: ``(k,)`` learned mixing weights :math:`w_k`.
        history: Training losses by stage and step, each a record with the
            total and its unweighted parts.
        predictor: The trained network behind a numpy interface.

    Note:
        Under the filtered alignment the emission at :math:`t` has seen
        :math:`y_t`, so ``emission`` is a reconstruction and its error is
        not a forecast error; ``fittedvalues`` are the published one-step
        protocol and should be read against a random walk before being
        read as skill. ``llf`` is a concentrated Gaussian likelihood at a
        penalized fit and ``n_params`` the nominal weight count, which
        overstates the effective freedom of a weight-decayed network, so
        criteria compare :math:`K` within one configuration only. Without
        ``ordered_frequency`` the components are identified only up to
        permutation: a refit with another seed may return the same
        decomposition with its columns reordered.

    Warning:
        Amplitude-type paths (``amplitude_raw``, ``amplitude``,
        ``components``) are returned in the units of ``endog`` by scaling
        the training-unit paths by the slope of ``scale`` alone; the
        trend and emission also receive the offset. Phase, frequency, gate
        and state carry no units and are the same on both scales.

    See Also:
        * :class:`NSSDN` -- the specification that produces this.
        * :class:`~ns_sdn.spectral_state.SpectralState` -- the measure-valued
          reading of the decomposition.
        * :class:`~ns_sdn.summary.SummaryTable` -- what :meth:`summary` returns.

    Example:
        >>> import numpy as np
        >>> y = np.sin(0.2 * np.arange(150.0)) + 0.01 * np.arange(150.0)
        >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 25),), seed=0).fit()
        >>> res.alignment, res.observed.shape, res.resid.shape
        ('filtered', (150,), (149,))
        >>> bool(np.allclose(res.resid, res.endog[1:] - res.fittedvalues))
        True
        >>> bool(np.all((res.gate >= 0.0) & (res.gate <= 1.0)))
        True
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    time: npt.NDArray[np.float64] = field(repr=False)
    exog: npt.NDArray[np.float64] | None = field(repr=False)
    k: int
    state_dim: int
    alignment: str
    transition: str
    ordered_frequency: bool
    gating: bool
    objective: str
    scaling: str
    scale: _MinMaxScale = field(repr=False)
    emission: npt.NDArray[np.float64] = field(repr=False)
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    resid: npt.NDArray[np.float64] = field(repr=False)
    llf: float
    nobs: int
    n_params: int
    sigma2: float
    trend: npt.NDArray[np.float64] = field(repr=False)
    amplitude_raw: npt.NDArray[np.float64] = field(repr=False)
    amplitude: npt.NDArray[np.float64] = field(repr=False)
    frequency: npt.NDArray[np.float64] = field(repr=False)
    phase: npt.NDArray[np.float64] = field(repr=False)
    phase_offset: npt.NDArray[np.float64] = field(repr=False)
    gate: npt.NDArray[np.float64] = field(repr=False)
    state: npt.NDArray[np.float64] = field(repr=False)
    mix: npt.NDArray[np.float64] = field(repr=False)
    history: tuple[_StageRecord, ...] = field(repr=False)
    predictor: _SpectralPredictor = field(repr=False)

    # -- assembly -------------------------------------------------------------

    @classmethod
    def _from_fit(
        cls, fit: _SpectralFit, model: _SpectralDecompositionModel[NSSDNResult]
    ) -> NSSDNResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The record ``_fit_family`` packed: paths, one-step fit,
                likelihood, history and predictor.
            model: The specification the fit came from; supplies the data
                and every option the result reports.

        Returns:
            The frozen result.
        """
        paths = fit.paths
        return cls(
            endog=model.endog,
            time=model.time,
            exog=model.exog,
            k=model.k,
            state_dim=model.state_dim,
            alignment=model.alignment,
            transition=model.transition,
            ordered_frequency=model.ordered_frequency,
            gating=model.gating,
            objective=model.objective,
            scaling=model.scaling,
            scale=fit.predictor.scale,
            emission=paths.emission,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            sigma2=fit.sigma2,
            trend=paths.trend,
            amplitude_raw=paths.amplitude_raw,
            amplitude=paths.amplitude,
            frequency=paths.frequency,
            phase=paths.phase,
            phase_offset=paths.offset,
            gate=paths.gate,
            state=paths.state,
            mix=fit.predictor.mix,
            history=fit.history,
            predictor=fit.predictor,
        )

    # -- the decomposition ----------------------------------------------------

    @property
    def observed(self) -> npt.NDArray[np.float64]:
        """``(r,)`` the slice of ``endog`` the path rows align with.

        The whole series under the filtered alignment, ``endog[1:]`` under
        the predicted one, so that ``observed[i]`` is the observation row
        ``i`` of every path refers to.

        Example:
            >>> import numpy as np
            >>> y = np.sin(0.2 * np.arange(100.0))
            >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 10),), seed=0).fit()
            >>> res.observed.shape == res.emission.shape
            True
        """
        return self.endog[self.endog.shape[0] - self.emission.shape[0] :]

    @property
    def effective_amplitude(self) -> npt.NDArray[np.float64]:
        r"""``(r, k)`` the amplitude of each contribution to the emission.

        The gated amplitude scaled by the magnitude of the mixing weight,

        .. math::

           \tilde A_{t,k} = |w_k|\, g_{t,k}\, A_{t,k},

        which is the quantity whose square is the power the spectral-state
        coordinates place at :math:`\omega_{t,k}`. The sign of :math:`w_k`
        is a phase flip and is left to ``components``.
        """
        return np.abs(self.mix) * self.amplitude

    @property
    def components(self) -> npt.NDArray[np.float64]:
        r"""``(r, k)`` the contribution of each component to the emission.

        .. math::

           c_{t,k} = w_k\, g_{t,k}\, A_{t,k}\, \sin\!\bigl(\theta_{t,k} + \phi_{t,k}\bigr),

        so that :math:`\hat y_t = B_t + \sum_k c_{t,k}`: summing over
        :math:`k` and adding ``trend`` recovers ``emission`` to floating
        point precision.

        Example:
            >>> import numpy as np
            >>> y = np.sin(0.2 * np.arange(120.0))
            >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 20),), seed=0).fit()
            >>> rebuilt = res.trend + res.components.sum(axis=1)
            >>> bool(np.allclose(rebuilt, res.emission, atol=1e-5))
            True
        """
        return self.mix * self.amplitude * np.sin(self.phase + self.phase_offset)

    @property
    def oscillation(self) -> npt.NDArray[np.float64]:
        r"""``(r,)`` the oscillatory part of the emission, :math:`\hat y_t - B_t`.

        Equal to ``components.sum(axis=1)``. A fit whose oscillation has
        negligible variance has put the series into the trend head, which
        short schedules do; the ratio of its variance to that of
        ``observed`` is the quickest check that the components carry
        anything.
        """
        return self.emission - self.trend

    @property
    def period(self) -> npt.NDArray[np.float64]:
        r"""``(r, k)`` instantaneous period :math:`2\pi / \omega_{t,k}` in time units.

        Frequencies are floored at :math:`10^{-12}` before the division so
        a vanishing frequency gives a large finite period rather than an
        infinity.
        """
        return 2.0 * np.pi / np.maximum(self.frequency, 1e-12)

    @property
    def power_share(self) -> npt.NDArray[np.float64]:
        r"""``(k,)`` each component's share of mean contribution power over the sample.

        .. math::

           \pi_k = \frac{\overline{\tilde A_{\cdot,k}^2}}
                        {\sum_{j=1}^{K} \overline{\tilde A_{\cdot,j}^2}},
           \qquad
           \overline{\tilde A_{\cdot,k}^2} = \frac{1}{r}\sum_{t} \tilde A_{t,k}^2,

        with :math:`\tilde A` the ``effective_amplitude``. The shares sum to
        one and are the ``power share`` column of the summary table.

        Example:
            >>> import numpy as np
            >>> y = np.sin(0.2 * np.arange(120.0))
            >>> res = NSSDN(y, k=3, state_dim=8, schedule=(Stage(1e-2, 20),), seed=0).fit()
            >>> res.power_share.shape, bool(np.isclose(res.power_share.sum(), 1.0))
            ((3,), True)
        """
        power = (self.effective_amplitude**2).mean(axis=0)
        return power / max(float(power.sum()), 1e-300)

    @property
    def reconstruction_resid(self) -> npt.NDArray[np.float64]:
        r"""``(r,)`` the same-row error ``observed - emission``.

        Under the filtered alignment this is :math:`y_t - \hat y_{t \mid t}`,
        the error of a reconstruction that has seen :math:`y_t`, and its
        root mean square is the summary's ``Emission RMSE``. Under the
        predicted alignment the rows are one-step predictions and this
        coincides with ``resid``.
        """
        return self.observed - self.emission

    @property
    def r_squared(self) -> float:
        r"""One minus the one-step residual share of variance over the effective sample.

        .. math::

           R^2 = 1 - \frac{\sum_{t=1}^{n-1} e_t^2}
                          {\sum_{t=1}^{n-1} (y_t - \bar y)^2},

        with :math:`\bar y` the mean of ``endog[1:]``. Zero when the
        target has no variance. A persistent series gives a random walk an
        :math:`R^2` near one as well, so read this next to a naive
        benchmark rather than on its own.
        """
        target = self.endog[1:]
        total = float(np.sum((target - target.mean()) ** 2))
        return 1.0 - float(self.resid @ self.resid) / total if total > 0.0 else 0.0

    def _paths_to_dict(
        self, observed: npt.NDArray[np.float64], paths: _SpectralPaths, mix: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Lay a pass of the network out as the dictionary ``decompose`` returns.

        Args:
            observed: ``(r,)`` the observations the rows align with, on the
                same scale as ``paths``.
            paths: One pass of the predictor.
            mix: ``(k,)`` the mixing vector, which the predictor's paths do
                not carry.

        Returns:
            The arrays keyed as :meth:`decompose` documents.
        """
        components = mix * paths.amplitude * np.sin(paths.phase + paths.offset)
        return {
            "observed": observed,
            "emission": paths.emission,
            "trend": paths.trend,
            "oscillation": paths.emission - paths.trend,
            "amplitude_raw": paths.amplitude_raw,
            "amplitude": paths.amplitude,
            "effective_amplitude": np.abs(mix) * paths.amplitude,
            "frequency": paths.frequency,
            "phase": paths.phase,
            "phase_offset": paths.offset,
            "gate": paths.gate,
            "components": components,
            "state": paths.state,
        }

    def decompose(
        self,
        endog: npt.ArrayLike | None = None,
        *,
        exog: npt.ArrayLike | None = None,
        time: npt.ArrayLike | None = None,
        scaled: bool = False,
    ) -> dict[str, npt.NDArray[np.float64]]:
        r"""Every path of the decomposition, for the fitted sample or a new one.

        With no arguments this is the fitted sample's decomposition, the
        same arrays the result carries as attributes, laid out as one
        dictionary. With ``endog`` the trained network is run over a new
        series under teacher forcing from its learned initial state
        :math:`h_0`, which is how the paper draws the full-sample figures:
        training and holdout concatenated, the network never refitted.

        Args:
            endog: A series to run the trained network over under teacher
                forcing; ``None`` returns the fitted sample's paths. At
                least two observations.
            exog: ``(len(endog), p)`` covariates for ``endog``; required
                when the fit used covariates and refused otherwise.
            time: ``(len(endog),)`` strictly increasing grid for ``endog``;
                the integers when ``None``.
            scaled: Return the paths in training units (the scale the
                paper's figures are drawn in, :math:`[-1, 1]` under min-max
                scaling) rather than on the original scale. ``observed`` is
                transformed to match.

        Returns:
            Arrays keyed ``observed, emission, trend, oscillation,
            amplitude_raw, amplitude, effective_amplitude, frequency, phase,
            phase_offset, gate, components, state``, each with :math:`r`
            rows (the length of the series under the filtered alignment,
            one fewer under the predicted one) and the column counts of the
            corresponding attributes.

        Raises:
            DimensionError: If ``endog`` is not one-dimensional or too
                short, or if ``exog`` or ``time`` is misaligned with it, or
                if the covariate width differs from the fit's.
            NumericalError: If any input contains non-finite values.
            SpecificationError: If ``time`` is not strictly increasing.

        Example:
            The paper's trend figure is on the training scale:

            >>> import numpy as np
            >>> y = np.sin(0.2 * np.arange(120.0)) + 5.0
            >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 20),), seed=0).fit()
            >>> paths = res.decompose(scaled=True)
            >>> bool(paths["trend"].max() <= 1.0 + 1e-6), paths["trend"].shape
            (True, (120,))

            Running over a longer sample does not refit; the first rows
            reproduce the fitted sample's paths:

            >>> longer = np.concatenate([y, np.sin(0.2 * np.arange(120.0, 150.0)) + 5.0])
            >>> out = res.decompose(longer)
            >>> out["emission"].shape, bool(np.allclose(out["emission"][:120], res.emission))
            ((150,), True)
        """
        if endog is None:
            y = self.endog
            paths = self.predictor.emissions(y, self.exog, self.time, scaled=scaled)
        else:
            y = _validate_endog(endog, min_len=2)
            paths = self.predictor.emissions(
                y,
                _validate_exog(exog, y.shape[0]),
                _validate_time(time, y.shape[0]),
                scaled=scaled,
            )
        observed = self.scale.transform(y) if scaled else y
        return self._paths_to_dict(
            observed[y.shape[0] - paths.emission.shape[0] :], paths, self.mix
        )

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """Observed, emission, trend, and every per-component path as columns.

        The layout behind ``to_dict`` and ``to_pandas``: the scalar paths
        first, then for each component :math:`j` (one-based in the column
        names) its amplitude, frequency, phase, phase offset, gate and
        contribution.

        Returns:
            Arrays of :math:`r` rows keyed ``observed, emission, trend,
            oscillation`` and ``amplitude_j, frequency_j, phase_j,
            phase_offset_j, gate_j, component_j`` for ``j = 1 .. k``.
        """
        out: dict[str, npt.NDArray[np.float64]] = {
            "observed": self.observed,
            "emission": self.emission,
            "trend": self.trend,
            "oscillation": self.oscillation,
        }
        components = self.components
        for j in range(self.k):
            out[f"amplitude_{j + 1}"] = self.amplitude[:, j]
            out[f"frequency_{j + 1}"] = self.frequency[:, j]
            out[f"phase_{j + 1}"] = self.phase[:, j]
            out[f"phase_offset_{j + 1}"] = self.phase_offset[:, j]
            out[f"gate_{j + 1}"] = self.gate[:, j]
            out[f"component_{j + 1}"] = components[:, j]
        return out

    # -- prediction -----------------------------------------------------------

    def predict(
        self,
        endog: npt.ArrayLike,
        *,
        exog: npt.ArrayLike | None = None,
        time: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""One-step conditional predictions over a sample.

        Runs the trained network from its learned initial state under
        teacher forcing and returns, for each :math:`y_{i+1}`, the emission
        built from the realized values the alignment allows: through
        :math:`y_i` under both alignments, which under the filtered one is
        the emission at :math:`i`,

        .. math::

           \hat y_{i+1 \mid i} =
           \begin{cases}
             \hat y_i, & \text{filtered: the emission after } y_i,\\
             \hat y_{i+1}, & \text{predicted: the row that targets } y_{i+1}.
           \end{cases}

        The published holdout protocol is
        ``res.predict(np.concatenate([train, holdout]))[-len(holdout):]``,
        the paper's ``yhat[split - 1 : -1]`` against ``y[split:]``: the
        network conditions on the realized holdout values one step at a
        time, which is a conditional one-step forecast and not a recursive
        one.

        Args:
            endog: The series, at least two observations, on the original
                scale.
            exog: ``(len(endog), p)`` covariates aligned with ``endog``;
                required when the fit used them.
            time: ``(len(endog),)`` strictly increasing grid; the integers
                when ``None``.

        Returns:
            ``(len(endog) - 1,)`` predictions on the original scale, aligned
            with ``endog[1:]``.

        Raises:
            DimensionError: If ``endog`` is mis-shaped or too short, or if
                ``exog`` or ``time`` is misaligned or the covariate width
                differs from the fit's.
            NumericalError: If any input contains non-finite values.
            SpecificationError: If ``time`` is not strictly increasing.

        Example:
            On the fitted sample the predictions are the result's own
            ``fittedvalues``:

            >>> import numpy as np
            >>> y = np.sin(0.2 * np.arange(120.0))
            >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 20),), seed=0).fit()
            >>> bool(np.allclose(res.predict(y), res.fittedvalues))
            True
        """
        y = _validate_endog(endog, min_len=2)
        paths = self.predictor.emissions(
            y, _validate_exog(exog, y.shape[0]), _validate_time(time, y.shape[0])
        )
        return paths.emission[:-1] if self.alignment == "filtered" else paths.emission

    def forecast(
        self,
        steps: int = 1,
        *,
        exog: npt.ArrayLike | None = None,
        time: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Recursive multi-step forecast from the end of the fitted sample.

        The network is warmed up on the sample under teacher forcing, the
        last observation :math:`y_{n-1}` is fed as the first future input,
        and every later input is the previous emission,

        .. math::

           h_{n+s} = F_\psi\bigl(h_{n+s-1}, \hat y_{n+s-1}, \Delta t_{n+s}\bigr),
           \qquad
           \theta_{n+s,k} = \theta_{n+s-1,k} + \omega_{n+s,k}\,\Delta t_{n+s},
           \qquad s = 1, \ldots, S,

        with :math:`\hat y_{n} \equiv y_{n-1}` for the first step. Error
        accumulates with the horizon; the validation notebook found this
        mode far weaker than the conditional one-step mode under the
        filtered alignment, and the paper reports both for that reason.

        Args:
            steps: Periods ahead :math:`S`, at least one.
            exog: ``(steps, p)`` future covariates; required when the fit
                used them. A one-dimensional array is reshaped to
                ``(steps, -1)``.
            time: ``(steps,)`` future grid after ``time[-1]``; defaults to
                continuing at the sample's last step size, so a regular
                grid needs nothing.

        Returns:
            ``(steps,)`` forecasts on the original scale.

        Raises:
            SpecificationError: If ``steps`` is not a positive integer, or
                ``time`` does not have ``steps`` entries.
            DimensionError: If the future covariates are missing or
                mis-shaped for a fit that used them.

        Example:
            >>> import numpy as np
            >>> y = np.sin(0.2 * np.arange(120.0))
            >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 20),), seed=0).fit()
            >>> res.forecast(3).shape, res.forecast(1).shape
            ((3,), (1,))
            >>> res.forecast(0)
            Traceback (most recent call last):
                ...
            ns_sdn.exceptions.SpecificationError: steps must be a positive integer; got 0.
        """
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise SpecificationError(f"steps must be a positive integer; got {steps!r}.")
        future_exog = (
            None if exog is None else np.asarray(exog, dtype=np.float64).reshape(steps, -1)
        )
        future_time = None if time is None else np.asarray(time, dtype=np.float64).reshape(-1)
        if future_time is not None and future_time.shape[0] != steps:
            raise SpecificationError(f"time must have {steps} entries; got {future_time.shape[0]}.")
        paths = self.predictor.recursive(
            self.endog,
            self.exog,
            self.time,
            steps=steps,
            future_exog=future_exog,
            future_time=future_time,
        )
        return paths.emission

    def spectral_state(self, *, active_threshold: float | None = None) -> SpectralState:
        r"""The decomposition read as a time-local spectral measure.

        Normalizes the contribution powers :math:`\tilde A_{t,k}^2` into
        weights on the frequency axis and computes the centroid,
        dispersion, entropy, effective count, Wasserstein velocity and
        disentanglement score of the resulting measure at every row; see
        :mod:`~ns_sdn.spectral_state` for the definitions.

        Args:
            active_threshold: Power-weight floor in :math:`[0, 1]` below
                which a component is inactive for the diagnostics; the
                package default (:math:`10^{-2}`) when ``None``.

        Returns:
            A :class:`~ns_sdn.spectral_state.SpectralState` with one row
            per path row.

        Raises:
            SpecificationError: If the threshold lies outside :math:`[0, 1]`.

        Example:
            >>> import numpy as np
            >>> y = np.sin(0.2 * np.arange(120.0))
            >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 20),), seed=0).fit()
            >>> state = res.spectral_state()
            >>> state.nobs, state.k, state.centroid.shape
            (120, 2, (120,))
        """
        from .spectral_state import SpectralState

        return SpectralState.from_result(self, active_threshold=active_threshold)

    # -- rendering ------------------------------------------------------------

    def _comparison_label(self) -> str:
        """The row label ``compare`` prints for this result.

        Returns:
            ``NSSDN(k=<k>, d=<state_dim>, <transition>, <alignment>)``.
        """
        return f"NSSDN(k={self.k}, d={self.state_dim}, {self.transition}, {self.alignment})"

    def _summary_table(self) -> SummaryTable:
        r"""Structured summary rendered by every display path.

        The metadata block carries the fit statistics (likelihood,
        criteria, parameter count, the one-step RMSE
        :math:`\hat\sigma` and the same-row emission RMSE); the table has
        one row per component with its mean gated amplitude, mean
        frequency, the period of that mean frequency, mean gate, mixing
        weight and power share; the notes state the alignment and what it
        makes of the numbers, the status of the likelihood, and, when
        frequencies are unordered, the permutation caveat.

        Returns:
            The :class:`~ns_sdn.summary.SummaryTable`.
        """
        criteria = self.information_criteria
        share = self.power_share
        rows = tuple(
            (
                f"component {j + 1}",
                f"{self.amplitude[:, j].mean():.4f}",
                f"{self.frequency[:, j].mean():.4f}",
                f"{2.0 * np.pi / max(float(self.frequency[:, j].mean()), 1e-12):.1f}",
                f"{self.gate[:, j].mean():.3f}",
                f"{self.mix[j]:+.3f}",
                f"{share[j]:.3f}",
            )
            for j in range(self.k)
        )
        reconstruction = self.reconstruction_resid
        notes = []
        if self.alignment == "filtered":
            notes.append(
                "Filtered alignment: the emission at t conditions on y[t], so emission is a "
                "reconstruction of the observed series and fittedvalues report the emission at t "
                "as the one-step forecast of y[t+1] (the published protocol); compare against a "
                "random walk before reading them as skill."
            )
        else:
            notes.append(
                "Predicted alignment: the emission at t conditions on y[t-1]; every path is a "
                "one-step-ahead prediction and fittedvalues are the emission rows."
            )
        notes.append(
            "llf is a concentrated Gaussian likelihood at a penalized fit and n_params the nominal "
            "weight count; criteria compare k within one configuration only."
        )
        if not self.ordered_frequency:
            notes.append(
                "Components are identified up to permutation; indices are not stable across refits."
            )
        return SummaryTable(
            title=f"NSSDN(k={self.k}) Results",
            metadata=(
                ("Model", self._comparison_label()),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Objective", self.objective),
                ("AIC", f"{criteria.aic:.3f}"),
                ("Scaling", self.scaling),
                ("BIC", f"{criteria.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{criteria.hqic:.3f}"),
                ("Parameters", f"{self.n_params}"),
                ("One-step RMSE", f"{np.sqrt(self.sigma2):.6g}"),
                ("Gating", str(self.gating)),
                ("Emission RMSE", f"{np.sqrt(np.mean(reconstruction**2)):.6g}"),
            ),
            columns=("", "mean amp", "mean freq", "period", "mean gate", "mix", "power share"),
            rows=rows,
            notes=tuple(notes),
        )


class NSSDN(_SpectralDecompositionModel[NSSDNResult]):
    r"""Non-Stationary Spectral Decomposition Network specification.

    The immutable statement of the model, validated at construction and
    estimated by :meth:`fit`. The defaults are the configuration of the
    IEEE-ORSS DGS10 experiments: sixteen components, a 128-dimensional GRU
    state, positive frequencies, gating, the filtered alignment, mean
    squared error on a min-max scaled series, and the two-stage Adam
    schedule with amplitude and frequency drift penalties,

    .. math::

       \mathcal{L} = \frac{1}{r}\sum_t \bigl(\hat y_t - y_t\bigr)^2
       + \lambda_A\, \frac{1}{(r-1)K}\sum_{t,k}
         \bigl(g_{t,k}A_{t,k} - g_{t-1,k}A_{t-1,k}\bigr)^2
       + \lambda_\omega\, \frac{1}{(r-1)K}\sum_{t,k}
         \bigl(\omega_{t,k} - \omega_{t-1,k}\bigr)^2,

    evaluated on the scaled series, with
    :math:`(\eta, \text{steps}, \lambda_A, \lambda_\omega)` equal to
    :math:`(10^{-3}, 1200, 2 \times 10^{-4}, 8 \times 10^{-4})` in stage A
    and :math:`(4 \times 10^{-4}, 1800, 3 \times 10^{-4}, 1.5 \times 10^{-3})`
    in stage B. With no covariates that network has 58,849 trainable
    parameters, the count of the published run.

    Args:
        endog: The series, one-dimensional, at least three observations,
            finite throughout; a ``(n, 1)`` column is accepted.
        exog: Optional ``(n, p)`` covariates known at each :math:`t`, used
            as given (no scaling) as extra columns of the transition input
            :math:`x_t`. A fit with covariates needs them again at
            forecast time.
        time: Optional strictly increasing grid; defaults to
            :math:`0, 1, \ldots, n - 1`. Irregular spacing enters the phase
            recursion through :math:`\Delta t_t` and, under the time-aware
            transition, the dynamics.
        k: Number of sinusoidal components :math:`K`, at least one.
        state_dim: Latent dimension :math:`d`, at least one.
        alignment: ``"filtered"`` (the published run: the emission at
            :math:`t` conditions on :math:`y_t`) or ``"predicted"`` (it
            conditions on :math:`y_{t-1}`, so the network trains as a
            one-step forecaster).
        transition: ``"gru"`` (a gated recurrent cell on :math:`x_t`) or
            ``"timeaware"`` (the residual Euler update
            :math:`h_t = h_{t-1} + \Delta t_t\, g_\psi(h_{t-1}, x_t)`).
        positive_frequency: Softplus the frequency head so phase always
            advances; off, the head is linear and a component may run
            backwards.
        ordered_frequency: Parameterize frequencies as cumulative positive
            gaps inside the Nyquist band
            :math:`[\omega_{\min}, \pi / \operatorname{median}\Delta t]`, so
            :math:`\omega_{t,1} < \cdots < \omega_{t,K}` at every :math:`t`
            and the components are identified without permutation. Implies
            positivity. Off by default to reproduce the published run.
        gating: Add the sigmoid gate head :math:`g_{t,k}`; off, every gate
            is one and ``amplitude`` equals ``amplitude_raw``.
        objective: ``"mse"`` or ``"huber"`` for the prediction term.
        delta: Huber threshold on the scaled residual, strictly positive;
            unused under ``"mse"``.
        penalties: :class:`Penalties` for stages that carry none; the
            stage-A weights when ``None``.
        schedule: Tuple of :class:`Stage`; the published two stages when
            ``None``.
        scaling: ``"minmax"`` (train on the series mapped to
            :math:`[-1, 1]`, report on the original scale) or ``"none"``.
        seed: Initialization seed; ``None`` leaves the fit unreproducible.
            Seeding is scoped to construction and leaves torch's global
            generator untouched.
        device: Torch device string, ``"cpu"`` by default.

    Raises:
        SpecificationError: If any option is outside its admissible set:
            an unknown alignment, transition, objective or scaling, a
            non-positive ``k``, ``state_dim`` or ``delta``, a malformed
            schedule or penalties record, a non-integer seed, or a time
            grid that is not strictly increasing.
        DimensionError: If the data are mis-shaped or misaligned: a
            multi-dimensional series, fewer than three observations, or
            covariates or a grid whose length differs from the series.
        NumericalError: If the data contain non-finite values, or (at
            :meth:`fit`) training diverges.

    Note:
        Training is one full-sequence gradient per step through the whole
        sample, so cost scales with :math:`n` times the schedule length;
        the published run is 3,000 steps over 9,186 daily observations.
        Under the GRU transition the teacher-forced pass is a single
        recurrent call rather than a Python loop. The parameter count of a
        GRU network with input width :math:`u = 1 + p` is

        .. math::

           3\bigl(d(u + d) + 2d\bigr) + (d + 1)(1 + 4K) + 2K + d,

        which is 58,849 at :math:`d = 128`, :math:`K = 16`, :math:`u = 1`;
        the ordered-frequency head adds :math:`d + 1` and removing the gate
        removes :math:`K(d + 1)`.

    See Also:
        * :class:`NSSDNResult` -- what :meth:`fit` returns.
        * :class:`Stage`, :class:`Penalties` -- the schedule records.
        * :mod:`~ns_sdn.typing` -- the admissible values of the categorical
          options.

    Example:
        >>> import numpy as np
        >>> y = np.sin(0.3 * np.arange(200.0)) + 0.01 * np.arange(200.0)
        >>> model = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 30),), seed=0)
        >>> model.k, model.alignment, model.transition, len(model.schedule)
        (2, 'filtered', 'gru', 1)
        >>> NSSDN(y, transition="lstm")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ns_sdn.exceptions.SpecificationError: transition must be one of 'gru', 'timeaware'; ...

        The published configuration is the default:

        >>> published = NSSDN(y)
        >>> published.k, published.state_dim, len(published.schedule), published.gating
        (16, 128, 2, True)
    """

    __slots__ = ()

    def fit(self) -> NSSDNResult:
        r"""Train the network on the sample and return the fitted result.

        In order: the series is scaled (min-max to :math:`[-1, 1]` or left
        as is), the network is built and seeded, the teacher-forced inputs
        are laid out under the alignment, every stage of the schedule runs
        a fresh Adam optimizer for its steps on the staged criterion, the
        trained network is run once more over the sample to produce the
        paths, and the one-step residuals, their variance and the
        concentrated likelihood are formed from the emission rows the
        alignment designates as predictions.

        Returns:
            The :class:`NSSDNResult`.

        Raises:
            NumericalError: If the loss becomes non-finite at any step;
                the message names the stage and step. Lower the learning
                rate or scale the series.

        Example:
            >>> import numpy as np
            >>> y = np.sin(0.3 * np.arange(200.0))
            >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 30),), seed=0).fit()
            >>> res.nobs, res.trend.shape, res.state.shape, res.fittedvalues.shape
            (199, (200,), (200, 8), (199,))

            The same seed gives the same fit:

            >>> again = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 30),), seed=0).fit()
            >>> bool(np.allclose(res.emission, again.emission))
            True
        """
        return NSSDNResult._from_fit(self._fit_family(), self)
