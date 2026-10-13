# Copyright (c) 2026 Nikhil Sunder
# SPDX-License-Identifier: MIT
r"""The Non-Stationary Spectral Decomposition Network: specification and result.

The model is a neural state-space representation of a scalar series as a
trend plus ``K`` sinusoids whose amplitude, frequency, phase offset and
participation evolve with a latent state (IEEE-ORSS equations 1-5),

.. math::

   \hat y_t = B_t + \sum_{k=1}^{K} w_k\, g_{t,k}\, A_{t,k}\,
   \sin(\theta_{t,k} + \phi_{t,k}),
   \qquad
   \theta_{t,k} = \theta_{t-1,k} + \omega_{t,k}\,\Delta t_t,
   \qquad
   h_t = F_\psi(h_{t-1}, x_t, \Delta t_t),

with every spectral quantity read off ``h_t`` by a linear head: ``B_t``
unconstrained, ``A_{t,k}`` by softplus, ``\omega_{t,k}`` by softplus (or,
under ``ordered_frequency``, as cumulative positive gaps inside the
Nyquist band), ``\phi_{t,k}`` unconstrained, ``g_{t,k}`` by a sigmoid,
and ``w`` the learned mixing vector of the paper's implementation note.

Held under the three-object discipline: :class:`NSSDN` is the immutable,
validated specification; training belongs to the family base it
inherits; :class:`NSSDNResult` is the frozen record the user handles.
The decomposition is the point of the model, so the result carries every
path the figures of the paper are drawn from -- ``trend`` (Fig. 3),
``amplitude`` (Fig. 4, the gated ``g A``), ``frequency`` (Fig. 5),
``gate`` (Fig. 6), ``phase``, ``phase_offset``, ``state``, and the
per-component ``components`` whose sum with ``trend`` is the ``emission``
(Fig. 1) -- on the original scale by default and in training units
through ``decompose(scaled=True)``, with one-step ``fittedvalues`` (Fig.
2) aligned with ``endog[1:]`` whatever the alignment.

Two alignments are available. ``"filtered"``, the default and the
published run, conditions the emission at ``t`` on ``y_t``: the
decomposition is a filtered reading of the observed series, the emission
reconstructs it, and the one-step forecast of ``y_{t+1}`` is the emission
at ``t`` (the paper's holdout protocol). ``"predicted"`` conditions the
emission at ``t`` on ``y_{t-1}``, so every path is a one-step-ahead
prediction and the network trains as a forecaster. The summary says which
one a result is and what that means for its numbers.

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
    *IEEE ORSS*.

    Sunder, N. (2026). Non-Stationary Spectral Decomposition Network for
    Econometric Time Series Forecasting. *FLAIRS-39*.

    Huang, N. E. et al. (1998). The empirical mode decomposition and the
    Hilbert spectrum for nonlinear and non-stationary time series
    analysis. *Proc. R. Soc. A*, 454, 903-995.

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

    Every path has ``r`` rows: ``r = n`` under the filtered alignment, where
    row ``t`` is the emission that reconstructs ``endog[t]`` from the
    state after ``endog[t]``; ``r = n - 1`` under the predicted alignment,
    where row ``i`` is the emission that predicts ``endog[i + 1]`` from the
    state after ``endog[i]``. ``observed`` is the slice of ``endog`` the
    rows align with. ``fittedvalues`` always has ``n - 1`` rows aligned with
    ``endog[1:]`` and is the one-step conditional prediction of each
    observation: the emission rows under the predicted alignment, the
    emissions through ``n - 2`` under the filtered one (the paper's
    ``yhat[split - 1 : -1]`` against ``y[split:]``).

    Attributes:
        endog: The full observed series.
        time: The time grid.
        exog: The covariates used, or ``None``.
        k: Number of components.
        state_dim: Latent dimension.
        alignment: ``"filtered"`` or ``"predicted"``.
        transition: Transition kind.
        ordered_frequency: Whether frequencies are strictly increasing in ``k``.
        gating: Whether a gate head was fitted.
        objective: Prediction loss.
        scaling: Scaling choice.
        scale: The affine map between training units and the original scale.
        emission: ``(r,)`` the synthesized series, ``trend + components.sum(1)``.
        fittedvalues: ``(n - 1,)`` one-step conditional predictions of ``endog[1:]``.
        resid: ``endog[1:] - fittedvalues``.
        llf: Concentrated Gaussian log-likelihood of ``resid``.
        nobs: Effective sample size, ``n - 1``.
        n_params: Trainable parameter count.
        sigma2: Residual variance.
        trend: ``(r,)`` trend head ``B_t`` (Fig. 3).
        amplitude_raw: ``(r, k)`` softplus amplitude ``A_{t,k}`` before the gate.
        amplitude: ``(r, k)`` gated amplitude ``g_{t,k} A_{t,k}`` (Fig. 4).
        frequency: ``(r, k)`` instantaneous angular frequency ``omega_{t,k}``,
            radians per unit time (Fig. 5).
        phase: ``(r, k)`` cumulative phase ``theta_{t,k}``.
        phase_offset: ``(r, k)`` phase offset ``phi_{t,k}``.
        gate: ``(r, k)`` gate ``g_{t,k}`` in ``(0, 1)`` (Fig. 6); ones when gating is off.
        state: ``(r, state_dim)`` latent state ``h_t``.
        mix: ``(k,)`` learned mixing weights ``w_k``.
        history: Training losses by stage and step.
        predictor: The trained network behind a numpy interface.

    Note:
        Under the filtered alignment the emission at ``t`` has seen
        ``y_t``, so ``emission`` is a reconstruction and its error is not a
        forecast error; ``fittedvalues`` are the published one-step
        protocol and should be read against a random walk before being
        read as skill. ``llf`` is a concentrated Gaussian likelihood at a
        penalized fit and ``n_params`` the nominal weight count, which
        overstates the effective freedom of a weight-decayed network, so
        criteria compare ``k`` within one configuration only. Without
        ``ordered_frequency`` the components are identified only up to
        permutation.

    See Also:
        * :class:`NSSDN` -- the specification that produces this.
        * :class:`~ns_sdn.spectral_state.SpectralState` -- the measure-valued
          reading of the decomposition.
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
        """Assemble the public result from a raw fit and its specification."""
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
        """``(r,)`` the slice of ``endog`` the path rows align with."""
        return self.endog[self.endog.shape[0] - self.emission.shape[0] :]

    @property
    def effective_amplitude(self) -> npt.NDArray[np.float64]:
        """``(r, k)`` the amplitude of each contribution, ``|w_k| g_{t,k} A_{t,k}``."""
        return np.abs(self.mix) * self.amplitude

    @property
    def components(self) -> npt.NDArray[np.float64]:
        """``(r, k)`` contributions ``w_k g_{t,k} A_{t,k} sin(theta_{t,k} + phi_{t,k})``.

        Summing over ``k`` and adding ``trend`` recovers ``emission`` exactly.

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
        """``(r,)`` the oscillatory part of the emission, ``emission - trend``."""
        return self.emission - self.trend

    @property
    def period(self) -> npt.NDArray[np.float64]:
        """``(r, k)`` instantaneous period ``2 pi / omega`` in time units."""
        return 2.0 * np.pi / np.maximum(self.frequency, 1e-12)

    @property
    def power_share(self) -> npt.NDArray[np.float64]:
        """``(k,)`` each component's share of mean contribution power over the sample."""
        power = (self.effective_amplitude**2).mean(axis=0)
        return power / max(float(power.sum()), 1e-300)

    @property
    def reconstruction_resid(self) -> npt.NDArray[np.float64]:
        """``(r,)`` ``observed - emission``; a same-step error under the filtered alignment."""
        return self.observed - self.emission

    @property
    def r_squared(self) -> float:
        """One minus the one-step residual share of variance over the effective sample."""
        target = self.endog[1:]
        total = float(np.sum((target - target.mean()) ** 2))
        return 1.0 - float(self.resid @ self.resid) / total if total > 0.0 else 0.0

    def _paths_to_dict(
        self, observed: npt.NDArray[np.float64], paths: _SpectralPaths, mix: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
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
        """Every path of the decomposition, for the fitted sample or a new one.

        Args:
            endog: A series to run the trained network over under teacher
                forcing; ``None`` returns the fitted sample's paths.
            exog: Covariates for ``endog``; required when the fit used them.
            time: Grid for ``endog``.
            scaled: Return the paths in training units (the scale the paper's
                figures are drawn in) rather than on the original scale.

        Returns:
            Arrays keyed ``observed, emission, trend, oscillation,
            amplitude_raw, amplitude, effective_amplitude, frequency, phase,
            phase_offset, gate, components, state``, each with ``r`` rows.

        Example:
            The paper's trend figure is on the training scale:

            >>> import numpy as np
            >>> y = np.sin(0.2 * np.arange(120.0)) + 5.0
            >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 20),), seed=0).fit()
            >>> paths = res.decompose(scaled=True)
            >>> bool(paths["trend"].max() <= 1.0 + 1e-6), paths["trend"].shape
            (True, (120,))
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
        """Observed, emission, trend, and every per-component path as columns."""
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
        """One-step conditional predictions over a sample.

        Runs the trained network from its learned initial state under
        teacher forcing and returns, for each ``endog[i + 1]``, the emission
        built from the realized values the alignment allows: through
        ``endog[i]`` under both alignments, which under the filtered one is
        the emission at ``i``. The published holdout protocol is
        ``res.predict(np.concatenate([train, holdout]))[-len(holdout):]``.

        Args:
            endog: The series, at least two observations.
            exog: Covariates aligned with ``endog``; required when the fit used them.
            time: Grid for ``endog``; defaults to the integers.

        Returns:
            ``(len(endog) - 1,)`` predictions on the original scale, aligned
            with ``endog[1:]``.
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
        """Recursive multi-step forecast from the end of the fitted sample.

        The network is warmed up on the sample under teacher forcing, the
        last observation is fed as the first future input, and every later
        input is the previous emission. Error accumulates with the horizon;
        the validation notebook found this mode far weaker than the
        conditional one-step mode under the filtered alignment.

        Args:
            steps: Periods ahead, at least one.
            exog: ``(steps, p)`` future covariates; required when the fit used them.
            time: ``(steps,)`` future grid after ``time[-1]``; defaults to
                continuing at the sample's last step size.

        Returns:
            ``(steps,)`` forecasts on the original scale.

        Raises:
            SpecificationError: If ``steps`` is not a positive integer.
            DimensionError: If the future covariates are missing or mis-shaped.
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
        """The decomposition read as a time-local spectral measure.

        Args:
            active_threshold: Power-weight floor below which a component is
                inactive for the diagnostics; the package default when ``None``.

        Returns:
            A :class:`~ns_sdn.spectral_state.SpectralState`.
        """
        from .spectral_state import SpectralState

        return SpectralState.from_result(self, active_threshold=active_threshold)

    # -- rendering ------------------------------------------------------------

    def _comparison_label(self) -> str:
        return f"NSSDN(k={self.k}, d={self.state_dim}, {self.transition}, {self.alignment})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
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
    schedule with amplitude and frequency drift penalties. With no
    covariates that network has 58,849 trainable parameters, the count of
    the published run.

    Args:
        endog: The series, one-dimensional, at least three observations.
        exog: Optional ``(n, p)`` covariates known at each ``t``, used as
            given (no scaling). A fit with covariates needs them again at
            forecast time.
        time: Optional strictly increasing grid; defaults to ``0 .. n - 1``.
            Irregular spacing enters the phase recursion and, under the
            time-aware transition, the dynamics.
        k: Number of sinusoidal components.
        state_dim: Latent dimension ``d``.
        alignment: ``"filtered"`` (the published run: the emission at ``t``
            conditions on ``y_t``) or ``"predicted"`` (it conditions on
            ``y_{t-1}``, so the network trains as a one-step forecaster).
        transition: ``"gru"`` or ``"timeaware"``.
        positive_frequency: Softplus the frequency head so phase always advances.
        ordered_frequency: Parameterize frequencies as cumulative positive
            gaps inside the Nyquist band, so ``omega_{t,1} < ... < omega_{t,K}``
            at every ``t`` and the components are identified without
            permutation. Off by default to reproduce the published run.
        gating: Add the sigmoid gate head.
        objective: ``"mse"`` or ``"huber"``.
        delta: Huber threshold on the scaled residual.
        penalties: :class:`Penalties` for stages that carry none.
        schedule: Tuple of :class:`Stage`; the published two stages by default.
        scaling: ``"minmax"`` or ``"none"``.
        seed: Initialization seed; ``None`` leaves the fit unreproducible.
        device: Torch device string.

    Raises:
        SpecificationError: If any option is outside its admissible set.
        DimensionError: If the data are mis-shaped or misaligned.
        NumericalError: If the data contain non-finite values, or training diverges.

    Note:
        Training is one full-sequence gradient per step through the whole
        sample, so cost scales with ``n`` times the schedule length; the
        published run is 3,000 steps over 9,186 daily observations. Under
        the GRU transition the teacher-forced pass is a single recurrent
        call rather than a Python loop.

    See Also:
        * :class:`NSSDNResult` -- what :meth:`fit` returns.
        * :class:`Stage`, :class:`Penalties` -- the schedule records.

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
    """

    __slots__ = ()

    def fit(self) -> NSSDNResult:
        """Train the network on the sample and return the fitted result.

        Returns:
            The :class:`NSSDNResult`.

        Example:
            >>> import numpy as np
            >>> y = np.sin(0.3 * np.arange(200.0))
            >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 30),), seed=0).fit()
            >>> res.nobs, res.trend.shape, res.state.shape, res.fittedvalues.shape
            (199, (200,), (200, 8), (199,))
        """
        return NSSDNResult._from_fit(self._fit_family(), self)
