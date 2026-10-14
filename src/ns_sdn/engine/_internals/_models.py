# filepath: /src/ns_sdn/engine/_internals/_models.py
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
r"""Protected base classes for model specifications.

``_BaseModel`` owns ``endog`` validation and the abstract ``fit``.
``_SpectralDecompositionModel`` is the family base: it validates every
specification argument at construction, builds the network, runs the
schedule, and packs the raw outputs into a ``_SpectralFit``. The public
:class:`~ns_sdn.spectral_decomposition.NSSDN` class adds nothing but
``fit``, which assembles the public result from that record and the
specification.

Both bases are generic in the result type ``R`` so that a subclass's
``fit`` is typed with its own result, and both use ``__slots__`` so a
specification cannot grow attributes after validation.

Example:
    >>> import numpy as np
    >>> from ns_sdn.spectral_decomposition import NSSDN
    >>> model = NSSDN(np.arange(10.0), time=np.arange(10.0) * 2.0)
    >>> isinstance(model, _SpectralDecompositionModel), isinstance(model, _BaseModel)
    (True, True)
    >>> model.nobs_total, model.exog_dim, model.positive_frequency
    (10, 0, True)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from ...exceptions import SpecificationError
from .._core._containers import _MinMaxScale, _Penalties, _Stage
from .._core._defaults import (
    _FREQUENCY_MIN,
    _HISTORY_EVERY,
    _HUBER_DELTA,
    _LOG_2PI,
    _SPECTRUM_NFFT,
    _STAGE_A,
    _STAGE_B,
    _TINY,
)
from .._core._types import Alignment, Objective, Scaling, Transition
from .._core._validators import (
    _validate_choice,
    _validate_endog,
    _validate_exog,
    _validate_nonnegative,
    _validate_positive_int,
    _validate_time,
)
from ._fits import _SpectralFit
from ._modules import _SpectralNetwork
from ._predictors import _SpectralPredictor
from ._trainers import _train

__all__ = ["_DEFAULT_SCHEDULE", "_BaseModel", "_SpectralDecompositionModel"]

_DEFAULT_SCHEDULE: tuple[_Stage, ...] = (
    _Stage(_STAGE_A[0], _STAGE_A[1], _Penalties(amplitude=_STAGE_A[2], frequency=_STAGE_A[3])),
    _Stage(_STAGE_B[0], _STAGE_B[1], _Penalties(amplitude=_STAGE_B[2], frequency=_STAGE_B[3])),
)
r"""The published two-stage schedule.

Stage A: :math:`\eta = 10^{-3}`, 1,200 steps,
:math:`\lambda_A = 2 \times 10^{-4}`, :math:`\lambda_\omega = 8 \times 10^{-4}`.
Stage B: :math:`\eta = 4 \times 10^{-4}`, 1,800 steps,
:math:`\lambda_A = 3 \times 10^{-4}`, :math:`\lambda_\omega = 1.5 \times 10^{-3}`.
Every other penalty weight is zero in both.
"""


class _BaseModel[R](ABC):
    """Root of every model specification, fitting to a result of type ``R``.

    Validates the series once, exposes it read-only, and declares the
    abstract :meth:`fit`. Nothing else is shared across families.

    Args:
        endog: The series, one-dimensional, at least three observations.

    Raises:
        DimensionError: If the series is mis-shaped or too short.
        NumericalError: If it contains non-finite values.
    """

    __slots__ = ("_endog",)

    def __init__(self, endog: npt.ArrayLike) -> None:
        self._endog = _validate_endog(endog)

    @property
    def endog(self) -> npt.NDArray[np.float64]:
        """The validated series, a ``(n,)`` float64 copy of what was given."""
        return self._endog

    @property
    def nobs_total(self) -> int:
        r"""Observations :math:`n` in the sample, before the one-step shift to :math:`n - 1`."""
        return int(self._endog.shape[0])

    @abstractmethod
    def fit(self) -> R:
        """Estimate the model and return its result object.

        Returns:
            A frozen result of type ``R``.
        """


class _SpectralDecompositionModel[R](_BaseModel[R]):
    r"""Specification surface for the spectral decomposition family.

    Every argument is validated and frozen at construction; the
    properties expose the validated values; :meth:`_fit_family` does the
    estimation. The Nyquist rate of the grid,
    :math:`\omega_{\max} = \pi / \operatorname{median}(\Delta t)`, is
    computed here once and handed to the network as the upper edge of
    the ordered-frequency band.

    Args:
        endog: The series, one-dimensional, at least three observations.
        exog: Optional ``(n, p)`` covariates known at each :math:`t`; used
            as given.
        time: Optional strictly increasing grid; defaults to
            :math:`0, \ldots, n - 1`.
        k: Number of sinusoidal components :math:`K`.
        state_dim: Latent dimension :math:`d`.
        alignment: ``"filtered"`` (the published run: the emission at
            :math:`t` conditions on :math:`y_t`) or ``"predicted"`` (the
            emission at :math:`t` conditions on :math:`y_{t-1}`).
        transition: ``"gru"`` or ``"timeaware"``.
        positive_frequency: Softplus the frequency head.
        ordered_frequency: Strictly increasing frequencies inside the band
            :math:`[\omega_{\min}, \pi / \operatorname{median}(\Delta t)]`;
            implies positivity.
        gating: Add the sigmoid gate head.
        objective: ``"mse"`` or ``"huber"``.
        delta: Huber threshold :math:`\delta` on the scaled residual,
            strictly positive.
        penalties: Weights for stages that carry none; the stage-A
            defaults when ``None``.
        schedule: The stages, in order; the published two when ``None``.
        scaling: ``"minmax"`` or ``"none"``.
        seed: Seed for parameter initialization; ``None`` leaves torch's
            global generator untouched and the fit unreproducible.
        device: Torch device string.

    Raises:
        SpecificationError: If any option is outside its admissible set,
            ``delta`` is zero, the schedule is empty or holds something
            other than ``Stage`` records, ``penalties`` is not a record, or
            ``seed`` is not an integer.
        DimensionError: If the data are mis-shaped or misaligned.
        NumericalError: If the data contain non-finite values.
    """

    __slots__ = (
        "_alignment",
        "_delta",
        "_device",
        "_exog",
        "_frequency_band",
        "_gating",
        "_k",
        "_objective",
        "_ordered_frequency",
        "_penalties",
        "_positive_frequency",
        "_scaling",
        "_schedule",
        "_seed",
        "_state_dim",
        "_time",
        "_transition",
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        exog: npt.ArrayLike | None = None,
        time: npt.ArrayLike | None = None,
        k: int = 16,
        state_dim: int = 128,
        alignment: str = "filtered",
        transition: str = "gru",
        positive_frequency: bool = True,
        ordered_frequency: bool = False,
        gating: bool = True,
        objective: str = "mse",
        delta: float = _HUBER_DELTA,
        penalties: _Penalties | None = None,
        schedule: tuple[_Stage, ...] | None = None,
        scaling: str = "minmax",
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(endog)
        n = self.nobs_total
        self._exog = _validate_exog(exog, n)
        self._time = _validate_time(time, n)
        self._k = _validate_positive_int(k, "k")
        self._state_dim = _validate_positive_int(state_dim, "state_dim")
        self._alignment = _validate_choice(alignment, Alignment, "alignment")
        self._transition = _validate_choice(transition, Transition, "transition")
        self._positive_frequency = bool(positive_frequency)
        self._ordered_frequency = bool(ordered_frequency)
        self._gating = bool(gating)
        self._objective = _validate_choice(objective, Objective, "objective")
        self._delta = _validate_nonnegative(delta, "delta")
        if self._delta == 0.0:
            raise SpecificationError("delta must be strictly positive; got 0.")
        self._penalties = _Penalties() if penalties is None else penalties
        if not isinstance(self._penalties, _Penalties):
            raise SpecificationError("penalties must be a Penalties record.")
        self._schedule = _DEFAULT_SCHEDULE if schedule is None else tuple(schedule)
        if not self._schedule or not all(isinstance(s, _Stage) for s in self._schedule):
            raise SpecificationError("schedule must be a non-empty sequence of Stage records.")
        self._scaling = _validate_choice(scaling, Scaling, "scaling")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise SpecificationError(f"seed must be an integer or None; got {seed!r}.")
        self._seed = seed
        self._device = str(device)
        nyquist = math.pi / float(np.median(np.diff(self._time)))
        self._frequency_band = (_FREQUENCY_MIN, nyquist)

    @property
    def exog(self) -> npt.NDArray[np.float64] | None:
        """The validated covariates as a ``(n, p)`` float64 copy, or ``None``."""
        return self._exog

    @property
    def time(self) -> npt.NDArray[np.float64]:
        """The time grid, ``(n,)`` float64, strictly increasing."""
        return self._time

    @property
    def k(self) -> int:
        r"""Number of components :math:`K`."""
        return self._k

    @property
    def state_dim(self) -> int:
        r"""Latent dimension :math:`d`."""
        return self._state_dim

    @property
    def alignment(self) -> str:
        r"""What the emission at :math:`t` conditions on: ``"filtered"`` or ``"predicted"``."""
        return self._alignment

    @property
    def transition(self) -> str:
        """The transition kind: ``"gru"`` or ``"timeaware"``."""
        return self._transition

    @property
    def positive_frequency(self) -> bool:
        """Whether frequencies are constrained positive; ordering implies it."""
        return self._positive_frequency or self._ordered_frequency

    @property
    def ordered_frequency(self) -> bool:
        r"""Whether frequencies are strictly increasing in :math:`k` at every step."""
        return self._ordered_frequency

    @property
    def gating(self) -> bool:
        """Whether the gate head is present."""
        return self._gating

    @property
    def objective(self) -> str:
        """The prediction loss: ``"mse"`` or ``"huber"``."""
        return self._objective

    @property
    def penalties(self) -> _Penalties:
        """Model-level penalty weights, used by stages that carry none of their own."""
        return self._penalties

    @property
    def schedule(self) -> tuple[_Stage, ...]:
        """The training stages, in order."""
        return self._schedule

    @property
    def scaling(self) -> str:
        """The scaling choice: ``"minmax"`` or ``"none"``."""
        return self._scaling

    @property
    def seed(self) -> int | None:
        """The initialization seed, or ``None``."""
        return self._seed

    @property
    def device(self) -> str:
        """The torch device string."""
        return self._device

    @property
    def exog_dim(self) -> int:
        r"""Number of covariate columns :math:`p`; zero without covariates."""
        return 0 if self._exog is None else int(self._exog.shape[1])

    # -- estimation -----------------------------------------------------------

    def _network_config(self) -> dict[str, Any]:
        """Keyword arguments that rebuild the network.

        Returns:
            The constructor arguments of ``_SpectralNetwork``, which the
            predictor also keeps so a pickled result can rebuild its
            module.
        """
        return {
            "input_dim": 1 + self.exog_dim,
            "state_dim": self._state_dim,
            "k": self._k,
            "transition": self._transition,
            "positive_frequency": self._positive_frequency,
            "ordered_frequency": self._ordered_frequency,
            "gating": self._gating,
            "frequency_min": self._frequency_band[0],
            "frequency_max": self._frequency_band[1],
        }

    def _build_network(self) -> _SpectralNetwork:
        """Construct the network, seeding initialization without touching global state.

        With a seed, construction runs inside ``torch.random.fork_rng`` so
        the global generator is restored afterwards; without one, torch's
        current global state initializes the weights.

        Returns:
            The module, moved to the configured device.
        """
        config = self._network_config()
        if self._seed is None:
            return _SpectralNetwork(**config).to(self._device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._seed)
            network = _SpectralNetwork(**config)
        return network.to(self._device)

    def _fit_family(self) -> _SpectralFit:
        r"""Train on the sample and pack the raw outputs.

        Fits the scale, builds the network and its predictor, lays out the
        teacher-forced tensors under the alignment, runs the schedule,
        then runs the trained network once more over the sample for the
        paths and forms the one-step residuals
        :math:`e_t = y_t - \hat y_{t \mid t-1}` over ``endog[1:]``, their
        variance :math:`\hat\sigma^2` (floored at :math:`10^{-12}`) and
        the concentrated likelihood
        :math:`\ell = -\tfrac{m}{2}(\log 2\pi + \log\hat\sigma^2 + 1)`.

        Returns:
            The :class:`_SpectralFit`.

        Raises:
            NumericalError: If training diverges.
        """
        scale = (
            _MinMaxScale.fit(self._endog) if self._scaling == "minmax" else _MinMaxScale.identity()
        )
        network = self._build_network()
        predictor = _SpectralPredictor(
            network,
            scale,
            exog_dim=self.exog_dim,
            alignment=self._alignment,
            config=self._network_config(),
            device=self._device,
        )
        x, dt, target = predictor.inputs(self._endog, self._exog, self._time)
        history = _train(
            network,
            x,
            dt,
            target,
            schedule=self._schedule,
            penalties=self._penalties,
            objective=self._objective,
            delta=self._delta,
            nfft=_SPECTRUM_NFFT,
            history_every=_HISTORY_EVERY,
        )
        paths = predictor.emissions(self._endog, self._exog, self._time)
        fittedvalues = paths.emission[:-1] if self._alignment == "filtered" else paths.emission
        resid = self._endog[1:] - fittedvalues
        nobs = int(resid.shape[0])
        sigma2 = max(float(resid @ resid) / nobs, _TINY)
        llf = -0.5 * nobs * (_LOG_2PI + math.log(sigma2) + 1.0)
        return _SpectralFit(
            paths=paths,
            fittedvalues=fittedvalues,
            resid=resid,
            llf=llf,
            nobs=nobs,
            n_params=predictor.n_parameters,
            sigma2=sigma2,
            history=history,
            predictor=predictor,
        )
