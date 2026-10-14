# filepath: /src/ns_sdn/engine/_internals/_predictors.py
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
r"""The fitted network as a numpy-facing object.

A predictor owns a trained :class:`_SpectralNetwork`, the scaling
constants it was trained under, the alignment it was trained with, and
the configuration needed to rebuild it. It runs the network in its two
modes and returns numpy paths, on the original scale unless asked for in
training units, so nothing above this module touches torch. It pickles
through the state dict, which is what lets a result be serialized without
a reference to a live module.

The alignment is implemented here and nowhere else, as the layout of the
teacher-forced inputs (:meth:`_SpectralPredictor.inputs`): which
observation sits in the input row at :math:`t`, which is the target, and
what the first step size is. Everything downstream (the trainer, the
network, the result) sees rows and targets and is alignment-agnostic.

Example:
    >>> import numpy as np
    >>> from ns_sdn.spectral_decomposition import NSSDN, Stage
    >>> y = np.sin(0.3 * np.arange(60.0))
    >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 5),), seed=0).fit()
    >>> pred = res.predictor
    >>> pred.alignment, pred.mix.shape, pred.n_parameters == res.n_params
    ('filtered', (2,), True)
    >>> x, dt, target = pred.inputs(y, None, np.arange(60.0))
    >>> tuple(x.shape), float(dt[0]), bool(np.allclose(target.numpy(), pred.scale.transform(y)))
    ((60, 1), 0.0, True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from ...exceptions import DimensionError
from .._core._containers import _MinMaxScale
from ._modules import _SpectralNetwork

__all__ = ["_SpectralPaths", "_SpectralPredictor"]


@dataclass(frozen=True, slots=True)
class _SpectralPaths:
    r"""Every path one pass of the network produces, as float64 arrays.

    Under the filtered alignment a pass over :math:`y_0, \ldots, y_{n-1}`
    has :math:`n` rows and row :math:`t` is the emission that reconstructs
    :math:`y_t` from the state after :math:`y_t`; under the predicted
    alignment it has :math:`n - 1` rows and row :math:`i` is the emission
    that predicts :math:`y_{i+1}` from the state after :math:`y_i`. A
    recursive pass has one row per forecast step.

    On the original scale the trend and emission carry the slope and
    offset of the scaling map, the amplitude-type paths the slope alone,
    and the dimensionless paths (frequency, phase, offset, gate, state)
    nothing.

    Attributes:
        emission: ``(r,)`` the synthesized series :math:`\hat y_t`.
        trend: ``(r,)`` trend head :math:`B_t`.
        amplitude_raw: ``(r, k)`` softplus amplitude :math:`A_{t,k}` before
            the gate.
        amplitude: ``(r, k)`` gated amplitude :math:`g_{t,k} A_{t,k}`, the
            paper's figures.
        frequency: ``(r, k)`` instantaneous angular frequency
            :math:`\omega_{t,k}` in radians per unit time.
        phase: ``(r, k)`` cumulative phase :math:`\theta_{t,k}`.
        offset: ``(r, k)`` phase offset :math:`\phi_{t,k}`.
        gate: ``(r, k)`` gate :math:`g_{t,k} \in (0, 1)`.
        state: ``(r, d)`` latent state :math:`h_t`.
        components: ``(r, k)`` contributions
            :math:`w_k g_{t,k} A_{t,k} \sin(\theta_{t,k} + \phi_{t,k})`.
    """

    emission: npt.NDArray[np.float64]
    trend: npt.NDArray[np.float64]
    amplitude_raw: npt.NDArray[np.float64]
    amplitude: npt.NDArray[np.float64]
    frequency: npt.NDArray[np.float64]
    phase: npt.NDArray[np.float64]
    offset: npt.NDArray[np.float64]
    gate: npt.NDArray[np.float64]
    state: npt.NDArray[np.float64]
    components: npt.NDArray[np.float64]


class _SpectralPredictor:
    r"""A trained network with its scale and alignment, callable on numpy data.

    The one object a result keeps that is not a plain array. Inputs are
    converted to float32 tensors on the network's device, passes run under
    ``torch.no_grad``, and outputs come back as float64 arrays with the
    scaling undone unless ``scaled=True``.

    Args:
        network: The trained module; put into eval mode on construction.
        scale: The affine map the series was trained under.
        exog_dim: Number of exogenous columns :math:`p` the network
            expects; a pass with a different width is refused.
        alignment: ``"filtered"`` or ``"predicted"``.
        config: Keyword arguments that rebuild ``network`` (what
            ``_models._network_config`` returns), kept for pickling.
        device: Torch device string the network lives on.

    Example:
        A predictor survives a pickle round trip with its weights, scale
        and alignment, and the restored copy emits the same paths:

        >>> import pickle
        >>> import numpy as np
        >>> from ns_sdn.spectral_decomposition import NSSDN, Stage
        >>> y = np.sin(0.3 * np.arange(60.0)) + 2.0
        >>> res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 5),), seed=0).fit()
        >>> clone = pickle.loads(pickle.dumps(res.predictor))
        >>> grid = np.arange(60.0)
        >>> same = clone.emissions(y, None, grid).emission
        >>> bool(np.allclose(same, res.emission)), clone.scale == res.predictor.scale
        (True, True)
    """

    __slots__ = ("_alignment", "_config", "_device", "_exog_dim", "_network", "_scale")

    def __init__(
        self,
        network: _SpectralNetwork,
        scale: _MinMaxScale,
        *,
        exog_dim: int,
        alignment: str,
        config: dict[str, Any],
        device: str,
    ) -> None:
        self._network = network.eval()
        self._scale = scale
        self._exog_dim = int(exog_dim)
        self._alignment = alignment
        self._config = dict(config)
        self._device = device

    # -- metadata -----------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        """Trainable parameter count of the network, the result's ``n_params``."""
        return self._network.n_parameters

    @property
    def scale(self) -> _MinMaxScale:
        """The scaling constants: the identity map under ``scaling="none"``."""
        return self._scale

    @property
    def alignment(self) -> str:
        """The alignment the network was trained with; passes use the same one."""
        return self._alignment

    @property
    def mix(self) -> npt.NDArray[np.float64]:
        r"""The learned mixing vector :math:`w`, shape ``(k,)``, as a float64 copy."""
        return self._network.mix.detach().cpu().numpy().astype(np.float64)

    @property
    def network(self) -> _SpectralNetwork:
        """The underlying module, for anyone extending the package.

        Mutating it changes what the result reports; the result's arrays
        were computed from the weights at fit time and are not refreshed.
        """
        return self._network

    # -- tensors ------------------------------------------------------------

    def _tensor(self, values: npt.ArrayLike) -> Tensor:
        """A float32 tensor on the network's device from any array-like.

        Args:
            values: Numbers of any shape.

        Returns:
            The tensor.
        """
        return torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=torch.float32).to(
            self._device
        )

    def inputs(
        self,
        endog: npt.NDArray[np.float64],
        exog: npt.NDArray[np.float64] | None,
        time: npt.NDArray[np.float64],
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Build the teacher-forced tensors for a sample under the alignment.

        With :math:`\tilde y` the scaled series and :math:`\Delta t_t =
        t_t - t_{t-1}`, the filtered layout has :math:`n` rows,

        .. math::

           x_t = [\tilde y_t;\ \text{exog}_t], \qquad
           \Delta t_0 = 0,\ \Delta t_t = t_t - t_{t-1}, \qquad
           \text{target}_t = \tilde y_t, \qquad t = 0, \ldots, n - 1,

        so the first phase increment is zero and the emission at :math:`t`
        is scored against the observation it conditioned on. The predicted
        layout has :math:`n - 1` rows,

        .. math::

           x_{t-1} = [\tilde y_{t-1};\ \text{exog}_t], \qquad
           \Delta t_t = t_t - t_{t-1}, \qquad
           \text{target}_{t-1} = \tilde y_t, \qquad t = 1, \ldots, n - 1,

        so the row that targets :math:`y_t` carries the previous
        observation and the covariates of :math:`t` itself, which are
        assumed known at forecast time.

        Args:
            endog: ``(n,)`` on the original scale.
            exog: ``(n, p)`` or ``None``.
            time: ``(n,)`` strictly increasing.

        Returns:
            ``(x, dt, target)`` as float32 tensors on the training scale,
            with ``n`` rows under the filtered alignment and ``n - 1``
            under the predicted one.

        Raises:
            DimensionError: If the exogenous width does not match the fit.
        """
        if (0 if exog is None else exog.shape[1]) != self._exog_dim:
            raise DimensionError(
                f"the fit used {self._exog_dim} exogenous columns; "
                f"got {0 if exog is None else exog.shape[1]}."
            )
        ys = self._scale.transform(endog)
        steps = np.diff(time)
        if self._alignment == "filtered":
            parts = [ys[:, None]] + ([] if exog is None else [exog])
            dt = np.concatenate([[0.0], steps])
            target = ys
        else:
            parts = [ys[:-1, None]] + ([] if exog is None else [exog[1:]])
            dt = steps
            target = ys[1:]
        x = np.concatenate(parts, axis=1)
        return self._tensor(x), self._tensor(dt), self._tensor(target)

    def _paths(self, raw: dict[str, Tensor], *, scaled: bool) -> _SpectralPaths:
        r"""Move a pass to numpy and, unless ``scaled``, undo the scaling.

        Trend and emission receive :math:`a\tilde y + c`; amplitude-type
        paths receive the slope :math:`a` alone, since they are
        differences from the trend rather than levels; the dimensionless
        paths are returned as they are.

        Args:
            raw: The tensors a sequence or recursive pass returned.
            scaled: Leave every path in training units.

        Returns:
            The paths as float64 arrays.
        """
        out = {key: value.detach().cpu().numpy().astype(np.float64) for key, value in raw.items()}
        slope, offset = (1.0, 0.0) if scaled else (self._scale.slope, self._scale.offset)
        return _SpectralPaths(
            emission=out["yhat"] * slope + offset,
            trend=out["trend"] * slope + offset,
            amplitude_raw=out["amplitude_raw"] * slope,
            amplitude=out["amplitude"] * slope,
            frequency=out["frequency"],
            phase=out["phase"],
            offset=out["offset"],
            gate=out["gate"],
            state=out["state"],
            components=out["components"] * slope,
        )

    # -- modes --------------------------------------------------------------

    def emissions(
        self,
        endog: npt.NDArray[np.float64],
        exog: npt.NDArray[np.float64] | None,
        time: npt.NDArray[np.float64],
        *,
        scaled: bool = False,
    ) -> _SpectralPaths:
        """Teacher-forced pass: every input is a realized value.

        The pass the fitted paths, ``decompose`` and ``predict`` come
        from. The network starts from its learned initial state, so a pass
        over a longer series that begins with the fitted sample reproduces
        the fitted sample's rows.

        Args:
            endog: ``(n,)`` the series on the original scale.
            exog: ``(n, p)`` or ``None``.
            time: ``(n,)`` strictly increasing.
            scaled: Return the paths in training units.

        Returns:
            Paths with ``n`` rows (filtered) or ``n - 1`` rows (predicted).

        Raises:
            DimensionError: If the exogenous width does not match the fit.
        """
        x, dt, _ = self.inputs(endog, exog, time)
        with torch.no_grad():
            raw = self._network.forward_sequence(x, dt)
        return self._paths(raw, scaled=scaled)

    def recursive(
        self,
        endog: npt.NDArray[np.float64],
        exog: npt.NDArray[np.float64] | None,
        time: npt.NDArray[np.float64],
        *,
        steps: int,
        future_exog: npt.NDArray[np.float64] | None,
        future_time: npt.NDArray[np.float64] | None,
        scaled: bool = False,
    ) -> _SpectralPaths:
        r"""Warm up on the sample, then roll forward feeding each output back.

        The first future input is the last observation :math:`y_{n-1}`;
        every later input is the previous emission. Under the filtered
        alignment the warm-up has already consumed :math:`y_{n-1}`, so it
        is fed once more as the first future input, which is the
        validation notebook's recursive protocol. The state and phase the
        rollout continues from are the last row of the warm-up pass, and
        the future step sizes are
        :math:`\Delta t_{n+s} = t_{n+s} - t_{n+s-1}` with
        :math:`t_{n} \equiv` ``time[-1]``.

        Args:
            endog: ``(n,)`` the sample to condition on.
            exog: ``(n, p)`` or ``None``.
            time: ``(n,)`` the sample's grid.
            steps: Periods to forecast, :math:`S \ge 1`.
            future_exog: ``(steps, p)`` when the fit used exogenous
                columns; must be ``None`` otherwise.
            future_time: ``(steps,)`` grid after ``time[-1]``; defaults to
                continuing at the sample's last step size (one, for a
                sample of a single observation).
            scaled: Return the paths in training units.

        Returns:
            Paths of ``steps`` rows.

        Raises:
            DimensionError: If the future exogenous block is missing or
                mis-shaped for a fit that used covariates.
        """
        if self._exog_dim and (future_exog is None or future_exog.shape != (steps, self._exog_dim)):
            raise DimensionError(
                f"forecasting a fit with {self._exog_dim} exogenous columns needs "
                f"exog of shape ({steps}, {self._exog_dim})."
            )
        x, dt, _ = self.inputs(endog, exog, time)
        last_dt = float(time[-1] - time[-2]) if time.shape[0] > 1 else 1.0
        if future_time is None:
            dt_future = np.full(steps, last_dt, dtype=np.float64)
        else:
            dt_future = np.diff(np.concatenate([[time[-1]], future_time]))
        ys = self._scale.transform(endog)
        with torch.no_grad():
            warm = self._network.forward_sequence(x, dt)
            raw = self._network.forward_recursive(
                warm["state"][-1:],
                warm["phase"][-1:],
                self._tensor([[ys[-1]]]),
                None if future_exog is None else self._tensor(future_exog),
                self._tensor(dt_future),
            )
        return self._paths(raw, scaled=scaled)

    # -- pickling -----------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        """Serialize through the state dict, as numpy.

        Returns:
            The network configuration, the scale as a tuple, the exogenous
            width, the alignment, the device string, and every entry of
            the state dict as a numpy array, so the pickle holds no torch
            objects.
        """
        return {
            "config": self._config,
            "scale": tuple(self._scale),
            "exog_dim": self._exog_dim,
            "alignment": self._alignment,
            "device": self._device,
            "weights": {
                key: value.detach().cpu().numpy()
                for key, value in self._network.state_dict().items()
            },
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild the network from the configuration and load the weights.

        Args:
            state: What :meth:`__getstate__` produced.
        """
        network = _SpectralNetwork(**state["config"])
        network.load_state_dict({k: torch.from_numpy(v) for k, v in state["weights"].items()})
        self._network = network.to(state["device"]).eval()
        self._scale = _MinMaxScale(*state["scale"])
        self._exog_dim = int(state["exog_dim"])
        self._alignment = state["alignment"]
        self._config = dict(state["config"])
        self._device = state["device"]
