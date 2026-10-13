# Copyright (c) 2026 Nikhil Sunder
# SPDX-License-Identifier: MIT
"""The fitted network as a numpy-facing object.

A predictor owns a trained :class:`_SpectralNetwork`, the scaling constants
it was trained under, the alignment it was trained with, and the
configuration needed to rebuild it. It runs the network in its two modes
and returns numpy paths, on the original scale unless asked for in
training units, so nothing above this module touches torch. It pickles
through the state dict, which is what lets a result be serialized without
a reference to a live module.
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
    """Every path one pass of the network produces.

    Under the filtered alignment a pass over ``y_0 .. y_{n-1}`` has ``n``
    rows and row ``t`` is the emission that reconstructs ``y_t`` from the
    state after ``y_t``; under the predicted alignment it has ``n - 1``
    rows and row ``i`` is the emission that predicts ``y_{i+1}`` from the
    state after ``y_i``.

    Attributes:
        emission: ``(r,)`` the synthesized series ``yhat``.
        trend: ``(r,)`` trend head ``B``.
        amplitude_raw: ``(r, k)`` softplus amplitude ``A`` before the gate.
        amplitude: ``(r, k)`` gated amplitude ``g A``, the paper's figures.
        frequency: ``(r, k)`` instantaneous angular frequency per unit time.
        phase: ``(r, k)`` cumulative phase ``theta``.
        offset: ``(r, k)`` phase offset ``phi``.
        gate: ``(r, k)`` gate ``g`` in ``(0, 1)``.
        state: ``(r, d)`` latent state.
        components: ``(r, k)`` contributions ``w_k g A sin(theta + phi)``.
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
    """A trained network with its scale and alignment, callable on numpy data.

    Args:
        network: The trained module.
        scale: The affine map the series was trained under.
        exog_dim: Number of exogenous columns the network expects.
        alignment: ``"filtered"`` or ``"predicted"``.
        config: Keyword arguments that rebuild ``network``.
        device: Torch device string the network lives on.
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
        """Trainable parameter count of the network."""
        return self._network.n_parameters

    @property
    def scale(self) -> _MinMaxScale:
        """The scaling constants."""
        return self._scale

    @property
    def alignment(self) -> str:
        """The alignment the network was trained with."""
        return self._alignment

    @property
    def mix(self) -> npt.NDArray[np.float64]:
        """The learned mixing vector ``w``, shape ``(k,)``."""
        return self._network.mix.detach().cpu().numpy().astype(np.float64)

    @property
    def network(self) -> _SpectralNetwork:
        """The underlying module, for anyone extending the package."""
        return self._network

    # -- tensors ------------------------------------------------------------

    def _tensor(self, values: npt.ArrayLike) -> Tensor:
        return torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=torch.float32).to(
            self._device
        )

    def inputs(
        self,
        endog: npt.NDArray[np.float64],
        exog: npt.NDArray[np.float64] | None,
        time: npt.NDArray[np.float64],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build the teacher-forced tensors for a sample under the alignment.

        Args:
            endog: ``(n,)`` on the original scale.
            exog: ``(n, p)`` or ``None``.
            time: ``(n,)`` strictly increasing.

        Returns:
            ``(x, dt, target)`` on the training scale. Filtered: ``n`` rows,
            row ``t`` is ``[y_t; exog_t]`` with ``dt_0 = 0`` and target
            ``y_t``. Predicted: ``n - 1`` rows, row ``t - 1`` is ``[y_{t-1};
            exog_t]`` with ``dt_t = time_t - time_{t-1}`` and target ``y_t``.

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
        """Move a pass to numpy and, unless ``scaled``, undo the scaling."""
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

        Returns:
            Paths with ``n`` rows (filtered) or ``n - 1`` rows (predicted).
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
        """Warm up on the sample, then roll forward feeding each output back.

        The first future input is the last observation ``y_{n-1}``; every
        later input is the previous emission. Under the filtered alignment
        the warm-up has already consumed ``y_{n-1}``, so it is fed once more
        as the first future input, which is the validation notebook's
        recursive protocol.

        Args:
            endog: ``(n,)`` the sample to condition on.
            exog: ``(n, p)`` or ``None``.
            time: ``(n,)`` the sample's grid.
            steps: Periods to forecast.
            future_exog: ``(steps, p)`` when the fit used exogenous columns.
            future_time: ``(steps,)`` grid after ``time[-1]``; defaults to
                continuing at the sample's last step size.
            scaled: Return the paths in training units.

        Returns:
            Paths of length ``steps``.

        Raises:
            DimensionError: If the future exogenous block is missing or misshaped.
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
        """Serialize through the state dict, as numpy."""
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
        """Rebuild the network from the configuration and load the weights."""
        network = _SpectralNetwork(**state["config"])
        network.load_state_dict({k: torch.from_numpy(v) for k, v in state["weights"].items()})
        self._network = network.to(state["device"]).eval()
        self._scale = _MinMaxScale(*state["scale"])
        self._exog_dim = int(state["exog_dim"])
        self._alignment = state["alignment"]
        self._config = dict(state["config"])
        self._device = state["device"]
