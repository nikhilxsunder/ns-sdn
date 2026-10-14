# filepath: /src/ns_sdn/engine/_internals/_modules.py
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
"""The torch network: latent transition, spectral emission heads, synthesis.

This is the only module in the package that defines ``nn.Module``
subclasses. Everything it emits is a plain tensor; the predictor layer
turns those into numpy and the result layer never sees torch.

The network is the IEEE-ORSS configuration, equation by equation. A latent
state ``h_t`` evolves under a GRU driven by ``[y; x]`` (or a residual
time-aware cell that also takes ``dt``); five linear heads read ``h_t``
into a trend ``B_t``, raw amplitudes ``A_t = softplus(.) + eps``,
instantaneous frequencies ``omega_t = softplus(.) + eps``, phase offsets
``phi_t`` and gates ``g_t = sigmoid(.)``; phase integrates as ``theta_t =
theta_{t-1} + omega_t dt_t`` from a learned ``theta_0``; and the learned
mixing vector ``w`` of the paper's equation (4) combines the components,

    yhat_t = B_t + sum_k w_k g_{t,k} A_{t,k} sin(theta_{t,k} + phi_{t,k}).

With ``k = 16``, ``d = 128``, a GRU and no covariates the network has
58,849 trainable parameters, the count of the published run. The
per-component frequency anchor the paper's text describes was a fixed
zero buffer in that run and is the frequency head's bias here.

Under teacher forcing the inputs are known in advance, so the GRU runs as
one ``nn.GRU`` call over the sequence and the phase recursion is a
cumulative sum; no Python loop over time. The recursive forecast must
loop, because each input is the previous output.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...exceptions import SpecificationError
from .._core._defaults import _AMPLITUDE_FLOOR, _FREQUENCY_FLOOR

__all__ = ["_Heads", "_SpectralNetwork", "_TimeAwareCell"]


class _Heads(NamedTuple):
    """One emission: what the heads read off a block of states.

    Attributes:
        trend: ``(n,)`` the trend head ``B``.
        amplitude: ``(n, k)`` raw softplus amplitudes ``A``, before the gate.
        frequency: ``(n, k)`` instantaneous angular frequencies ``omega``.
        offset: ``(n, k)`` phase offsets ``phi``.
        gate: ``(n, k)`` gates ``g`` in ``(0, 1)``; ones when gating is off.
    """

    trend: Tensor
    amplitude: Tensor
    frequency: Tensor
    offset: Tensor
    gate: Tensor


class _TimeAwareCell(nn.Module):
    """Residual Euler transition ``h_t = h_{t-1} + dt_t * g(h_{t-1}, u_t)``.

    The step size multiplies the update instead of entering as a feature,
    which is the discrete-time analogue of a neural ODE.
    """

    def __init__(self, input_dim: int, state_dim: int, *, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, h_prev: Tensor, x_row: Tensor, dt: Tensor) -> Tensor:
        """Advance one step: ``h_prev (1, d)``, ``x_row (1, input_dim)``, ``dt (1, 1)``."""
        return h_prev + dt * self.net(torch.cat([h_prev, x_row], dim=1))


class _SpectralNetwork(nn.Module):
    """Latent transition plus spectral emission for one series.

    Args:
        input_dim: Width of a transition input row, ``1 + exog_dim``.
        state_dim: Latent dimension ``d``.
        k: Number of sinusoidal components.
        transition: ``"gru"`` or ``"timeaware"``.
        positive_frequency: Softplus the frequency head so phase always advances.
        ordered_frequency: Parameterize frequencies as cumulative positive gaps
            inside ``[frequency_min, frequency_max]``, so they are strictly
            increasing in ``k`` at every ``t`` and identified without permutation.
        gating: Add the sigmoid gate head.
        frequency_min: Lower edge of the ordered band.
        frequency_max: Upper edge of the ordered band (the Nyquist rate).
    """

    def __init__(
        self,
        *,
        input_dim: int,
        state_dim: int,
        k: int,
        transition: str,
        positive_frequency: bool,
        ordered_frequency: bool,
        gating: bool,
        frequency_min: float,
        frequency_max: float,
    ) -> None:
        super().__init__()
        if ordered_frequency and not frequency_min < frequency_max:
            raise SpecificationError(
                f"ordered frequencies need frequency_min < frequency_max; "
                f"got {frequency_min} and {frequency_max}."
            )
        self.k = int(k)
        self.state_dim = int(state_dim)
        self.transition_kind = transition
        self.positive_frequency = bool(positive_frequency)
        self.ordered_frequency = bool(ordered_frequency)
        self.frequency_min = float(frequency_min)
        self.frequency_max = float(frequency_max)

        self.cell: nn.Module
        if transition == "gru":
            self.cell = nn.GRU(input_dim, state_dim, batch_first=True)
        else:
            self.cell = _TimeAwareCell(input_dim, state_dim)

        self.trend = nn.Linear(state_dim, 1)
        self.amplitude = nn.Linear(state_dim, k)
        self.frequency = nn.Linear(state_dim, k + 1 if ordered_frequency else k)
        self.offset = nn.Linear(state_dim, k)
        self.gate = nn.Linear(state_dim, k) if gating else None
        self.mix = nn.Parameter(torch.empty(k))
        nn.init.uniform_(self.mix, -1.0 / math.sqrt(k), 1.0 / math.sqrt(k))
        self.h0 = nn.Parameter(torch.zeros(state_dim))
        self.theta0 = nn.Parameter(torch.zeros(k))

    # -- transition ---------------------------------------------------------

    def initial_state(self) -> Tensor:
        """The learned ``h_0`` as a ``(1, d)`` block."""
        return self.h0.view(1, -1)

    def step(self, h_prev: Tensor, x_row: Tensor, dt: Tensor) -> Tensor:
        """One transition from ``(1, d)`` with input ``(1, input_dim)`` and step ``(1, 1)``."""
        if isinstance(self.cell, nn.GRU):
            _, h_next = self.cell(x_row.view(1, 1, -1), h_prev.view(1, 1, -1))
            return h_next.view(1, -1)
        return self.cell(h_prev, x_row, dt)

    def states(self, x: Tensor, dt: Tensor) -> Tensor:
        """Run the transition over a known input sequence.

        Args:
            x: ``(n, input_dim)`` rows.
            dt: ``(n,)`` step sizes; read only by the time-aware cell.

        Returns:
            ``(n, d)`` states, ``h_1 .. h_n`` from the learned ``h_0``.
        """
        if isinstance(self.cell, nn.GRU):
            out, _ = self.cell(x.unsqueeze(0), self.initial_state().view(1, 1, -1))
            return out[0]
        h = self.initial_state()
        rows: list[Tensor] = []
        for n in range(x.shape[0]):
            h = self.cell(h, x[n : n + 1], dt[n].view(1, 1))
            rows.append(h)
        return torch.cat(rows, dim=0)

    # -- emission -----------------------------------------------------------

    def emit(self, h: Tensor) -> _Heads:
        """Read the five heads off a ``(n, d)`` block of states."""
        trend = self.trend(h)[:, 0]
        amplitude = F.softplus(self.amplitude(h)) + _AMPLITUDE_FLOOR
        raw = self.frequency(h)
        if self.ordered_frequency:
            gaps = torch.softmax(raw, dim=1)[:, : self.k]
            frequency = self.frequency_min + (
                self.frequency_max - self.frequency_min
            ) * torch.cumsum(gaps, dim=1)
        elif self.positive_frequency:
            frequency = F.softplus(raw) + _FREQUENCY_FLOOR
        else:
            frequency = raw
        offset = self.offset(h)
        gate = torch.sigmoid(self.gate(h)) if self.gate is not None else torch.ones_like(amplitude)
        return _Heads(trend, amplitude, frequency, offset, gate)

    def synthesize(self, heads: _Heads, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Equation (4): combine heads and phase into the prediction.

        Returns:
            ``(yhat (n,), amplitude (n, k), components (n, k))`` where
            ``amplitude`` is the gated amplitude ``g A`` the paper's figures
            plot and ``components`` are ``w_k g A sin(theta + phi)``.
        """
        amplitude = heads.gate * heads.amplitude
        components = self.mix * amplitude * torch.sin(theta + heads.offset)
        return heads.trend + components.sum(dim=1), amplitude, components

    def _collect(self, h: Tensor, heads: _Heads, theta: Tensor) -> dict[str, Tensor]:
        yhat, amplitude, components = self.synthesize(heads, theta)
        return {
            "yhat": yhat,
            "trend": heads.trend,
            "amplitude_raw": heads.amplitude,
            "amplitude": amplitude,
            "frequency": heads.frequency,
            "phase": theta,
            "offset": heads.offset,
            "gate": heads.gate,
            "state": h,
            "components": components,
        }

    # -- sequence modes -----------------------------------------------------

    def forward_sequence(self, x: Tensor, dt: Tensor) -> dict[str, Tensor]:
        """Teacher-forced pass over a known input sequence.

        Args:
            x: ``(n, input_dim)`` rows ``[y; exog]``.
            dt: ``(n,)`` step sizes for the phase recursion (and the
                time-aware cell).

        Returns:
            Paths keyed ``yhat, trend, amplitude_raw, amplitude, frequency,
            phase, offset, gate, state, components``, each ``(n, ...)``.
        """
        h = self.states(x, dt)
        heads = self.emit(h)
        theta = self.theta0.view(1, -1) + torch.cumsum(heads.frequency * dt[:, None], dim=0)
        return self._collect(h, heads, theta)

    def forward_recursive(
        self,
        h: Tensor,
        theta: Tensor,
        y_prev: Tensor,
        exog: Tensor | None,
        dt: Tensor,
    ) -> dict[str, Tensor]:
        """Free-running pass: each input after the first is the previous output.

        Args:
            h: ``(1, d)`` state to continue from.
            theta: ``(1, k)`` phase to continue from.
            y_prev: ``(1, 1)`` the first input, on the training scale.
            exog: ``(s, p)`` future exogenous rows, or ``None``.
            dt: ``(s,)`` future step sizes.

        Returns:
            The same keys as :meth:`forward_sequence`, each ``(s, ...)``.
        """
        rows: list[dict[str, Tensor]] = []
        for s in range(dt.shape[0]):
            x_row = y_prev if exog is None else torch.cat([y_prev, exog[s : s + 1]], dim=1)
            step = dt[s].view(1, 1)
            h = self.step(h, x_row, step)
            heads = self.emit(h)
            theta = theta + heads.frequency * step
            rows.append(self._collect(h, heads, theta))
            y_prev = rows[-1]["yhat"].view(1, 1)
        return {key: torch.cat([row[key] for row in rows], dim=0) for key in rows[0]}

    @property
    def n_parameters(self) -> int:
        """Trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
