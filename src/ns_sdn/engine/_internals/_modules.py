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
r"""The torch network: latent transition, spectral emission heads, synthesis.

This is the only module in the package that defines ``nn.Module``
subclasses. Everything it emits is a plain tensor; the predictor layer
turns those into numpy and the result layer never sees torch.

The network is the IEEE-ORSS configuration, equation by equation. A
latent state :math:`h_t \in \mathbb{R}^d` evolves under a GRU driven by
the input row :math:`x_t = [y; x]` (or a residual time-aware cell that
also takes :math:`\Delta t_t`),

.. math::

   h_t = F_\psi(h_{t-1}, x_t, \Delta t_t), \qquad h_0 \text{ learned};

five linear heads read :math:`h_t` into a trend, raw amplitudes,
instantaneous frequencies, phase offsets and gates,

.. math::

   B_t = W_B h_t + b_B, \quad
   A_{t} = \operatorname{softplus}(W_A h_t + b_A) + \varepsilon_A, \quad
   \omega_{t} = \operatorname{softplus}(W_\omega h_t + b_\omega) + \varepsilon_\omega,

.. math::

   \phi_{t} = W_\phi h_t + b_\phi, \qquad
   g_{t} = \sigma(W_g h_t + b_g);

the phase integrates the frequency from a learned :math:`\theta_0`,

.. math::

   \theta_{t} = \theta_{t-1} + \omega_{t}\,\Delta t_t
   \quad\Longleftrightarrow\quad
   \theta_t = \theta_0 + \sum_{s \le t} \omega_s\,\Delta t_s;

and the learned mixing vector :math:`w \in \mathbb{R}^K` of the paper's
equation (4) combines the components,

.. math::

   \hat y_t = B_t + \sum_{k=1}^{K} w_k\, g_{t,k}\, A_{t,k}\,
   \sin\!\bigl(\theta_{t,k} + \phi_{t,k}\bigr).

With :math:`K = 16`, :math:`d = 128`, a GRU and no covariates the network
has 58,849 trainable parameters, the count of the published run. The
per-component frequency anchor the paper's text describes was a fixed
zero buffer in that run and is the frequency head's bias here.

Under teacher forcing the inputs are known in advance, so the GRU runs as
one ``nn.GRU`` call over the sequence and the phase recursion is a
cumulative sum; no Python loop over time. The recursive forecast must
loop, because each input is the previous output.

Example:
    >>> import torch
    >>> net = _SpectralNetwork(
    ...     input_dim=1, state_dim=128, k=16, transition="gru", positive_frequency=True,
    ...     ordered_frequency=False, gating=True, frequency_min=1e-3, frequency_max=3.14,
    ... )
    >>> net.n_parameters
    58849
    >>> paths = net.forward_sequence(torch.zeros(10, 1), torch.ones(10))
    >>> paths["yhat"].shape, paths["phase"].shape, paths["state"].shape
    (torch.Size([10]), torch.Size([10, 16]), torch.Size([10, 128]))
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
    r"""One emission: what the five heads read off a block of states.

    The heads are evaluated together on a ``(n, d)`` block, so a
    teacher-forced pass reads them once for the whole sequence and the
    recursive pass once per step on a ``(1, d)`` block.

    Attributes:
        trend: ``(n,)`` the trend head :math:`B_t`.
        amplitude: ``(n, k)`` raw softplus amplitudes :math:`A_{t,k}`,
            before the gate, at least :math:`\varepsilon_A`.
        frequency: ``(n, k)`` instantaneous angular frequencies
            :math:`\omega_{t,k}` in radians per unit time.
        offset: ``(n, k)`` phase offsets :math:`\phi_{t,k}`, unconstrained.
        gate: ``(n, k)`` gates :math:`g_{t,k} \in (0, 1)`; ones when gating
            is off.
    """

    trend: Tensor
    amplitude: Tensor
    frequency: Tensor
    offset: Tensor
    gate: Tensor


class _TimeAwareCell(nn.Module):
    r"""Residual Euler transition :math:`h_t = h_{t-1} + \Delta t_t\, g_\psi([h_{t-1}; u_t])`.

    The step size multiplies the update instead of entering as a feature,
    which is the discrete-time analogue of a neural ODE
    :math:`\dot h = g_\psi(h, u)` integrated by one explicit Euler step per
    observation. :math:`g_\psi` is a one-hidden-layer tanh network from
    :math:`\mathbb{R}^{d + u}` to :math:`\mathbb{R}^d`. A zero step leaves
    the state unchanged, so repeated timestamps would stall the dynamics;
    the time validator refuses them.

    Args:
        input_dim: Width :math:`u` of the input row.
        state_dim: Latent dimension :math:`d`.
        hidden_dim: Width of the hidden layer of :math:`g_\psi`.

    Example:
        >>> import torch
        >>> cell = _TimeAwareCell(input_dim=1, state_dim=4)
        >>> h = torch.zeros(1, 4)
        >>> bool(torch.equal(cell(h, torch.ones(1, 1), torch.zeros(1, 1)), h))
        True
    """

    def __init__(self, input_dim: int, state_dim: int, *, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, h_prev: Tensor, x_row: Tensor, dt: Tensor) -> Tensor:
        r"""Advance one step.

        Args:
            h_prev: ``(1, d)`` the state :math:`h_{t-1}`.
            x_row: ``(1, input_dim)`` the input :math:`u_t`.
            dt: ``(1, 1)`` the step :math:`\Delta t_t`.

        Returns:
            ``(1, d)`` the state :math:`h_t`.
        """
        return h_prev + dt * self.net(torch.cat([h_prev, x_row], dim=1))


class _SpectralNetwork(nn.Module):
    r"""Latent transition plus spectral emission for one series.

    Holds the cell, the five heads, the mixing vector :math:`w`, the
    initial state :math:`h_0` and the initial phase :math:`\theta_0`, and
    runs them in the two sequence modes: teacher-forced over known inputs
    (:meth:`forward_sequence`) and free-running from a state
    (:meth:`forward_recursive`).

    The frequency head has three parameterizations. Linear (neither flag),
    :math:`\omega_t = W_\omega h_t + b_\omega`, which can run a component
    backwards. Positive (``positive_frequency``),
    :math:`\omega_t = \operatorname{softplus}(W_\omega h_t + b_\omega)
    + \varepsilon_\omega`, the published run. Ordered
    (``ordered_frequency``), in which the head emits :math:`K + 1` logits
    :math:`r_t`, and

    .. math::

       \omega_{t,k} = \omega_{\min} + (\omega_{\max} - \omega_{\min})
       \sum_{j=1}^{k} \operatorname{softmax}(r_t)_j,
       \qquad k = 1, \ldots, K,

    so the frequencies are strictly increasing in :math:`k` and the
    :math:`(K+1)`-th gap keeps :math:`\omega_{t,K}` strictly below
    :math:`\omega_{\max}`. Initialization is torch's defaults for the cell
    and the heads, :math:`w \sim U(-1/\sqrt{K}, 1/\sqrt{K})`,
    :math:`h_0 = 0` and :math:`\theta_0 = 0`.

    Args:
        input_dim: Width of a transition input row, ``1 + exog_dim``.
        state_dim: Latent dimension :math:`d`.
        k: Number of sinusoidal components :math:`K`.
        transition: ``"gru"`` or ``"timeaware"``.
        positive_frequency: Softplus the frequency head so phase always
            advances.
        ordered_frequency: Parameterize frequencies as cumulative positive
            gaps inside ``[frequency_min, frequency_max]``, so they are
            strictly increasing in :math:`k` at every :math:`t` and
            identified without permutation. Takes precedence over
            ``positive_frequency``.
        gating: Add the sigmoid gate head.
        frequency_min: Lower edge :math:`\omega_{\min}` of the ordered band.
        frequency_max: Upper edge :math:`\omega_{\max}` of the ordered band
            (the Nyquist rate of the grid).

    Raises:
        SpecificationError: If ``ordered_frequency`` is set and the band is
            empty (``frequency_min >= frequency_max``).

    Example:
        Ordered frequencies are strictly increasing across components at
        every step and stay inside the band:

        >>> import torch
        >>> net = _SpectralNetwork(
        ...     input_dim=1, state_dim=8, k=4, transition="gru", positive_frequency=True,
        ...     ordered_frequency=True, gating=False, frequency_min=0.01, frequency_max=1.0,
        ... )
        >>> freq = net.forward_sequence(torch.randn(20, 1), torch.ones(20))["frequency"]
        >>> bool((freq.diff(dim=1) > 0).all()), bool((freq > 0.01).all() and (freq < 1.0).all())
        (True, True)
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
        r"""The learned :math:`h_0` as a ``(1, d)`` block.

        Returns:
            A view of the ``h0`` parameter, so gradients reach it.
        """
        return self.h0.view(1, -1)

    def step(self, h_prev: Tensor, x_row: Tensor, dt: Tensor) -> Tensor:
        r"""One transition :math:`h_t = F_\psi(h_{t-1}, x_t, \Delta t_t)`.

        Used by the recursive pass, where inputs arrive one at a time. The
        GRU ignores ``dt``; the time-aware cell multiplies its update by it.

        Args:
            h_prev: ``(1, d)`` the state :math:`h_{t-1}`.
            x_row: ``(1, input_dim)`` the input :math:`x_t`.
            dt: ``(1, 1)`` the step :math:`\Delta t_t`.

        Returns:
            ``(1, d)`` the state :math:`h_t`.
        """
        if isinstance(self.cell, nn.GRU):
            _, h_next = self.cell(x_row.view(1, 1, -1), h_prev.view(1, 1, -1))
            return h_next.view(1, -1)
        return self.cell(h_prev, x_row, dt)

    def states(self, x: Tensor, dt: Tensor) -> Tensor:
        r"""Run the transition over a known input sequence.

        Under the GRU this is one ``nn.GRU`` call with the learned
        :math:`h_0` as the initial hidden state; under the time-aware cell
        it is a Python loop over the rows.

        Args:
            x: ``(n, input_dim)`` rows :math:`x_1, \ldots, x_n`.
            dt: ``(n,)`` step sizes; read only by the time-aware cell.

        Returns:
            ``(n, d)`` states :math:`h_1, \ldots, h_n` from the learned
            :math:`h_0`.
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
        r"""Read the five heads off a block of states.

        Applies the trend, amplitude, frequency, offset and gate heads with
        their links: softplus plus :math:`\varepsilon_A` on the amplitude,
        the configured frequency parameterization, identity on the trend
        and offset, sigmoid on the gate (ones when there is no gate head).

        Args:
            h: ``(n, d)`` states.

        Returns:
            The :class:`_Heads` with ``n`` rows.
        """
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
        r"""Equation (4): combine heads and phase into the prediction.

        .. math::

           \bar A_{t,k} = g_{t,k} A_{t,k}, \qquad
           c_{t,k} = w_k\, \bar A_{t,k} \sin(\theta_{t,k} + \phi_{t,k}), \qquad
           \hat y_t = B_t + \sum_k c_{t,k}.

        Args:
            heads: The heads read off ``n`` states.
            theta: ``(n, k)`` the cumulative phase :math:`\theta_{t,k}`.

        Returns:
            ``(yhat, amplitude, components)`` of shapes ``(n,)``,
            ``(n, k)`` and ``(n, k)``, where ``amplitude`` is the gated
            amplitude :math:`\bar A` the paper's figures plot and
            ``components`` are the :math:`c_{t,k}`.
        """
        amplitude = heads.gate * heads.amplitude
        components = self.mix * amplitude * torch.sin(theta + heads.offset)
        return heads.trend + components.sum(dim=1), amplitude, components

    def _collect(self, h: Tensor, heads: _Heads, theta: Tensor) -> dict[str, Tensor]:
        """Synthesize and lay every path out under the keys the predictor expects.

        Args:
            h: ``(n, d)`` states.
            heads: The heads read off them.
            theta: ``(n, k)`` the phase.

        Returns:
            Tensors keyed ``yhat, trend, amplitude_raw, amplitude, frequency,
            phase, offset, gate, state, components``.
        """
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
        r"""Teacher-forced pass over a known input sequence.

        States by one recurrent call, heads by one batched read, and the
        phase by a cumulative sum,
        :math:`\theta_t = \theta_0 + \sum_{s \le t} \omega_s \Delta t_s`,
        so the whole pass is loop-free under the GRU. This is the pass
        training differentiates through and the pass the fitted paths come
        from.

        Args:
            x: ``(n, input_dim)`` rows :math:`[y; \text{exog}]` on the
                training scale, laid out under the alignment by the
                predictor.
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
        r"""Free-running pass: each input after the first is the previous output.

        .. math::

           h_{s} = F_\psi(h_{s-1}, [\hat y_{s-1}; \text{exog}_s], \Delta t_s), \qquad
           \theta_{s} = \theta_{s-1} + \omega_{s}\,\Delta t_s, \qquad
           \hat y_s = \hat y(h_s, \theta_s),

        with :math:`\hat y_0` the given ``y_prev``. Loops over the horizon
        in Python because each input depends on the previous output.

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
        r"""Trainable parameter count.

        For the GRU transition with input width :math:`u` this is
        :math:`3(d(u + d) + 2d)` for the cell, :math:`(d + 1)(1 + 4K)` for
        the five heads (:math:`(d + 1)(2 + 4K)` under ordered
        frequencies, :math:`K(d + 1)` fewer without the gate), plus
        :math:`K` for :math:`w`, :math:`d` for :math:`h_0` and :math:`K`
        for :math:`\theta_0`.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
