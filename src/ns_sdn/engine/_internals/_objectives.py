# filepath: /src/ns_sdn/engine/_internals/_objectives.py
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
"""The training criterion: a prediction loss plus drift and sparsity penalties.

The penalties are the TVP-VAR intuition that spectral parameters should
move slowly: squared first differences of the effective amplitude and of
the instantaneous frequency (the two the published run used), optional
second differences of the phase, first differences of the trend, a group
penalty that drives whole components to silence, and a power-spectrum
match. All are evaluated on the scaled series and reported separately in
the training history so a stage can be read as a trade-off, not a total.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .._core._containers import _Penalties

__all__ = ["_criterion"]


def _prediction_loss(residual: Tensor, objective: str, delta: float) -> Tensor:
    """Mean squared or Huber loss on the residual."""
    if objective == "huber":
        return F.huber_loss(residual, torch.zeros_like(residual), delta=delta)
    return residual.pow(2).mean()


def _spectrum_loss(target: Tensor, yhat: Tensor, nfft: int) -> Tensor:
    """Mean squared difference of rFFT power spectra."""
    power = torch.fft.rfft(target, n=nfft).abs().pow(2)
    power_hat = torch.fft.rfft(yhat, n=nfft).abs().pow(2)
    return F.mse_loss(power_hat, power)


def _criterion(
    target: Tensor,
    paths: dict[str, Tensor],
    weights: _Penalties,
    *,
    objective: str,
    delta: float,
    nfft: int,
) -> tuple[Tensor, dict[str, float]]:
    """Total loss and its parts.

    Args:
        target: ``(n,)`` scaled one-step targets.
        paths: Output of ``_SpectralNetwork.forward_sequence``.
        weights: Penalty weights.
        objective: ``"mse"`` or ``"huber"``.
        delta: Huber threshold.
        nfft: rFFT length for the spectrum penalty.

    Returns:
        The scalar loss to backpropagate, and each unweighted part as a float.
    """
    yhat = paths["yhat"]
    parts: dict[str, Tensor] = {"prediction": _prediction_loss(yhat - target, objective, delta)}
    total = parts["prediction"]

    if weights.amplitude > 0.0:
        parts["amplitude"] = torch.diff(paths["amplitude"], dim=0).pow(2).mean()
        total = total + weights.amplitude * parts["amplitude"]
    if weights.frequency > 0.0:
        parts["frequency"] = torch.diff(paths["frequency"], dim=0).pow(2).mean()
        total = total + weights.frequency * parts["frequency"]
    if weights.phase_curvature > 0.0 and yhat.shape[0] >= 3:
        parts["phase_curvature"] = torch.diff(paths["phase"], n=2, dim=0).pow(2).mean()
        total = total + weights.phase_curvature * parts["phase_curvature"]
    if weights.trend > 0.0:
        parts["trend"] = torch.diff(paths["trend"]).pow(2).mean()
        total = total + weights.trend * parts["trend"]
    if weights.sparsity > 0.0:
        parts["sparsity"] = paths["amplitude"].pow(2).mean(dim=0).sqrt().sum()
        total = total + weights.sparsity * parts["sparsity"]
    if weights.spectrum > 0.0:
        parts["spectrum"] = _spectrum_loss(target, yhat, nfft)
        total = total + weights.spectrum * parts["spectrum"]

    return total, {key: float(value.detach()) for key, value in parts.items()}
