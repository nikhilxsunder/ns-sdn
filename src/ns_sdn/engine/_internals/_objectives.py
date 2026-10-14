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
r"""The training criterion: a prediction loss plus drift and sparsity penalties.

The criterion is

.. math::

   \mathcal{L} = \mathcal{L}_{\mathrm{pred}}(\hat y - y)
   + \lambda_A \mathcal{P}_A + \lambda_\omega \mathcal{P}_\omega
   + \lambda_\theta \mathcal{P}_\theta + \lambda_B \mathcal{P}_B
   + \lambda_{\mathrm{sp}} \mathcal{P}_{\mathrm{sp}}
   + \lambda_{\mathrm{spec}} \mathcal{P}_{\mathrm{spec}},

with the prediction term the mean squared or Huber loss on the scaled
residual and the penalties the TVP-VAR intuition that spectral
parameters should move slowly: squared first differences of the gated
amplitude and of the instantaneous frequency (the two the published run
used), optional second differences of the phase, first differences of
the trend, a group penalty that drives whole components to silence, and
a power-spectrum match. The terms are written out on
:class:`~ns_sdn.spectral_decomposition.Penalties`. All are evaluated on
the scaled series and reported separately, unweighted, in the training
history so a stage can be read as a trade-off, not a total.

Example:
    >>> import torch
    >>> from ns_sdn.spectral_decomposition import Penalties
    >>> paths = {
    ...     "yhat": torch.zeros(4), "amplitude": torch.ones(4, 2), "frequency": torch.ones(4, 2),
    ...     "phase": torch.zeros(4, 2), "trend": torch.zeros(4),
    ... }
    >>> total, parts = _criterion(
    ...     torch.ones(4), paths, Penalties(), objective="mse", delta=1.0, nfft=8
    ... )
    >>> float(total), sorted(parts)
    (1.0, ['amplitude', 'frequency', 'prediction'])
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .._core._containers import _Penalties

__all__ = ["_criterion"]


def _prediction_loss(residual: Tensor, objective: str, delta: float) -> Tensor:
    r"""Mean squared or Huber loss on the residual.

    .. math::

       \mathcal{L}_{\mathrm{mse}} = \frac{1}{r}\sum_t e_t^2,
       \qquad
       \mathcal{L}_{\mathrm{huber}} = \frac{1}{r}\sum_t
       \begin{cases}
         \tfrac{1}{2} e_t^2, & |e_t| \le \delta,\\
         \delta(|e_t| - \tfrac{1}{2}\delta), & |e_t| > \delta.
       \end{cases}

    Args:
        residual: ``(r,)`` the scaled residual :math:`e = \hat y - y`.
        objective: ``"mse"`` or ``"huber"``.
        delta: The Huber threshold :math:`\delta`; ignored under ``"mse"``.

    Returns:
        A scalar tensor on the graph.
    """
    if objective == "huber":
        return F.huber_loss(residual, torch.zeros_like(residual), delta=delta)
    return residual.pow(2).mean()


def _spectrum_loss(target: Tensor, yhat: Tensor, nfft: int) -> Tensor:
    r"""Mean squared difference of rFFT power spectra.

    .. math::

       \mathcal{P}_{\mathrm{spec}} = \frac{1}{F}\sum_{f=0}^{F-1}
       \Bigl(|\mathcal{F}_F \hat y|_f^2 - |\mathcal{F}_F y|_f^2\Bigr)^2,
       \qquad F = \lfloor n_{\mathrm{fft}} / 2 \rfloor + 1,

    where :math:`\mathcal{F}_F` is the real FFT of the series zero-padded
    or truncated to ``nfft`` points. A match on power ignores phase, so
    this term asks the prediction to carry the right energy at each
    frequency, not to be in step.

    Args:
        target: ``(r,)`` the scaled targets.
        yhat: ``(r,)`` the scaled predictions.
        nfft: Transform length.

    Returns:
        A scalar tensor on the graph.
    """
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
    r"""Total loss and its parts.

    A penalty with weight zero is not evaluated at all, so a stage that
    uses only the published two terms builds exactly the published graph.
    The phase-curvature term needs at least three rows and is skipped
    otherwise.

    Args:
        target: ``(n,)`` scaled one-step targets, aligned with
            ``paths["yhat"]``.
        paths: Output of ``_SpectralNetwork.forward_sequence``; the keys
            ``yhat``, ``amplitude``, ``frequency``, ``phase`` and ``trend``
            are read as their weights require.
        weights: Penalty weights :math:`\lambda`.
        objective: ``"mse"`` or ``"huber"``.
        delta: Huber threshold :math:`\delta`.
        nfft: rFFT length for the spectrum penalty.

    Returns:
        The scalar loss to backpropagate, and each unweighted part as a
        float keyed ``prediction`` and the names of the penalties that
        were on.
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
