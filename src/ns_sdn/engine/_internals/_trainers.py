# filepath: /src/ns_sdn/engine/_internals/_trainers.py
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
r"""Staged full-sequence training.

One Adam optimizer per stage, one gradient per step over the whole
sequence under teacher forcing, the published two-stage schedule as the
default. Each step is

.. math::

   \psi \leftarrow \operatorname{Adam}_\eta\!\Bigl(\psi,\;
   \nabla_\psi\, \mathcal{L}\bigl(\hat y(\psi; x_{1:r}, \Delta t_{1:r}),\, y_{1:r}\bigr)\Bigr),

a full forward pass over the :math:`r` rows the alignment laid out, the
criterion of ``_objectives``, one backward pass and one optimizer step;
there is no mini-batching, no truncation of backpropagation through
time, and no learning-rate schedule within a stage. The inputs and
targets are built once by the predictor under the alignment: under
``"predicted"`` the input at :math:`t` is :math:`y_{t-1}` and the target
:math:`y_t`, under ``"filtered"`` both are :math:`y_t`.

The history records the total and every unweighted part at step 0, every
``history_every`` steps and the last step of each stage, which is enough
to plot a loss curve per term without storing one per step.

References:
    Kingma, D. P. & Ba, J. (2015). Adam: a method for stochastic
    optimization. *ICLR*.

    Williams, R. J. & Zipser, D. (1989). A learning algorithm for
    continually running fully recurrent neural networks. *Neural
    Computation*, 1(2), 270-280. For teacher forcing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ...exceptions import NumericalError
from .._core._containers import _Penalties, _Stage
from ._modules import _SpectralNetwork
from ._objectives import _criterion

__all__ = ["_StageRecord", "_train"]


@dataclass(frozen=True, slots=True)
class _StageRecord:
    """One recorded point of the training history.

    The value of the loss *before* the optimizer step at that point, so
    the record at step 0 of a stage is the loss the stage started from.

    Attributes:
        stage: Zero-based stage index.
        step: Zero-based step within the stage.
        loss: Total weighted loss.
        parts: Each loss term, unweighted, keyed ``prediction`` and the
            names of the penalties that were on.

    Example:
        >>> record = _StageRecord(stage=0, step=100, loss=0.5, parts={"prediction": 0.4})
        >>> record.stage, round(record.loss - record.parts["prediction"], 6)
        (0, 0.1)
    """

    stage: int
    step: int
    loss: float
    parts: dict[str, float]


def _train(
    network: _SpectralNetwork,
    x: Tensor,
    dt: Tensor,
    target: Tensor,
    *,
    schedule: tuple[_Stage, ...],
    penalties: _Penalties,
    objective: str,
    delta: float,
    nfft: int,
    history_every: int,
) -> tuple[_StageRecord, ...]:
    r"""Run every stage in order on one series.

    Puts the network in training mode, constructs a fresh Adam optimizer
    at each stage's learning rate (so the moment estimates restart with
    the stage), runs the stage's steps, and leaves the network in eval
    mode. Divergence is checked on every step before the backward pass,
    so a non-finite loss never reaches the parameters.

    Args:
        network: The module to train in place.
        x: ``(n, input_dim)`` transition inputs on the training scale.
        dt: ``(n,)`` step sizes.
        target: ``(n,)`` scaled targets.
        schedule: The stages, in order; at least one.
        penalties: Weights for stages that do not carry their own.
        objective: Prediction loss, ``"mse"`` or ``"huber"``.
        delta: Huber threshold :math:`\delta`.
        nfft: rFFT length for the spectrum penalty.
        history_every: Record the loss every this many steps, plus the
            last step of each stage.

    Returns:
        The training history, one record per recorded step across all
        stages.

    Raises:
        NumericalError: If the loss becomes non-finite; the message names
            the stage and step.
    """
    history: list[_StageRecord] = []
    network.train()
    for index, stage in enumerate(schedule):
        weights = penalties if stage.penalties is None else stage.penalties
        optimizer = torch.optim.Adam(network.parameters(), lr=stage.learning_rate)
        for step in range(stage.steps):
            optimizer.zero_grad(set_to_none=True)
            paths = network.forward_sequence(x, dt)
            loss, parts = _criterion(
                target, paths, weights, objective=objective, delta=delta, nfft=nfft
            )
            if not torch.isfinite(loss):
                raise NumericalError(
                    f"training diverged at stage {index}, step {step}: loss is {float(loss)}. "
                    "Lower the learning rate or scale the series."
                )
            loss.backward()
            optimizer.step()
            if step % history_every == 0 or step == stage.steps - 1:
                history.append(_StageRecord(index, step, float(loss.detach()), parts))
    network.eval()
    return tuple(history)
