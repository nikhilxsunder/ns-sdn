# Copyright (c) 2026 Nikhil Sunder
# SPDX-License-Identifier: MIT
"""Staged full-sequence training.

One Adam optimizer per stage, one gradient per step over the whole
sequence under teacher forcing, the published two-stage schedule as the
default. The target at step ``t`` is ``y_t`` and the input is ``y_{t-1}``:
the network is trained as a one-step forecaster, not as a reconstructor of
the observation it is given.
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

    Attributes:
        stage: Zero-based stage index.
        step: Zero-based step within the stage.
        loss: Total weighted loss.
        parts: Each loss term, unweighted.
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
    """Run every stage in order on one series.

    Args:
        network: The module to train in place.
        x: ``(n, input_dim)`` transition inputs.
        dt: ``(n,)`` step sizes.
        target: ``(n,)`` scaled targets.
        schedule: The stages.
        penalties: Weights for stages that do not carry their own.
        objective: Prediction loss.
        delta: Huber threshold.
        nfft: rFFT length for the spectrum penalty.
        history_every: Record the loss every this many steps, plus the last.

    Returns:
        The training history.

    Raises:
        NumericalError: If the loss becomes non-finite.
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
