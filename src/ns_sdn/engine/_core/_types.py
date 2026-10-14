# filepath: /src/ns_sdn/engine/_core/_types.py
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
"""Categorical specification aliases.

PEP 695 aliases over ``Literal`` so a misspelled option is a type error at
the call site, and the same alias is what the runtime validator reads, so
the type and the check cannot drift apart.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["Alignment", "Objective", "Scaling", "Transition"]

type Alignment = Literal["filtered", "predicted"]
"""What the emission at step ``t`` conditions on, and therefore what it is.

``"filtered"`` is the IEEE-ORSS run: the transition at ``t`` consumes
``y_t`` and the emission is scored against ``y_t``, so the decomposition
at ``t`` is a filtered reading of the observed series given information
through ``t`` (the sense in which an empirical mode decomposition or a
Kalman filtered state is "of" the data). The one-step forecast of
``y_{t+1}`` under this alignment is the emission at ``t``, which is the
published holdout protocol. ``"predicted"`` shifts the input: the
transition at ``t`` consumes ``y_{t-1}`` and the emission is scored
against ``y_t``, so every path is a one-step-ahead prediction and the
network is trained as a forecaster.
"""

type Transition = Literal["gru", "timeaware"]
"""Latent transition ``F_psi``.

``"gru"`` is a gated recurrent cell on ``[y; x]``, the configuration of the
IEEE-ORSS experiments; the step size enters only the phase recursion.
``"timeaware"`` is the residual Euler update ``h_t = h_{t-1} + dt_t *
g(h_{t-1}, y, x)``, the discrete-time analogue of a neural ODE, in which
irregular spacing enters the dynamics explicitly.
"""

type Objective = Literal["mse", "huber"]
"""Prediction loss on the scaled residual.

``"mse"`` is the published objective. ``"huber"`` is the canonical
specification's robust alternative for series with jumps; its threshold is
``delta`` on the scaled series.
"""

type Scaling = Literal["minmax", "none"]
"""How the sample is scaled before training.

``"minmax"`` maps the sample to ``[-1, 1]`` with constants the result keeps
and inverts, so every reported path is on the original scale unless asked
for in training units. ``"none"`` trains on the raw series.
"""
