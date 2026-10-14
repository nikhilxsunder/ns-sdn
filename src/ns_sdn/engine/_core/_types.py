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
r"""Categorical specification aliases.

PEP 695 aliases over ``Literal`` so a misspelled option is a type error at
the call site, and the same alias is what the runtime validator
(``_validators._validate_choice``) reads, so the type and the check cannot
drift apart. Each alias documents what every one of its values does to
the fit; the public re-export is :mod:`ns_sdn.typing`.

Example:
    >>> from typing import get_args
    >>> get_args(Objective.__value__), get_args(Scaling.__value__)
    (('mse', 'huber'), ('minmax', 'none'))
"""

from __future__ import annotations

from typing import Literal

__all__ = ["Alignment", "Objective", "Scaling", "Transition"]

type Alignment = Literal["filtered", "predicted"]
r"""What the emission at step :math:`t` conditions on, and therefore what it is.

``"filtered"`` is the IEEE-ORSS run: the transition at :math:`t` consumes
:math:`y_t` and the emission is scored against :math:`y_t`,

.. math::

   h_t = F_\psi(h_{t-1}, y_t), \qquad \hat y_{t \mid t} = \hat y(h_t),
   \qquad \mathcal{L} \ni \bigl(\hat y_{t \mid t} - y_t\bigr)^2,

so the decomposition at :math:`t` is a filtered reading of the observed
series given information through :math:`t` (the sense in which an
empirical mode decomposition or a Kalman filtered state is "of" the
data). The one-step forecast of :math:`y_{t+1}` under this alignment is
the emission at :math:`t`, which is the published holdout protocol, and
the paths have :math:`n` rows.

``"predicted"`` shifts the input: the transition at :math:`t` consumes
:math:`y_{t-1}` and the emission is scored against :math:`y_t`,

.. math::

   h_t = F_\psi(h_{t-1}, y_{t-1}), \qquad \hat y_{t \mid t-1} = \hat y(h_t),
   \qquad \mathcal{L} \ni \bigl(\hat y_{t \mid t-1} - y_t\bigr)^2,

so every path is a one-step-ahead prediction, the network is trained as a
forecaster, and the paths have :math:`n - 1` rows.
"""

type Transition = Literal["gru", "timeaware"]
r"""Latent transition :math:`F_\psi`.

``"gru"`` is a gated recurrent cell on the input row :math:`x_t = [y; x]`,
the configuration of the IEEE-ORSS experiments,

.. math::

   r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r), \quad
   z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z), \quad
   \tilde h_t = \tanh\bigl(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h\bigr),

.. math::

   h_t = (1 - z_t) \odot \tilde h_t + z_t \odot h_{t-1},

in which the step size enters only the phase recursion. ``"timeaware"``
is the residual Euler update

.. math::

   h_t = h_{t-1} + \Delta t_t \, g_\psi\bigl([h_{t-1}; x_t]\bigr),

with :math:`g_\psi` a one-hidden-layer tanh network, the discrete-time
analogue of a neural ODE, in which irregular spacing enters the dynamics
explicitly. The GRU runs the teacher-forced pass as one recurrent call;
the time-aware cell loops over time in Python.
"""

type Objective = Literal["mse", "huber"]
r"""Prediction loss on the scaled residual :math:`e_t = \hat y_t - y_t`.

``"mse"`` is the published objective, :math:`\frac{1}{r}\sum_t e_t^2`.
``"huber"`` is the canonical specification's robust alternative for
series with jumps,

.. math::

   \frac{1}{r}\sum_t
   \begin{cases}
     \tfrac{1}{2} e_t^2, & |e_t| \le \delta,\\
     \delta\bigl(|e_t| - \tfrac{1}{2}\delta\bigr), & |e_t| > \delta,
   \end{cases}

with threshold :math:`\delta` (``delta``) on the scaled series, so under
min-max scaling :math:`\delta = 1` is half the range of the sample.
"""

type Scaling = Literal["minmax", "none"]
r"""How the sample is scaled before training.

``"minmax"`` maps the sample to :math:`[-1, 1]`,

.. math::

   \tilde y_t = 2\,\frac{y_t - \min y}{\max y - \min y} - 1,

with constants the result keeps and inverts, so every reported path is on
the original scale unless asked for in training units; the penalty
weights and the Huber threshold are then interpretable across series.
``"none"`` trains on the raw series, which is the right choice only when
the series is already of unit order.
"""
