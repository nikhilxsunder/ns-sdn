.. meta::
   :description: The NS-SDN model: the state transition, the five emission heads, the phase recursion, the synthesis equation, the training objective with its drift penalties, the staged schedule, and the published configuration.

The model
=========

NS-SDN represents a scalar series as a trend plus :math:`K` sinusoids
whose amplitude, instantaneous frequency, phase offset and participation
are read off a latent state at every step. It sits between three
traditions it borrows from: the state-space model, whose latent state
summarizes the past :cite:p:`durbin2012ssm`; empirical mode
decomposition, whose intrinsic mode functions are amplitude- and
frequency-modulated oscillations :cite:p:`huang1998emd`; and implicit
neural representations with sinusoidal activations
:cite:p:`sitzmann2020siren`. This page states the model as the package
implements it, which is the IEEE-ORSS formulation
:cite:p:`sunder2026orss` of the architecture introduced at FLAIRS
:cite:p:`sunder2026flairs`.

The state
---------

A gated recurrent unit :cite:p:`cho2014gru` carries the state,

.. math::

   h_t = \mathrm{GRU}_\psi\bigl(h_{t-1},\, x_t\bigr),
   \qquad
   x_t = \bigl[\,y_t;\; z_t\,\bigr],

where :math:`y_t` is the scaled observation, :math:`z_t` the optional
covariates, and :math:`h_0` a learned vector. The state dimension
:math:`d` is ``state_dim``; the published run uses :math:`d = 128`.
Under ``transition="timeaware"`` the cell also receives the step
:math:`\Delta t_t`, so an irregular grid changes the dynamics and not
only the phase; under the default ``"gru"`` the step enters the phase
recursion alone.

Which observation :math:`x_t` carries is the alignment. Under
``"filtered"`` it is :math:`y_t` and the emission at :math:`t`
describes :math:`y_t`; under ``"predicted"`` it is :math:`y_{t-1}` and
the emission at :math:`t` predicts :math:`y_t`. The
:doc:`forecasting page <forecasting>` has the consequences; the rest of
this page holds for both.

The heads
---------

Five linear maps read the spectral quantities off :math:`h_t`, each
through the link that gives it the right range:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Head
     - Link
     - Reading
   * - Trend :math:`B_t`
     - identity
     - The level the oscillations sit on. Unconstrained.
   * - Amplitude :math:`A_{t,k}`
     - softplus + :math:`10^{-4}`
     - Positive. The ``amplitude_raw`` path.
   * - Frequency :math:`\omega_{t,k}`
     - softplus + :math:`10^{-4}` (default), or ordered gaps
     - Radians per unit time. Positive, so the phase always advances.
   * - Offset :math:`\phi_{t,k}`
     - identity
     - A phase shift the state can apply without touching the frequency.
   * - Gate :math:`g_{t,k}`
     - sigmoid
     - In :math:`(0, 1)`. Participation of component :math:`k` at :math:`t`.

A learned mixing vector :math:`w \in \mathbb{R}^K` completes the
emission; it is the implementation note of the IEEE-ORSS paper and the
only spectral quantity that does not vary with :math:`t`. Its sign is
what lets a component contribute with either orientation, and its
magnitude is folded into the ``effective_amplitude`` the spectral state
reads power from.

With ``ordered_frequency=True`` the frequency head instead emits
:math:`K + 1` logits, takes a softmax over them, and accumulates the
first :math:`K` as gaps inside the band :math:`[\omega_{\min},
\omega_{\max}]`:

.. math::

   \omega_{t,k} = \omega_{\min} + (\omega_{\max} - \omega_{\min})
   \sum_{j \le k} \mathrm{softmax}(r_t)_j,

so :math:`\omega_{t,1} < \cdots < \omega_{t,K}` at every :math:`t` and
the components are identified without permutation. The band is
:math:`\omega_{\min} = 10^{-3}` to the Nyquist frequency
:math:`\pi / \mathrm{median}(\Delta t)`. The option is off by default
because the published run did not use it.

The phase
---------

The phase integrates the instantaneous frequency,

.. math::

   \theta_{t,k} = \theta_{t-1,k} + \omega_{t,k}\,\Delta t_t,
   \qquad \theta_{0,k} \text{ learned},

so a component is a chirp: a sinusoid whose frequency the state can move
step by step. This is the reason the model can describe non-stationary
oscillations with a fixed number of components, and the reason the
``frequency`` path is readable as an instantaneous frequency in the
sense of the Hilbert spectrum. Under teacher forcing the whole recursion
is one cumulative sum, so the forward pass is a single recurrent call
rather than a Python loop.

The synthesis
-------------

.. math::
   :label: synthesis

   \hat y_t = B_t + \sum_{k=1}^{K} w_k\, g_{t,k}\, A_{t,k}\,
   \sin\!\bigl(\theta_{t,k} + \phi_{t,k}\bigr).

The result exposes each factor: ``trend`` is :math:`B_t`,
``amplitude`` is the gated :math:`g_{t,k} A_{t,k}` (the paper's
figures), ``components`` is the full summand, and ``emission`` is
:math:`\hat y_t`. The identity ``trend + components.sum(axis=1) ==
emission`` holds to floating-point precision on every result, and the
:doc:`decomposition page <decomposition>` is organised around it.

The objective
-------------

The network is trained on the scaled series by minimizing a prediction
loss plus drift penalties,

.. math::

   \mathcal{L} = \frac{1}{n}\sum_t \ell\bigl(\hat y_t - y_t\bigr)
   + \lambda_A \,\overline{\bigl(\Delta_t\, g_{t,k} A_{t,k}\bigr)^2}
   + \lambda_\omega \,\overline{\bigl(\Delta_t\, \omega_{t,k}\bigr)^2}
   + \cdots,

where :math:`\ell` is the squared error (``objective="mse"``) or the
Huber loss with threshold ``delta`` (``"huber"``), :math:`\Delta_t` the
first difference along time, and the bar the mean over steps and
components. The two penalties shown are the published ones: they are the
time-varying-parameter intuition that spectral parameters should move
slowly, and they are what keeps the amplitude and frequency paths smooth
enough to read. Four further terms are available through
:class:`~ns_sdn.spectral_decomposition.Penalties` and are off by default:

``phase_curvature``
   Mean squared second difference of :math:`\theta`, a smoother
   frequency path by a different route.
``trend``
   Mean squared first difference of :math:`B_t`, which pushes level
   movement out of the trend head and into the components.
``sparsity``
   A group penalty :math:`\sum_k \sqrt{\overline{(g A)^2_{\cdot,k}}}`
   that drives whole components to silence, for choosing :math:`K`.
``spectrum``
   Mean squared difference of the rFFT power spectra of the emission and
   the target, which rewards the right frequency content independently
   of its timing.

Every part is reported unweighted in the result's ``history`` so a stage
can be read as a trade-off rather than a total.

The schedule
------------

Training is Adam :cite:p:`kingma2015adam` in stages, each a
:class:`~ns_sdn.spectral_decomposition.Stage` with its own learning
rate, step count and, optionally, penalties. The published schedule is

.. list-table::
   :header-rows: 1
   :widths: 14 20 14 18 18

   * - Stage
     - Learning rate
     - Steps
     - :math:`\lambda_A`
     - :math:`\lambda_\omega`
   * - 1
     - :math:`10^{-3}`
     - 1200
     - :math:`2 \times 10^{-4}`
     - :math:`8 \times 10^{-4}`
   * - 2
     - :math:`4 \times 10^{-4}`
     - 1800
     - :math:`3 \times 10^{-4}`
     - :math:`1.5 \times 10^{-3}`

A first stage that fits, a second that smooths. Each step is one
gradient through the whole sequence, so the cost of a fit is the sample
length times three thousand recurrent passes; the published run on 9,186
daily observations takes well under an hour on a laptop CPU. A fit diverging to a
non-finite loss raises :class:`~ns_sdn.exceptions.NumericalError`
rather than returning a result.

The published configuration
---------------------------

The defaults of :class:`~ns_sdn.spectral_decomposition.NSSDN` are the
IEEE-ORSS DGS10 experiments: :math:`K = 16`, :math:`d = 128`, a GRU
transition, positive unordered frequencies, gating, the filtered
alignment, squared error on a series min-max scaled to :math:`[-1, 1]`,
the two-stage schedule above. With no covariates that network has
58,849 trainable parameters, the count of the published run, and the
test suite checks it:

.. testcode::

   import numpy as np
   from ns_sdn.spectral_decomposition import NSSDN

   model = NSSDN(np.random.default_rng(0).standard_normal(50))
   print(model.k, model.state_dim, model.alignment, model.gating, model.ordered_frequency)

.. testoutput::

   16 128 filtered True False

The data of that run are the 10-year Treasury constant-maturity yield
(:fred:`DGS10`) in logs, placed on a daily calendar grid with gaps of up
to seven days filled forward, 9,551 observations from February 2000,
the last 365 held out. The :doc:`reproduction example <../auto_examples/plot_03_reproduce_orss>` runs exactly that protocol through the
package and writes the paper's metrics table and figures; the
:doc:`DGS10 example <../auto_examples/plot_02_dgs10_spectral_state>`
runs a shorter version at documentation-build time.
