.. meta::
   :description: The spectral state of an NS-SDN fit: the decomposition as a time-local measure on the frequency axis, its centroid, dispersion, entropy and effective count, Wasserstein velocity split into frequency and mass motion, the active set, and the disentanglement score.

The spectral state
==================

Sixteen amplitude paths and sixteen frequency paths are a lot to read,
and a claim such as "the spectrum compressed in 2020" is a claim about
all of them at once. :class:`~ns_sdn.spectral_state.SpectralState`
reduces the decomposition at each step to a probability measure on the
frequency axis and reports the coordinates of that measure: where it
sits, how spread it is, how many components it effectively uses, how
fast it moves, and how well the components can be told apart. The
coordinates are diagnostics of a model trained on the prediction
criterion alone; nothing here enters a loss.

The measure
-----------

At row :math:`t` component :math:`k` contributes with amplitude
:math:`a_{t,k} = |w_k|\, g_{t,k} A_{t,k}` (``effective_amplitude``) at
frequency :math:`\omega_{t,k}`. Its power is :math:`p_{t,k} =
a_{t,k}^2`, and normalizing gives a discrete measure on the frequency
axis,

.. math::

   \mu_t = \sum_{k=1}^{K} \pi_{t,k}\, \delta_{\omega_{t,k}},
   \qquad
   \pi_{t,k} = \frac{p_{t,k}}{\sum_j p_{t,j}}.

``weight`` is :math:`\pi`, shape ``(r, k)``; ``frequency`` is the
support. Every coordinate below is a functional of :math:`\mu_t`.

Position and spread
-------------------

.. math::

   \bar\omega_t = \sum_k \pi_{t,k}\, \omega_{t,k},
   \qquad
   s_t = \Bigl(\sum_k \pi_{t,k} (\omega_{t,k} - \bar\omega_t)^2\Bigr)^{1/2}.

``centroid`` is the power-weighted mean frequency, ``dispersion`` the
power-weighted standard deviation. A falling centroid is the spectrum
slowing down; a falling dispersion is the components converging in
frequency.

Effective count
---------------

.. math::

   H_t = -\sum_k \pi_{t,k} \log \pi_{t,k},
   \qquad
   m_t = e^{H_t},
   \qquad
   \mathrm{IPR}_t = \Bigl(\sum_k \pi_{t,k}^2\Bigr)^{-1}.

``entropy`` is the spectral entropy, ``effective_count`` its exponential
(between 1 and :math:`K`: the number of equally weighted components
that would have the same entropy), and ``participation_ratio`` the
inverse participation ratio, a second effective count that weights the
large components more. Both fall when power concentrates in fewer
components.

Compression, in these terms, is a *joint* contraction of ``dispersion``
and ``effective_count``: the components move together in frequency and
fewer of them carry the power. Either alone is a different story, and
the IEEE-ORSS reading of 2020, frequency convergence with amplitude
concentration, is the first.

Velocity
--------

How fast the measure moves between consecutive rows is the
1-Wasserstein distance :cite:p:`villani2009ot` per unit time,

.. math::

   \mathcal{V}_t = \frac{W_1(\mu_t, \mu_{t-1})}{\Delta t_t},
   \qquad
   W_1(\mu, \nu) = \int \bigl|F_\mu(\omega) - F_\nu(\omega)\bigr|\, d\omega,

computed in closed form from the two cumulative distribution functions.
``velocity`` has one entry fewer than the rows, since it sits between
them. Two decompositions of the same step distinguish *why* the measure
moved: ``velocity_frequency`` is :math:`W_1` between the measures at
:math:`t` and :math:`t - 1` with the weights frozen at :math:`t - 1`
(the components moved), and ``velocity_mass`` with the frequencies
frozen at :math:`t - 1` (the power moved between components). They do
not add exactly to ``velocity``, but their relative size says whether a
spike in it was a shift in frequencies or a reallocation of power.

Active set and disentanglement
------------------------------

A component whose weight is tiny is unidentified: its frequency, phase
and offset are not pinned down by the data because it contributes
nothing to the emission. ``active`` marks, per row and component,
whether :math:`\pi_{t,k}` is at least ``active_threshold`` (default
0.01), and per-component readings should be restricted to active rows.

``contribution_correlation`` is the :math:`K \times K` correlation
matrix of the ``components`` paths over the sample, and
``disentanglement_score`` is one minus the off-diagonal share of its
squared entries, in :math:`[0, 1]`: one when the contributions are
mutually uncorrelated, falling as components come to carry the same
movement. The measure alone cannot tell converged frequencies from
redundant components, and this score is the check: identifiability
degrades exactly as :math:`\omega_k \to \omega_l`, so the compression
regime is also the regime in which per-component readings are least
trustworthy. State that caveat with any compression claim rather than
beside it.

Computing it
------------

``spectral_state()`` on a result, or
:meth:`SpectralState.from_result <ns_sdn.spectral_state.SpectralState.from_result>`:

.. testcode::

   import numpy as np
   from ns_sdn.spectral_decomposition import NSSDN, Stage

   t = np.arange(240.0)
   y = np.sin(0.3 * t) + 0.5 * np.sin(0.07 * t)
   res = NSSDN(y, k=3, state_dim=8, schedule=(Stage(1e-2, 40),), seed=0).fit()
   state = res.spectral_state()

   print(state.nobs, state.k)
   print(state.weight.shape, state.centroid.shape, state.velocity.shape, state.active.shape)
   print(bool(np.allclose(state.weight.sum(axis=1), 1.0)))
   print(bool(np.all(state.effective_count >= 1.0)), bool(np.all(state.effective_count <= 3.0)))
   print(0.0 <= state.disentanglement_score <= 1.0)

.. testoutput::

   240 3
   (240, 3) (240,) (239,) (240, 3)
   True
   True True
   True

``summary()`` prints the min, mean and max of each scalar coordinate
with the disentanglement score and the mean number of active components
in the header; ``to_dict()`` and ``to_pandas(index=...)`` lay the scalar
coordinates out one row per path row, velocities padded with ``nan`` in
the first row so the frames align:

.. testcode::

   table = state.to_dict()
   print(sorted(table))
   print(np.isnan(table["velocity"][0]), table["velocity"].shape)

.. testoutput::

   ['centroid', 'dispersion', 'effective_count', 'entropy', 'n_active', 'participation_ratio', 'velocity', 'velocity_frequency', 'velocity_mass']
   True (240,)

Testing a regime claim
----------------------

A regime-change claim in these coordinates is a before-and-after
comparison. Mark the event on the result's ``time``, and compare the
coordinates over a window on each side:

.. testcode::

   event = 160
   before = slice(event - 40, event)
   after = slice(event, event + 40)
   for name in ("centroid", "dispersion", "effective_count"):
       values = getattr(state, name)
       print(name, values[before].shape, values[after].shape)

.. testoutput::

   centroid (40,) (40,)
   dispersion (40,) (40,)
   effective_count (40,) (40,)

What makes the comparison evidence is that it holds across refits and
across ``k``: the coordinates are invariant to the permutation of
components, which the raw paths are not, so a seed sweep over the same
specification is the natural check. The
:doc:`DGS10 example <../auto_examples/plot_02_dgs10_spectral_state>`
draws the coordinates with the 2020 window marked.
