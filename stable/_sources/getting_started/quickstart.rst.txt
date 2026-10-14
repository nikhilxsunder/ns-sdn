.. meta::
   :description: A fifteen-minute tour of ns-sdn: fit a spectral decomposition to a series with a known answer, read every path, run the one-step and recursive forecast protocols, compute the spectral state, compare two fits.

Quickstart
==========

This page runs one small analysis from start to finish: build a series
whose decomposition is known, fit the network to it, read the trend,
amplitude, frequency and gate paths back, run the two forecast
protocols, reduce the decomposition to its spectral-state coordinates,
and compare two component counts. Every block runs as written. The data
are simulated so that the page needs no network connection and the true
components are known; the :doc:`examples <../auto_examples/index>`
repeat the steps on FRED data.

Imports name the leaf module. The top-level ``ns_sdn`` namespace holds
modules only and re-exports nothing.

A series with a known decomposition
-----------------------------------

A level that drifts up, a slow oscillation whose frequency rises from
0.10 to 0.19 radians per step, a fast oscillation at 0.55, and a little
noise. The last thirty observations are held out.

.. testcode::

   import numpy as np

   rng = np.random.default_rng(3)
   t = np.arange(300.0)
   slow = np.sin(np.cumsum(0.10 + 0.0003 * t))          # a chirp: frequency drifts upward
   fast = 0.4 * np.sin(0.55 * t)
   level = 2.0 + 0.003 * t
   y = level + slow + fast + 0.05 * rng.standard_normal(300)
   train, test = y[:270], y[270:]

Specify and fit
---------------

A model is constructed with the series and the specification, and
``fit()`` returns the result. Three components and a sixteen-dimensional
state are far below the published configuration (sixteen and 128), and
one stage of eighty Adam steps is a fraction of the published three
thousand; both are chosen so the page runs in seconds. The specification
is immutable and validated at construction:

.. testcode::

   from ns_sdn.spectral_decomposition import NSSDN, Stage

   model = NSSDN(train, k=3, state_dim=16, schedule=(Stage(1e-2, 80),), seed=0)
   print(model.k, model.state_dim, model.alignment, model.transition, len(model.schedule))

   res = model.fit()
   print(type(res).__name__, res.nobs, res.n_params)

.. testoutput::

   3 16 filtered gru 1
   NSSDNResult 269 1155

``nobs`` is the effective sample, one fewer than the series because the
first observation has no one-step prediction; ``n_params`` is the
trainable weight count. ``seed`` makes the initialization reproducible;
without it two fits of the same specification differ.

Read the summary
----------------

``summary()`` returns a :class:`~ns_sdn.summary.SummaryTable`, which is
what a bare result prints:

.. testcode::

   print(res.summary())

.. testoutput::
   :hide:
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

   NSSDN(k=3) Results
   ...
   Model: NSSDN(k=3, d=16, gru, filtered) Log-likelihood: ...
   ...
   component 1 ...
   component 2 ...
   component 3 ...
   ...
   Filtered alignment: ...

.. code-block:: text

                                          NSSDN(k=3) Results
   ================================================================================================
   Model:          NSSDN(k=3, d=16, gru, filtered)  Log-likelihood:                          47.550
   Objective:                                  mse  AIC:                                   2214.901
   Scaling:                                 minmax  BIC:                                   6366.792
   Observations:                               269  HQIC:                                  3882.309
   Parameters:                                1155  One-step RMSE:                         0.202766
   Gating:                                    True  Emission RMSE:                        0.0318907
   ------------------------------------------------------------------------------------------------
                   mean amp   mean freq   period   mean gate      mix   power share
   component 1       0.1031      0.4796     13.1       0.189   +0.241         0.945
   component 2       0.2300      0.8186      7.7       0.365   +0.017         0.021
   component 3       0.1594      0.7729      8.1       0.250   -0.030         0.033
   ================================================================================================
   Filtered alignment: the emission at t conditions on y[t], so emission is a reconstruction of the
   observed series and fittedvalues report the emission at t as the one-step forecast of y[t+1]
   (the published protocol); compare against a random walk before reading them as skill.
   llf is a concentrated Gaussian likelihood at a penalized fit and n_params the nominal weight
   count; criteria compare k within one configuration only.
   Components are identified up to permutation; indices are not stable across refits.

Three things to read off it. *Emission RMSE* is the same-step
reconstruction error and *One-step RMSE* the error of predicting each
observation from the state one step earlier; under the default
``filtered`` alignment they are different quantities, and the first
note says why. The component rows show the state concentrating power in
one component (share 0.945) at a mean frequency of 0.48 rather than
recovering the 0.12-to-0.19 chirp and the 0.55 oscillation separately:
with this little training the trend head has absorbed most of the slow
movement, a behaviour the :doc:`nuances page <../user_guide/nuances>`
discusses and the published schedule is designed around. The last note
is the identifiability caveat: without ``ordered_frequency=True`` the
component indices are arbitrary.

Read the decomposition
----------------------

The decomposition is the point of the model, so the result carries every
path as an array. Under the filtered alignment each path has one row per
observation; the one-step fit has one fewer. Trend plus the sum of the
component contributions is the emission, exactly:

.. testcode::

   print(res.emission.shape, res.trend.shape, res.components.shape, res.fittedvalues.shape)
   print(bool(np.allclose(res.trend + res.components.sum(axis=1), res.emission, atol=1e-5)))

.. testoutput::

   (270,) (270,) (270, 3) (269,)
   True

Per-component paths are ``(rows, k)`` arrays: ``amplitude`` (the gated
amplitude the paper's figures show), ``amplitude_raw``, ``frequency`` in
radians per unit time, ``period`` in time units, ``phase``,
``phase_offset`` and ``gate``; ``mix`` is the ``(k,)`` mixing vector and
``power_share`` each component's share of contribution power:

.. testcode::

   print(res.frequency.mean(axis=0).round(3))
   print(res.period.mean(axis=0).round(1))
   print(res.power_share.round(3))

.. testoutput::
   :options: +ELLIPSIS

   [0.48  0.819 0.773]
   [13.1  7.7  8.1]
   [0.945 0.021 0.033]

The paths come on the original scale of the series. The network trains
on a min-max scaled copy, which is the scale the paper's figures are
drawn in; ``decompose(scaled=True)`` returns the same paths in training
units, and ``decompose(other_series)`` runs the trained network over a
new sample:

.. testcode::

   scaled = res.decompose(scaled=True)
   print(sorted(scaled))
   print(float(scaled["observed"].min()) >= -1.0 - 1e-9, float(scaled["observed"].max()) <= 1.0 + 1e-9)

.. testoutput::

   ['amplitude', 'amplitude_raw', 'components', 'effective_amplitude', 'emission', 'frequency', 'gate', 'observed', 'oscillation', 'phase', 'phase_offset', 'state', 'trend']
   True True

With :mod:`pandas` installed, ``to_pandas()`` lays the observed series,
the emission, the trend and every per-component path out as columns,
one row per path row; pass ``index=`` to attach your dates:

.. testcode::

   frame = res.to_pandas()
   print(frame.shape)
   print(list(frame.columns)[:10])

.. testoutput::

   (270, 22)
   ['observed', 'emission', 'trend', 'oscillation', 'amplitude_1', 'frequency_1', 'phase_1', 'phase_offset_1', 'gate_1', 'component_1']

Forecast two ways
-----------------

The published evaluation is *conditional one-step*: run the trained
network over the training and holdout observations together under
teacher forcing, and take, for each holdout observation, the emission
built from the realized values up to the step before. ``predict`` does
exactly this and returns one value per observation after the first, so
the last thirty rows are the holdout predictions. Read them against the
random walk, which predicts each value by the one before it:

.. testcode::

   one = res.predict(y)[-30:]
   naive = y[269:-1]
   rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
   print(round(rmse(test, one), 4), round(rmse(test, naive), 4))

.. testoutput::
   :options: +ELLIPSIS

   0.2... 0.2...

On this run the two are close, with the random walk slightly ahead: the
network is predicting mostly by persistence. That is the comparison the
summary's first note asks for, and it is the right first question about
any one-step number from a model that conditions on the current
observation.

*Recursive* forecasting feeds each emission back as the next input.
``forecast(steps)`` warms the network up on the fitted sample and rolls
forward; error compounds with the horizon:

.. testcode::

   path = res.forecast(10)
   print(path.shape)

.. testoutput::

   (10,)

Both protocols, and what the difference means, are the subject of the
:doc:`forecasting page <../user_guide/forecasting>`.

The spectral state
------------------

The per-component paths are hard to reason about as a whole once ``k``
is sixteen. ``spectral_state()`` normalizes each row's contribution
powers into a probability measure on the frequency axis and reports its
coordinates: where it sits (centroid), how spread it is (dispersion),
how many components it effectively uses (effective count, participation
ratio), how fast it moves between rows (a Wasserstein velocity, split
into the part from frequencies moving and the part from mass moving),
and how distinguishable the components are (disentanglement):

.. testcode::

   state = res.spectral_state()
   print(state.summary())

.. testoutput::
   :hide:
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

   Spectral state (k=3)
   ...
   centroid ...
   dispersion ...
   effective count ...
   ...

.. code-block:: text

                                 Spectral state (k=3)
   ================================================================================
   Rows:                              270  Disentanglement:                 0.988
   Active threshold:                 0.01  Mean active:                      3.00
   --------------------------------------------------------------------------------
   coordinate                               min / mean / max
   centroid                          0.4829 / 0.4981 / 0.578
   dispersion                    0.04368 / 0.07437 / 0.08533
   effective count                     1.207 / 1.301 / 1.402
   participation ratio                  1.082 / 1.127 / 1.18
   velocity                  4.713e-05 / 0.001083 / 0.004251
   velocity (frequency)     1.171e-05 / 0.0004414 / 0.002188
   velocity (mass)           1.417e-05 / 0.0006716 / 0.00206
   ================================================================================
   Diagnostics of a model trained on the prediction criterion alone; nothing here
   enters a loss.
   Identifiability degrades as frequencies converge: compression is also the regime
   of weakest disentanglement, and per-component readings of inactive rows are not
   meaningful.

An effective count of 1.3 out of 3 says the same thing the power shares
did: one component carries the oscillation. Every coordinate is also an
array on the state, one entry per row (velocities one fewer, since they
sit between rows):

.. testcode::

   print(state.weight.shape, state.velocity.shape)
   print(round(float(state.effective_count.mean()), 2), round(state.disentanglement_score, 3))

.. testoutput::
   :options: +ELLIPSIS

   (270, 3) (269,)
   1.3 0.9...

The :doc:`spectral state page <../user_guide/spectral_state>` defines
each coordinate and says what a regime-change claim looks like in them.

Compare two fits
----------------

Results fitted to the same series compare directly on an information
criterion. ``compare`` ranks them and reports each fit's distance from
the best:

.. testcode::

   alt = NSSDN(train, k=2, state_dim=16, schedule=(Stage(1e-2, 80),), seed=0).fit()
   print(res.compare(alt))

.. testoutput::
   :hide:
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

   Model comparison by AIC
   ...
   model llf k AIC dAIC
   NSSDN(k=2, d=16, gru, filtered) ... 1085 ... 0.000
   NSSDN(k=3, d=16, gru, filtered) ... 1155 ...
   ...

.. code-block:: text

                               Model comparison by AIC
   ================================================================================
   --------------------------------------------------------------------------------
   model                                  llf      k        AIC      dAIC
   NSSDN(k=2, d=16, gru, filtered)     46.792   1085   2076.416     0.000
   NSSDN(k=3, d=16, gru, filtered)     47.550   1155   2214.901   138.485
   ================================================================================
   Criteria are comparable only across fits sharing an engine configuration.

The criteria charge the nominal weight count, so they favour fewer
components strongly; the closing note and the second note of the result
summary say when the comparison is meaningful. ``criterion="bic"`` or
``"hqic"`` ranks by another criterion.

Save and reload
---------------

A result pickles through the network's state dict, so it round-trips
without a reference to a live torch module and forecasts identically
after reloading:

.. testcode::

   import pickle

   restored = pickle.loads(pickle.dumps(res))
   print(bool(np.allclose(restored.forecast(5), res.forecast(5))))

.. testoutput::

   True

What a refusal looks like
-------------------------

Every argument is validated at construction or at the call, and the
error names the argument and the offending value:

.. testcode::

   from ns_sdn.exceptions import SpecificationError

   try:
       NSSDN(train, k=0)
   except SpecificationError as error:
       print(error)
   try:
       res.forecast(0)
   except SpecificationError as error:
       print(error)

.. testoutput::

   k must be at least 1; got 0.
   steps must be a positive integer; got 0.

Where next
----------

The :doc:`user guide <../user_guide/index>` derives the model and
explains how to read each path; :doc:`../user_guide/nuances` lists what
behaves differently from what a reader of the papers expects; the
:doc:`examples <../auto_examples/index>` run the published DGS10
analysis.
