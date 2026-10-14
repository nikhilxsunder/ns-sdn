.. meta::
   :description: Forecasting with NS-SDN: the one-step conditional protocol of the papers, recursive multi-step forecasts, what the filtered and predicted alignments make of each, the random-walk benchmark, and the fit statistics.

Forecasting
===========

The model produces two kinds of forecast and trains under two
alignments, and the four combinations do not mean the same thing. This
page states each protocol as the package implements it, says which one
the published numbers are, and gives the benchmark every one-step
number should be read against.

One-step conditional
--------------------

The published evaluation runs the trained network over the training and
holdout observations together under teacher forcing, every input a
realized value, and reads the emission at each step as the prediction
of the next. ``predict`` is that protocol: for a series of length
:math:`m` it returns :math:`m - 1` values aligned with ``endog[1:]``,
each built from realized values through the step before. The published
holdout numbers are the last ``len(holdout)`` of them:

.. testcode::

   import numpy as np
   from ns_sdn.spectral_decomposition import NSSDN, Stage

   rng = np.random.default_rng(1)
   t = np.arange(260.0)
   y = 0.002 * t + np.sin(0.2 * t) + 0.1 * rng.standard_normal(260)
   train, holdout = y[:230], y[230:]

   res = NSSDN(train, k=2, state_dim=8, schedule=(Stage(1e-2, 40),), seed=0).fit()
   one_step = res.predict(y)[-len(holdout):]
   print(one_step.shape)

.. testoutput::

   (30,)

On the fitted sample itself ``predict`` reproduces ``fittedvalues``, so
the in-sample one-step statistics on the result (``resid``, ``sigma2``,
``llf``, ``r_squared``) are the same protocol applied to the training
segment:

.. testcode::

   print(bool(np.allclose(res.predict(train), res.fittedvalues)))
   print(bool(np.allclose(res.resid, train[1:] - res.fittedvalues)))

.. testoutput::

   True
   True

The random walk
---------------

A one-step forecast of a persistent series looks good by construction:
predicting each value by the one before it is already hard to beat, and
the network conditions on that value. The first question about any
one-step number is therefore how it compares with the no-change
forecast, and the summary's first note says so. Compute both:

.. testcode::

   def rmse(actual, predicted):
       return float(np.sqrt(np.mean((predicted - actual) ** 2)))

   naive = y[len(train) - 1 : -1]              # y[t-1] as the forecast of y[t]
   print(round(rmse(holdout, one_step), 2), round(rmse(holdout, naive), 2))

.. testoutput::
   :options: +ELLIPSIS

   0.2... 0.2...

On this short fit the random walk wins outright; on the published DGS10
run the one-step errors are of the same order as the daily changes of
the yield. The :doc:`reproduction example <../auto_examples/plot_03_reproduce_orss>` adds the random-walk row to
the paper's metrics table for exactly this reason. Skill over
persistence is a claim to establish, not a property of the protocol.

Recursive
---------

``forecast(steps)`` warms the network up on the fitted sample under
teacher forcing, feeds the last observation as the first future input,
and then feeds each emission back as the next input. No realized value
enters after the first step, so error compounds with the horizon; the
validation notebook behind the papers found this mode far weaker than
the conditional one, and the IEEE-ORSS paper evaluates the conditional
protocol for that reason.

.. testcode::

   path = res.forecast(12)
   print(path.shape, bool(np.all(np.isfinite(path))))

.. testoutput::

   (12,) True

A fit with covariates needs ``exog`` of shape ``(steps, p)`` here, and
an irregular grid continues through ``time``, a ``(steps,)`` array of
future times after the sample's last; without it the sample's last step
size is repeated.

Filtered and predicted
----------------------

The alignment decides what the emission at :math:`t` has seen, and
therefore what each protocol measures.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - ``alignment="filtered"`` (default, published)
     - ``alignment="predicted"``
   * - State at :math:`t` has consumed
     - :math:`y_t`
     - :math:`y_{t-1}`
   * - ``emission[t]`` is
     - a reconstruction of :math:`y_t`
     - a prediction of :math:`y_t`
   * - Path rows
     - :math:`n`
     - :math:`n - 1`
   * - ``fittedvalues[i]`` is
     - ``emission[i]``, read as the forecast of :math:`y_{i+1}`
     - ``emission[i]``, the prediction of :math:`y_{i+1}`
   * - Training target of the emission at :math:`t`
     - :math:`y_t` (same step)
     - :math:`y_t` (next step after the input)
   * - Decomposition describes
     - the observed series, filtered
     - the one-step prediction of it

Under the filtered alignment the network is trained to reproduce the
observation it has just seen, so ``emission`` fits the series closely
(the *Emission RMSE* of the summary) while ``fittedvalues`` carries the
same emission forward by one step and is judged against the next
observation (the *One-step RMSE*). The gap between the two RMSEs is the
one-step change in the series the decomposition did not anticipate.
Under the predicted alignment the two coincide, because the emission is
the one-step prediction, and the network trains as a forecaster:

.. testcode::

   fwd = NSSDN(train, k=2, state_dim=8, alignment="predicted",
               schedule=(Stage(1e-2, 40),), seed=0).fit()
   print(fwd.emission.shape, fwd.fittedvalues.shape, fwd.observed.shape)
   print(bool(np.allclose(fwd.emission, fwd.fittedvalues)))

.. testoutput::

   (229,) (229,) (229,)
   True

Which to use. The filtered alignment is the published model and the
right one for reading the decomposition of an observed series, which is
what the IEEE-ORSS paper does; its one-step numbers are a by-product and
should be benchmarked. The predicted alignment is the right one when the
forecast is the object, with the caveat on the
:doc:`nuances page <nuances>` that a GRU with a linear trend head trained
to predict a persistent series learns persistence first and oscillation
later, if at all, under short schedules.

Fit statistics
--------------

The one-step residuals carry the fit statistics, under both alignments:

``sigma2``
   Mean squared one-step residual; its root is the summary's *One-step
   RMSE*.
``llf``
   The concentrated Gaussian log-likelihood
   :math:`-\tfrac{n}{2}\bigl(\log 2\pi\hat\sigma^2 + 1\bigr)` of the
   one-step residuals, with :math:`n` the effective sample.
``r_squared``
   One minus the residual share of the variance of ``endog[1:]``.
``information_criteria``, ``aic``, ``bic``, ``hqic``
   From ``llf``, ``nobs`` and ``n_params``.

``compare(*others, criterion=...)`` ranks results fitted to the same
series. Read it with the summary's second note in mind: the likelihood
is concentrated at a penalized fit and ``n_params`` is the nominal
weight count, which overstates the freedom of a weight-decayed network,
so the criteria discriminate between values of ``k`` within one engine
configuration and not across configurations. The metrics of the papers
(RMSE, MAE, MAPE on the holdout, on the scaled, log and level series)
are computed from ``predict`` by the :doc:`reproduction example <../auto_examples/plot_03_reproduce_orss>`.
