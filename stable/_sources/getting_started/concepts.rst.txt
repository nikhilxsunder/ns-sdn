.. meta::
   :description: The design ns-sdn shares with cultivars: a model holds a specification, fit returns a frozen result record, summaries state their assumptions, the two alignments, the staged schedule, and the error hierarchy.

Core concepts
=============

ns-sdn follows the conventions of :mod:`cultivars`, so that a workflow
written against one library reads the other and so that cultivars can
expose the model under its experimental namespace without adapters. This
page states them once, so that the API reference can assume them.

Model, fit, result
------------------

A model object holds the series and the specification and nothing else.
``fit()`` trains the network and returns a *result*:

.. testcode::

   import numpy as np
   from ns_sdn.spectral_decomposition import NSSDN, Stage

   y = np.sin(0.2 * np.arange(160.0)) + 0.01 * np.arange(160.0)
   model = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 30),), seed=0)   # specification
   result = model.fit()                                                       # estimation
   print(type(result).__name__)

.. testoutput::

   NSSDNResult

Everything that describes *what model* goes to the constructor: the
number of components, the state dimension, the transition, the heads'
constraints, the alignment, the objective, the penalties, the schedule,
the scaling, the seed. There is nothing to pass to ``fit()``. One
specification can be fitted again by calling ``fit()`` again; with a
seed the two fits agree.

Results are records
-------------------

A result is a frozen dataclass. It carries the decomposition, the
one-step fit, the trained network behind a numpy interface, and the
training history, all as attributes, and it cannot be changed after it
is built:

.. testcode::

   print(result.nobs, result.k, result.trend.shape, result.amplitude.shape)
   try:
       result.k = 3
   except AttributeError as error:
       print(type(error).__name__)

.. testoutput::

   159 2 (160,) (160, 2)
   FrozenInstanceError

Because results are values rather than handles to a model, anything that
reads a result's attributes can take any result that has them: a
:class:`~ns_sdn.spectral_state.SpectralState` is built from any result
with ``frequency``, ``effective_amplitude``, ``components`` and ``time``;
``compare`` takes any results with ``llf``, ``nobs`` and ``n_params``.
To change a fit, build a new model and fit it; the old result stays
exactly as it was. Results pickle, through the network's state dict.

Summaries state their assumptions
---------------------------------

``summary()`` returns a :class:`~ns_sdn.summary.SummaryTable`, a record
with four parts:

- ``title``: the specification;
- ``metadata``: label/value pairs such as sample size, log-likelihood,
  information criteria and the two RMSEs;
- ``rows``: one table with a row per component;
- ``notes``: what the numbers assume and what they are not.

It renders as text (``to_text()``, which is also ``str(result)``), as
HTML in a notebook, and as a dataframe (``to_pandas()``). The notes are
part of the result, not decoration: the first says which alignment the
fit used and what that makes of its one-step numbers, the second that
the likelihood is concentrated at a penalized fit, the third, when it
appears, that the components are identified only up to permutation.

Two alignments
--------------

The network reads the series through a GRU, and the question is whether
the emission at step :math:`t` is read from the state *after* the
network has seen :math:`y_t` or *before*.

``alignment="filtered"`` (the default, and the published run)
   The state at :math:`t` has consumed :math:`y_t`. The emission at
   :math:`t` is a *reconstruction* of :math:`y_t`, and the decomposition
   is a filtered reading of the observed series. The one-step forecast of
   :math:`y_{t+1}` is the emission at :math:`t`, which is the paper's
   holdout protocol. Every path has one row per observation.

``alignment="predicted"``
   The state at :math:`t` has consumed only :math:`y_{t-1}`. Every path is
   a one-step-ahead prediction and the network trains as a forecaster.
   Every path has one row fewer than the series, because the first
   observation has nothing before it.

Under both, ``fittedvalues`` has ``n - 1`` rows aligned with
``endog[1:]`` and is the one-step conditional prediction of each
observation, so one-step diagnostics (``resid``, ``sigma2``, ``llf``,
``r_squared``) mean the same thing whichever alignment produced them.
What differs is the decomposition's meaning: a description of the series
versus a prediction of it. The
:doc:`forecasting page <../user_guide/forecasting>` works through the
consequences.

The schedule
------------

Training is staged. A :class:`~ns_sdn.spectral_decomposition.Stage` is a
learning rate, a number of Adam steps, and optionally its own
:class:`~ns_sdn.spectral_decomposition.Penalties`; the model's
``penalties`` argument supplies the penalties for stages that carry
none. The default schedule is the published two stages:

.. testcode::

   from ns_sdn.spectral_decomposition import Penalties

   schedule = (
       Stage(1e-3, 1200, Penalties(amplitude=2e-4, frequency=8e-4)),
       Stage(4e-4, 1800, Penalties(amplitude=3e-4, frequency=1.5e-3)),
   )
   print(sum(stage.steps for stage in schedule))

.. testoutput::

   3000

The two published penalties discourage step-to-step changes in the
gated amplitudes and in the frequencies, which is what keeps the
components slow enough to read. The four further penalties
(``phase_curvature``, ``trend``, ``sparsity``, ``spectrum``) are off by
default and exist for the experiments the
:doc:`model page <../user_guide/model>` describes. The result's
``history`` records the loss and its parts by stage and step.

Data conventions
----------------

Inputs are anything :func:`numpy.asarray` accepts. The series is
one-dimensional with at least three finite observations; an optional
``exog`` block is ``(n, p)`` and is used as given, without scaling; an
optional ``time`` grid is strictly increasing and defaults to the
integers, and its spacing enters the phase recursion. Results carry
integer positions and leave your index alone: ``to_pandas(index=dates)``
attaches dates on the way out. The series is min-max scaled to
:math:`[-1, 1]` on the sample before training (``scaling="minmax"``), and
every path is mapped back to the original scale unless asked for in
training units. The :doc:`conventions page <../user_guide/conventions>`
lists every shape.

Refusals, not silent failure
----------------------------

Three exceptions, one per kind of fix, all under
:class:`~ns_sdn.exceptions.NSSDNError`:

:class:`~ns_sdn.exceptions.DimensionError`
   The geometry of an input is wrong: a two-dimensional series, an
   ``exog`` with the wrong number of rows, a forecast without the
   covariates the fit used. Reshape the data.

:class:`~ns_sdn.exceptions.SpecificationError`
   The content of a choice is wrong: ``k=0``, an unknown transition, a
   negative penalty, a threshold outside the unit interval. Change the
   argument.

:class:`~ns_sdn.exceptions.NumericalError`
   The arithmetic could not be carried out: non-finite data, a diverged
   fit. Look at the numbers.

Each also derives from the matching builtin (``ValueError`` for the first
two, ``RuntimeError`` for the third), and the hierarchy mirrors
:mod:`cultivars.exceptions`, so one ``except`` clause handles both
libraries.
