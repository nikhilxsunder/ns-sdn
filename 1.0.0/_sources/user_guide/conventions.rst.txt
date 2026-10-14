.. meta::
   :description: ns-sdn data conventions: input shapes and validation, the time grid, covariates, scaling, how returned arrays are indexed, seeds and determinism, devices, pickling, and the training history.

Conventions
===========

The rules every input and output obeys, stated once. The
:doc:`decomposition page <decomposition>` lists the output arrays; this
page is about the edges of the library.

Inputs
------

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - Argument
     - Shape
     - Rule
   * - ``endog``
     - ``(n,)``
     - Any array-like :func:`numpy.asarray` accepts, converted to
       ``float64``. One-dimensional (an ``(n, 1)`` column is accepted
       and flattened), at least three observations, all finite. Missing
       data is not supported: fill or drop before fitting, and say so in
       the write-up.
   * - ``exog``
     - ``(n, p)`` or ``(n,)``
     - Covariates known at each step, used as given with no scaling. A
       one-dimensional block is read as one column. Must have ``n``
       rows and be finite. A fit with covariates needs them again in
       ``decompose``, ``predict`` and ``forecast``.
   * - ``time``
     - ``(n,)``
     - Strictly increasing, finite; defaults to ``0, 1, ..., n - 1``. The
       differences enter the phase recursion (and, under
       ``transition="timeaware"``, the state transition), so a daily
       series on a calendar grid and one on a business-day grid are
       different models with different frequency units.

A violation raises at construction: the wrong rank or length a
:class:`~ns_sdn.exceptions.DimensionError`, a non-increasing grid or an
option outside its set a :class:`~ns_sdn.exceptions.SpecificationError`,
non-finite values a :class:`~ns_sdn.exceptions.NumericalError`.

.. testcode::

   import numpy as np
   from ns_sdn.exceptions import DimensionError, NumericalError, SpecificationError
   from ns_sdn.spectral_decomposition import NSSDN

   y = np.sin(0.2 * np.arange(60.0))
   for bad in (
       dict(endog=np.c_[y, y]),
       dict(endog=np.r_[y[:-1], np.nan]),
       dict(endog=y, time=np.zeros(60)),
       dict(endog=y, exog=np.ones((59, 1))),
   ):
       try:
           NSSDN(**bad)
       except (DimensionError, NumericalError, SpecificationError) as error:
           print(type(error).__name__)

.. testoutput::

   DimensionError
   NumericalError
   SpecificationError
   DimensionError

Options
-------

Every categorical option is a PEP 695 alias over a ``Literal`` in
:mod:`ns_sdn.typing`, which is both the annotation the type checker
reads and the set the runtime validator checks against:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Alias
     - Values
     - Argument
   * - ``Alignment``
     - ``"filtered"``, ``"predicted"``
     - ``alignment``
   * - ``Transition``
     - ``"gru"``, ``"timeaware"``
     - ``transition``
   * - ``Objective``
     - ``"mse"``, ``"huber"``
     - ``objective``
   * - ``Scaling``
     - ``"minmax"``, ``"none"``
     - ``scaling``

Integers (``k``, ``state_dim``, ``steps``) must be ``int`` and at least
one; a ``bool`` is rejected rather than read as an integer. Penalty
weights are non-negative finite floats; ``active_threshold`` lies in
:math:`[0, 1]`.

Outputs
-------

Results carry integer positions and leave your index alone. Paths have
:math:`r` rows aligned with ``observed``; ``fittedvalues`` and ``resid``
have :math:`n - 1` rows aligned with ``endog[1:]``; ``predict`` returns
one value per observation after the first of the series it is given;
``forecast`` returns one per requested step. Attach dates on the way
out:

.. code-block:: python

   frame = res.to_pandas(index=dates[-res.observed.shape[0]:])
   coords = res.spectral_state().to_pandas(index=dates[-res.observed.shape[0]:])

Every array on a result is a fresh ``float64`` array on the original
scale of the series; ``decompose(scaled=True)`` is the only way to get
training units. Arrays are not views into the network: modifying one
changes nothing else.

Scaling
-------

``scaling="minmax"`` maps the fitted sample onto :math:`[-1, 1]` by the
affine map stored in ``scale`` (``low``, ``high``, ``target_low``,
``target_high``) and uses that same map for every later call, so a new
sample that exceeds the fitted range is simply mapped outside
:math:`[-1, 1]`. ``scaling="none"`` fits the raw series; use it only for
series already of unit order, since softplus heads and Adam learning
rates are tuned to the scaled range.

Reproducibility
---------------

``seed`` controls the initialization of the network and the learned
initial state and phase. With a seed, two fits of the same
specification on the same machine agree to floating-point precision;
without one they differ. Across machines, BLAS backends and torch
versions, the same seed gives the same architecture but not bit-equal
weights after training, so report numbers with a tolerance and sweep
seeds when a claim rests on a particular component. The published run
used ``seed=1337``.

Devices
-------

``device`` is any torch device string. The network is small and
training is one full-sequence recurrent pass per step, so the CPU is
the right device for the published size; a GPU helps only for much
larger ``state_dim`` or many fits in parallel. Results always come back
as numpy arrays on the host, whatever the device.

Pickling
--------

A result pickles through the network's ``state_dict`` as numpy arrays,
plus the configuration needed to rebuild the module and the scaling
map. It reloads on a machine with torch installed, without the original
module object, and forecasts identically. The training ``history`` and
every path travel with it.

History
-------

``history`` is a tuple of records, one at the first step, one every
hundred steps after it, and one at the last step of each stage, each
with the stage index, the step, the total weighted loss and every
unweighted loss part by name:

.. testcode::

   from ns_sdn.spectral_decomposition import Stage

   res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 120),), seed=0).fit()
   print([(record.stage, record.step) for record in res.history])
   print(sorted(res.history[-1].parts))

.. testoutput::

   [(0, 0), (0, 100), (0, 119)]
   ['amplitude', 'frequency', 'prediction']

A stage with only the prediction part is one whose penalties are all
zero; a part present but tiny is a penalty that bound nothing.
