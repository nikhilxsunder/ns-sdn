.. meta::
   :description: Reading an NS-SDN decomposition: the paths on the result, how they add up to the emission, the original and training scales, running the network over a new sample, and the dataframe layout.

The decomposition
=================

The decomposition is the product of the model, so the result carries
every path the paper's figures are drawn from, as arrays. This page
lists them, shows how they combine, and says which scale each is on.
The :doc:`model page <model>` defines the quantities; here they are
read.

Throughout, :math:`r` is the number of path rows: the sample length
:math:`n` under the filtered alignment, :math:`n - 1` under the
predicted one. ``observed`` is the slice of the series the rows align
with.

The paths
---------

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Attribute
     - Shape
     - Meaning
   * - ``observed``
     - ``(r,)``
     - The series rows the paths describe.
   * - ``emission``
     - ``(r,)``
     - :math:`\hat y_t`, the synthesized series (Fig. 1 of IEEE-ORSS).
   * - ``trend``
     - ``(r,)``
     - :math:`B_t` (Fig. 3).
   * - ``oscillation``
     - ``(r,)``
     - ``emission - trend``, everything the components contribute.
   * - ``amplitude_raw``
     - ``(r, k)``
     - :math:`A_{t,k}`, the softplus amplitude before the gate.
   * - ``amplitude``
     - ``(r, k)``
     - :math:`g_{t,k} A_{t,k}`, the gated amplitude (Fig. 4).
   * - ``effective_amplitude``
     - ``(r, k)``
     - :math:`|w_k|\, g_{t,k} A_{t,k}`, the amplitude of each contribution.
   * - ``frequency``
     - ``(r, k)``
     - :math:`\omega_{t,k}`, radians per unit time (Fig. 5).
   * - ``period``
     - ``(r, k)``
     - :math:`2\pi / \omega_{t,k}`, in time units.
   * - ``phase``
     - ``(r, k)``
     - :math:`\theta_{t,k}`, the cumulative phase.
   * - ``phase_offset``
     - ``(r, k)``
     - :math:`\phi_{t,k}`.
   * - ``gate``
     - ``(r, k)``
     - :math:`g_{t,k}` (Fig. 6); ones when ``gating=False``.
   * - ``components``
     - ``(r, k)``
     - :math:`w_k\, g_{t,k} A_{t,k} \sin(\theta_{t,k} + \phi_{t,k})`, each summand.
   * - ``state``
     - ``(r, d)``
     - :math:`h_t`.
   * - ``mix``
     - ``(k,)``
     - :math:`w`, the mixing vector.
   * - ``power_share``
     - ``(k,)``
     - Each component's share of mean contribution power over the sample.

Fields are arrays; ``period``, ``effective_amplitude``, ``components``,
``oscillation``, ``power_share`` and ``observed`` are properties computed
from them, so they cost a little each call and are always consistent
with the fields.

How they add up
---------------

The synthesis equation is an identity on the result:

.. testcode::

   import numpy as np
   from ns_sdn.spectral_decomposition import NSSDN, Stage

   t = np.arange(240.0)
   y = 1.0 + 0.5 * np.sin(0.3 * t) + 0.3 * np.sin(0.08 * t)
   res = NSSDN(y, k=2, state_dim=8, schedule=(Stage(1e-2, 40),), seed=0).fit()

   summands = res.mix * res.amplitude * np.sin(res.phase + res.phase_offset)
   print(bool(np.allclose(summands, res.components)))
   print(bool(np.allclose(res.trend + res.components.sum(axis=1), res.emission, atol=1e-5)))
   print(bool(np.allclose(res.oscillation, res.components.sum(axis=1), atol=1e-5)))
   print(bool(np.allclose(np.abs(res.mix) * res.amplitude, res.effective_amplitude)))

.. testoutput::

   True
   True
   True
   True

Reading a single component means reading the row of ``components`` it
contributes, the ``amplitude`` that bounds it, the ``frequency`` that
paces it, and the ``gate`` that switches it. A component whose gate sits
near zero over a stretch is silent there whatever its raw amplitude
says, which is why the figures plot the gated amplitude.

Two scales
----------

The network trains on a copy of the series min-max scaled to
:math:`[-1, 1]` on the fitted sample (``scaling="minmax"``; ``"none"``
trains on the raw series). Every array on the result is mapped back to
the original scale: ``emission``, ``trend`` and ``observed`` through the
full affine map, the amplitudes and components through its slope alone,
since they are differences rather than levels. ``frequency``, ``phase``,
``phase_offset``, ``gate`` and ``state`` are scale-free.

The paper's figures are drawn in training units. ``decompose(scaled=True)``
returns the same paths without the inverse map, and the ``scale`` field
holds the map itself:

.. testcode::

   paths = res.decompose(scaled=True)
   print(paths["trend"].shape, bool(np.all(np.abs(paths["observed"]) <= 1.0 + 1e-9)))
   print(bool(np.allclose(res.scale.inverse(paths["emission"]), res.emission, atol=1e-6)))

.. testoutput::

   (240,) True
   True

A new sample
------------

``decompose(endog)`` runs the trained network over another series under
teacher forcing from the learned initial state and returns its paths,
on the original scale of the *fitted* series' map. This is how the
paper's figures are drawn over the full sample after fitting on the
training segment, and how a decomposition is read off out-of-sample
data without refitting:

.. testcode::

   longer = np.concatenate([y, 1.0 + 0.5 * np.sin(0.3 * np.arange(240.0, 300.0))])
   paths = res.decompose(longer)
   print(paths["emission"].shape, paths["components"].shape)

.. testoutput::

   (300,) (300, 2)

A fit with covariates needs ``exog`` again here, with one row per
observation of the new series, and ``time`` takes the new grid.

As a dataframe
--------------

``to_dict()`` returns the observed series, the emission, the trend, the
oscillation and every per-component path as a flat mapping with
one-based component suffixes; ``to_pandas(index=...)`` is the same
mapping as a frame, with your dates attached if you pass them:

.. testcode::

   columns = list(res.to_dict())
   print(len(columns), columns[:4], columns[4:10])

.. testoutput::

   16 ['observed', 'emission', 'trend', 'oscillation'] ['amplitude_1', 'frequency_1', 'phase_1', 'phase_offset_1', 'gate_1', 'component_1']

The flat layout is for plotting and export. For analysis, the ``(r, k)``
arrays on the result are the better handle, and the
:doc:`spectral state <spectral_state>` is the better summary.

Reading the figures
-------------------

The IEEE-ORSS figures correspond to the paths as follows, each drawn in
training units over the full DGS10 sample with the holdout boundary
marked:

1. *Forecast versus actual*: ``observed`` and ``emission`` (training
   segment) with ``predict`` over the holdout.
2. *Squared error*: ``resid ** 2`` from the one-step fit.
3. *Learned trend*: ``trend``.
4. *Amplitude paths*: columns of ``amplitude`` for a few components.
5. *Instantaneous frequency paths*: columns of ``frequency``.
6. *Gating paths*: columns of ``gate``.

The :doc:`DGS10 example <../auto_examples/plot_02_dgs10_spectral_state>`
draws them, and the :doc:`reproduction example <../auto_examples/plot_03_reproduce_orss>` writes the full set.
