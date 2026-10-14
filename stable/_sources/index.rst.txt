.. meta::
   :description: ns-sdn is the Non-Stationary Spectral Decomposition Network for Python: a neural state-space model that emits an interpretable time-varying spectral decomposition of economic and financial time series, on numpy and torch.

ns-sdn
======

**A neural state-space model that decomposes a non-stationary series into
a trend and a few sinusoids whose amplitude, frequency and participation
move with a latent state.** Fit it to one series, read the decomposition
back as arrays, forecast from it, and summarize the whole spectral
picture in a handful of coordinates. One model, one ``fit()``, one frozen
result, in the same shape as :mod:`cultivars`.

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting_started/index
      :link-type: doc
      :text-align: center

      Install, fit a first decomposition, read its summary and its paths.

   .. grid-item-card:: User guide
      :link: user_guide/index
      :link-type: doc
      :text-align: center

      The model in its equations, the decomposition, forecasting, the
      spectral state, and what to be careful about.

   .. grid-item-card:: Examples
      :link: auto_examples/index
      :link-type: doc
      :text-align: center

      A synthetic chirp with a known answer, then the 10-year Treasury
      yield from FRED.

   .. grid-item-card:: API reference
      :link: api/index
      :link-type: doc
      :text-align: center

      The specification, the result, the spectral state and the schedule
      records, with their sources.

Release status
--------------

.. admonition:: ns-sdn |release| is an alpha release
   :class: warning

   :bdg-warning:`API: settling` :bdg-secondary:`numerics: match the published run`

   The model-fit-result pattern and the names ``NSSDN``, ``NSSDNResult``,
   ``SpectralState``, ``Stage`` and ``Penalties`` will not change. Argument
   names, result fields and summary wording may still be renamed before
   ``1.0.0``; each rename is listed in the
   :doc:`release notes <project/release_notes>`. The default configuration
   reproduces the IEEE-ORSS DGS10 run to the parameter count. Pin the exact
   version in anything that must reproduce:

   .. parsed-literal::

      pip install "ns-sdn==\ |release|\ "

Install
-------

Python 3.12 or newer. The core depends on :mod:`numpy` and :mod:`torch`;
:mod:`pandas` is optional and switches on ``to_pandas()`` on every
result, and :mod:`fedfred` is optional and fetches the FRED series the
examples use.

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install ns-sdn
         pip install "ns-sdn[pandas,data]"     # dataframes and FRED access

   .. tab-item:: uv

      .. code-block:: bash

         uv add ns-sdn
         uv add "ns-sdn[pandas,data]"

   .. tab-item:: From source

      .. code-block:: bash

         git clone https://github.com/nikhilxsunder/ns-sdn.git
         cd ns-sdn
         uv sync --group dev

On Linux, PyPI's ``torch`` wheel pulls the CUDA stack; the
:doc:`installation page <getting_started/installation>` shows the
CPU-only install, which is all the model needs.

In sixty seconds
----------------

A series with two oscillations and a drift, a three-component model with
a short schedule, and the three things every result gives you: the
decomposition, a forecast, and a summary that says what it assumes. The
example is synthetic so it runs without a network connection; the
:doc:`examples <auto_examples/index>` do the same on FRED data.

.. testcode::

   import numpy as np
   from ns_sdn.spectral_decomposition import NSSDN, Stage

   t = np.arange(300.0)
   y = 2.0 + 0.003 * t + np.sin(0.12 * t) + 0.4 * np.sin(0.55 * t)
   y += 0.05 * np.random.default_rng(0).standard_normal(300)

   res = NSSDN(y, k=3, state_dim=16, schedule=(Stage(1e-2, 80),), seed=0).fit()

   print(res.trend.shape, res.components.shape, res.frequency.shape)   # the decomposition
   print(np.allclose(res.trend + res.components.sum(axis=1), res.emission, atol=1e-5))
   print(res.forecast(5).shape)                                        # a forecast
   print(res.summary().title, res.nobs, res.n_params)                  # a summary

.. testoutput::

   (300,) (300, 3) (300, 3)
   True
   (5,)
   NSSDN(k=3) Results 299 1155

``print(res)`` renders the summary table: one row per component with its
mean amplitude, frequency, period, gate, mixing weight and share of
power, the fit statistics above it, and under it the notes that qualify
the numbers. A result is a frozen dataclass, so everything in the table is
also an attribute, and ``to_pandas()`` returns every path as a column when
:mod:`pandas` is installed.

The model
---------

At each step the network reads a latent state and emits a trend plus
:math:`K` sinusoids,

.. math::

   \hat y_t = B_t + \sum_{k=1}^{K} w_k\, g_{t,k}\, A_{t,k}\,
   \sin\!\bigl(\theta_{t,k} + \phi_{t,k}\bigr),
   \qquad
   \theta_{t,k} = \theta_{t-1,k} + \omega_{t,k}\,\Delta t,
   \qquad
   h_t = F_\psi(h_{t-1}, y_t),

where :math:`B_t` is the trend, :math:`A_{t,k}` the amplitude,
:math:`\omega_{t,k}` the instantaneous frequency, :math:`\phi_{t,k}` the
phase offset and :math:`g_{t,k}` the gate of component :math:`k`, each a
linear head on the state :math:`h_t` of a GRU, and :math:`w` a learned
mixing vector. The phase integrates the frequency, so each component is a
chirp whose frequency the state can move. Everything to the right of the
first equals sign is returned to you as an array; the
:doc:`user guide <user_guide/model>` derives the model and the
:doc:`decomposition page <user_guide/decomposition>` says how to read
each path.

What is inside
--------------

Five public modules. The top-level ``ns_sdn`` namespace holds modules
only; every public name is imported from the module that defines it, and
the :doc:`API reference <api/index>` is organised the same way.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :mod:`ns_sdn.spectral_decomposition`
      :link: api/generated/ns_sdn.spectral_decomposition
      :link-type: doc

      ``NSSDN``, the specification; ``NSSDNResult``, the fitted record
      with every path of the decomposition, one-step and recursive
      forecasts; ``Stage`` and ``Penalties``, the training schedule.

   .. grid-item-card:: :mod:`ns_sdn.spectral_state`
      :link: api/generated/ns_sdn.spectral_state
      :link-type: doc

      ``SpectralState``: the decomposition read as a time-local measure
      on the frequency axis, with its centroid, dispersion, effective
      count, Wasserstein velocity and disentanglement score.

   .. grid-item-card:: :mod:`ns_sdn.summary`
      :link: api/generated/ns_sdn.summary
      :link-type: doc

      ``SummaryTable`` and ``InformationCriteria``, the records every
      result renders itself through, as text, HTML or a dataframe.

   .. grid-item-card:: :mod:`ns_sdn.typing`
      :link: api/generated/ns_sdn.typing
      :link-type: doc

      The categorical option aliases: ``Alignment``, ``Transition``,
      ``Objective``, ``Scaling``.

   .. grid-item-card:: :mod:`ns_sdn.exceptions`
      :link: api/generated/ns_sdn.exceptions
      :link-type: doc

      ``DimensionError``, ``SpecificationError`` and ``NumericalError``,
      one per kind of fix, under a common ``NSSDNError``.

   .. grid-item-card:: Engine
      :link: engine/index
      :link-type: doc

      The private ``_core`` and ``_internals`` layers: validators,
      containers, the torch module, the objective, the trainer and the
      predictor.

How the library is designed
---------------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: One pattern

      A model is a validated specification; ``fit()`` returns a frozen
      result. Results carry integer positions and leave your index alone;
      inputs are anything :func:`numpy.asarray` accepts. The pattern is
      :mod:`cultivars`'s, so a workflow written against one library reads
      the other.

   .. grid-item-card:: Refusals, not silent failure

      Malformed shapes raise :class:`~ns_sdn.exceptions.DimensionError`,
      inconsistent specifications
      :class:`~ns_sdn.exceptions.SpecificationError`, and non-finite data
      or a diverged fit :class:`~ns_sdn.exceptions.NumericalError`, each
      with the numbers that triggered it.

   .. grid-item-card:: Summaries that argue

      Every summary says which alignment it was fitted under and what that
      makes of its one-step numbers, that its likelihood is concentrated at
      a penalized fit, and that components without an ordering constraint
      are identified only up to permutation.

   .. grid-item-card:: The decomposition is the product

      The result carries every path the paper's figures are drawn from,
      on the original scale by default and in training units on request,
      and a ``SpectralState`` that reduces them to coordinates a
      regime-change claim can be tested against.

Citing
------

If ns-sdn contributes to published work, cite the IEEE-ORSS paper the
package implements :cite:p:`sunder2026orss`, the FLAIRS paper that
introduced the architecture :cite:p:`sunder2026flairs`, and the software;
the :doc:`citing page <project/citing>` has the entries.

.. parsed-literal::

   @software{sunder_nssdn,
     author  = {Sunder, Nikhil},
     title   = {ns-sdn: the Non-Stationary Spectral Decomposition Network},
     year    = {2026},
     version = {\ |release|\ },
     url     = {https://github.com/nikhilxsunder/ns-sdn},
     license = {MIT}
   }

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   user_guide/index
   auto_examples/index
   api/index
   engine/index
   project/index
