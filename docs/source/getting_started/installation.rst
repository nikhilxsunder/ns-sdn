.. meta::
   :description: Install ns-sdn with pip or uv, choose a CPU-only torch on Linux, pick the pandas and data extras, pin a version, and set up a development checkout.

Installation
============

Requirements
------------

ns-sdn supports CPython 3.12, 3.13 and 3.14 on Linux, macOS and Windows.
The core needs :mod:`numpy` and :mod:`torch`; everything else is an
extra:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Extra
     - Installs
     - Switches on
   * - *(none)*
     - numpy, torch
     - The model, the result, the spectral state. Inputs are array-likes;
       outputs are :class:`numpy.ndarray` and plain records.
   * - ``pandas``
     - pandas
     - ``to_pandas()`` on results, spectral states and summary tables.
   * - ``data``
     - fedfred
     - FRED access for the two DGS10 examples.
   * - ``plot``
     - matplotlib
     - Figures in the examples. Nothing in the library draws.
   * - ``all``
     - pandas, fedfred, matplotlib
     - All of the above.

Install
-------

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install ns-sdn
         pip install "ns-sdn[pandas,data]"

   .. tab-item:: uv

      .. code-block:: bash

         uv add ns-sdn
         uv add "ns-sdn[pandas,data]"

   .. tab-item:: Pinned

      .. parsed-literal::

         pip install "ns-sdn==\ |release|\ "

Pin the exact version in anything that must reproduce: a paper's
replication package, a CI job, a lockfile. From ``1.0.0`` the public
names, argument names, result fields and summary wording are frozen for
the ``1.x`` line; a release that changes the output of the default
configuration on the published data is a major version, and every change
to a public surface is listed in the
:doc:`release notes <../project/release_notes>`.

torch on Linux
--------------

The model runs on the CPU; the published run is a 58,849-parameter
network and trains well under an hour on a laptop CPU. PyPI's Linux ``torch`` wheel
nevertheless depends on the CUDA libraries, several gigabytes that a
CPU-only machine never loads. Install torch from the CPU index first and
ns-sdn will use it:

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install torch --index-url https://download.pytorch.org/whl/cpu
         pip install ns-sdn

   .. tab-item:: uv (project)

      Pin the source in ``pyproject.toml`` and the lock stays CPU-only on
      Linux while macOS and Windows keep PyPI's wheel:

      .. code-block:: toml

         [tool.uv.sources]
         torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

         [[tool.uv.index]]
         name = "pytorch-cpu"
         url = "https://download.pytorch.org/whl/cpu"
         explicit = true

macOS wheels are CPU and Metal already. A GPU is supported through the
``device`` argument of :class:`~ns_sdn.spectral_decomposition.NSSDN` but
buys little at this size: training is one full-sequence recurrent pass
per step, which is latency-bound rather than throughput-bound.

Check the installation
----------------------

.. code-block:: bash

   python -c "from importlib.metadata import version; print(version('ns-sdn'))"

The same string is available inside Python as
``ns_sdn.__about__.__version__``, and the quick test that everything
imports and trains is the doctest on the package:

.. code-block:: bash

   python -c "import doctest, ns_sdn; print(doctest.testmod(ns_sdn))"

Data for the examples
---------------------

The examples download the 10-year Treasury yield from FRED through
`fedfred <https://nikhilxsunder.github.io/fedfred/>`_, which needs a
free API key from the
`St. Louis Fed <https://fred.stlouisfed.org/docs/api/api_key.html>`_.
Set it once in your shell:

.. code-block:: bash

   export FRED_API_KEY=your_key

The quickstart and every docstring example run on simulated data and
need no key.

Development install
-------------------

A development checkout uses `uv <https://docs.astral.sh/uv/>`_ and the
``dev`` dependency group, which brings in the test, lint, type-checking,
documentation and notebook tools:

.. code-block:: bash

   git clone https://github.com/nikhilxsunder/ns-sdn.git
   cd ns-sdn
   uv sync --group dev
   uv run pre-commit install

Then:

.. code-block:: bash

   uv run pytest                                  # doctests + tests, with coverage
   uv run ruff check src/ns_sdn tests examples    # lint
   uv run mypy                                    # type-check
   uv run make -C docs html                       # docs, running the examples
   DOCS_PLOT_GALLERY=0 uv run make -C docs html   # docs, skipping the examples

Keep the checkout outside folders a sync service mirrors, such as
iCloud Drive's Desktop & Documents or Dropbox. Those services create
numbered copies (``file 2.py``) of files that tools rewrite quickly,
and Python, Sphinx and uv pick the copies up as real files.

Troubleshooting
---------------

``Requires-Python >=3.12``
   ns-sdn uses syntax from Python 3.12 (PEP 695 type aliases). Create an
   environment with a supported interpreter, for example
   ``uv venv --python 3.13``.

``ImportError: to_pandas requires pandas``
   Install the ``pandas`` extra, or read the result's array attributes
   and ``to_dict()`` directly.

``NumericalError: training diverged``
   The loss became non-finite. Lower the learning rate of the first
   :class:`~ns_sdn.spectral_decomposition.Stage`, keep ``scaling="minmax"``,
   or check the series for outliers the min-max scale is stretching over.

The fit is slow
   Cost scales with the sample length times the number of steps, because
   every step is a gradient through the whole sequence. Use a shorter
   schedule while exploring (``schedule=(Stage(1e-2, 200),)``) and the
   published two stages only for the final run; set ``seed`` so the
   exploration reproduces.
