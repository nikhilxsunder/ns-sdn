Citing
======

If ns-sdn contributes to published work, cite the two papers and the
software: the FLAIRS paper introduced the architecture, and the IEEE-ORSS
paper is the formulation the package implements, with its spectral emission
heads and the DGS10 experiments the defaults reproduce.

The papers
----------

.. code-block:: bibtex

   @article{sunder2026flairs,
     title   = {Non-Stationary Spectral Decomposition Network for Econometric Time Series Forecasting},
     author  = {Sunder, Nikhil},
     journal = {The International FLAIRS Conference Proceedings},
     volume  = {39},
     number  = {1},
     year    = {2026},
     doi     = {10.32473/flairs.39.1.141588}
   }

   @inproceedings{sunder2026orss,
     title     = {Non-Stationary Spectral Decomposition Network: Adaptive Spectral Emission Heads and Frequency Modulation},
     author    = {Sunder, Nikhil and Leon, Maikel},
     booktitle = {2026 IEEE Opportunity Research Scholars Symposium (ORSS)},
     address   = {Atlanta, GA, USA},
     publisher = {IEEE},
     pages     = {1--5},
     year      = {2026},
     doi       = {10.1109/ORSS71174.2026.11684282}
   }

The software
------------

.. parsed-literal::

   @software{sunder_nssdn,
     author    = {Sunder, Nikhil},
     title     = {ns-sdn: the Non-Stationary Spectral Decomposition Network},
     year      = {2026},
     publisher = {Zenodo},
     doi       = {10.5281/zenodo.23368625},
     url       = {https://github.com/nikhilxsunder/ns-sdn},
     license   = {MIT}
   }

The DOI is the Zenodo concept DOI, which resolves to the latest release;
every release is archived under its own DOI, listed on the
`Zenodo record <https://doi.org/10.5281/zenodo.23368625>`_. To pin a
version in a citation, add ``version = {\ |release|\ }`` and cite that
release's DOI. The :github:`CITATION.cff` file in the repository carries
the same entries in a form GitHub and Zenodo read directly.

Bibliography
------------

Every work the documentation cites; citations elsewhere link here.

.. bibliography::
   :all:
