API reference
=============

Every public name lives in one of the five modules below. The top-level
``ns_sdn`` namespace holds modules only and re-exports nothing, so a name
is imported from the module that defines it:
``from ns_sdn.spectral_decomposition import NSSDN``. Names beginning with
an underscore, and the ``engine`` package, are implementation detail and
may change between releases without notice; the
:doc:`engine reference <../engine/index>` documents them for contributors.

.. autosummary::
   :toctree: generated
   :recursive:
   :nosignatures:

   ns_sdn.spectral_decomposition
   ns_sdn.spectral_state
   ns_sdn.summary
   ns_sdn.typing
   ns_sdn.exceptions
