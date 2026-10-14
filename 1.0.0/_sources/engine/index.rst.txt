Engine reference
================

The layers behind the public API, for contributors. ``_core`` holds the
pure values: the option aliases, the defaults, the validators, and the
containers (penalties, stages, the scaling map, the summary records).
``_internals`` holds the mechanism: the torch module with its heads and
recursions, the objective, the staged trainer, the predictor that runs
the trained network on numpy data, the fit record, the model base, and
the behaviour mixins the result composes. The public modules appear a
second time below with each class's own private members. Everything on
these pages may change between releases without notice.

Private layers
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: engine/module.rst

   ns_sdn.engine._core
   ns_sdn.engine._internals

Public surface, private members
-------------------------------

.. autosummary::
   :toctree: generated
   :recursive:
   :nosignatures:
   :template: surface/module.rst

   ns_sdn.spectral_decomposition
   ns_sdn.spectral_state
