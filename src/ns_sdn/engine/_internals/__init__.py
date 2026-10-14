# filepath: /src/ns_sdn/engine/_internals/__init__.py
#
# Copyright (c) 2026 Nikhil Sunder
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Mechanism layer: the network, the trainer, the predictor, the family base.

Torch is confined to ``_modules``, ``_objectives``, ``_trainers`` and
``_predictors``; the fit record and the public result see numpy only.
Read in dependency order:

* ``_modules`` -- the ``nn.Module``: transition, heads, synthesis, the
  teacher-forced and recursive passes.
* ``_objectives`` -- the criterion: prediction loss plus penalties.
* ``_trainers`` -- the staged Adam loop and the history record.
* ``_predictors`` -- the trained network behind a numpy interface, the
  alignment's input layout, pickling.
* ``_fits`` -- the raw fit record.
* ``_models`` -- the model bases that validate a specification and run
  the pipeline.
* ``_mixins`` -- rendering, frame interop and comparison for results.

Example:
    >>> from ns_sdn.engine import _internals
    >>> _internals.__all__
    ['_fits', '_mixins', '_models', '_modules', '_objectives', '_predictors', '_trainers']
"""

from __future__ import annotations

from . import _fits, _mixins, _models, _modules, _objectives, _predictors, _trainers

__all__ = [
    "_fits",
    "_mixins",
    "_models",
    "_modules",
    "_objectives",
    "_predictors",
    "_trainers",
]
