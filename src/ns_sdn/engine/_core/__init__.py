# filepath: /src/ns_sdn/engine/_core/__init__.py
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
"""Primitive layer: constants, value objects, validators, and type aliases.

Nothing in this subpackage knows what a model is, and nothing here
imports ``torch``. Every symbol has more than one consumer; mono-consumer
mechanism lives in ``_internals``.

* ``_defaults`` -- every numeric constant, including the published
  schedule.
* ``_types`` -- the ``Literal`` aliases behind :mod:`ns_sdn.typing`.
* ``_validators`` -- the coercing checks behind every constructor.
* ``_containers`` -- the penalty, stage, criteria, scale and summary
  records.

Example:
    >>> from ns_sdn.engine._core import _defaults, _validators
    >>> _validators._validate_positive_int(_defaults._SPECTRUM_NFFT, "nfft")
    256
"""

from __future__ import annotations

from . import _containers, _defaults, _types, _validators

__all__ = ["_containers", "_defaults", "_types", "_validators"]
