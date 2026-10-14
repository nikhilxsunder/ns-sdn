# filepath: /src/ns_sdn/engine/__init__.py
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
"""Engine subpackage: the private ``_core`` and ``_internals`` layers, for extenders.

Nothing here is public API; names and signatures may change between
releases without notice, and the public modules re-export the few
records a user handles under stable names. The split is by dependency:
``_core`` imports numpy only and holds values, validators and constants;
``_internals`` imports torch and holds mechanism. A subclass of the
family base that adds a head, a penalty or a transition lives at this
level and surfaces through a public module of its own.

Example:
    >>> from ns_sdn import engine
    >>> engine.__all__
    ['_core', '_internals']
    >>> import sys
    >>> "torch" in sys.modules  # _internals imports torch; _core alone would not
    True
"""

from __future__ import annotations

from . import _core, _internals

__all__ = ["_core", "_internals"]
