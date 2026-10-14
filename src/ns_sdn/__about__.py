# filepath: /src/ns_sdn/__about__.py
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
"""Package metadata: the single source of the version and the identity strings.

The build reads ``__version__`` from this file, so the distribution's
version and the installed package's agree by construction. Nothing here
imports ``numpy`` or ``torch``, so the version is readable at zero cost.

Example:
    >>> from ns_sdn.__about__ import __title__, __license__
    >>> __title__, __license__
    ('ns-sdn', 'MIT')
"""

__title__ = "ns-sdn"
__description__ = (
    "Non-Stationary Spectral Decomposition Network: a neural state-space model"
    " emitting an interpretable time-varying spectral decomposition."
)
__version__ = "1.0.0a1"
__copyright__ = "Copyright (c) 2026 Nikhil Sunder"
__author__ = "Nikhil Sunder"
__email__ = "nsunder724@gmail.com"
__license__ = "MIT"
__repository__ = "https://github.com/nikhilxsunder/ns-sdn"
__docs__ = "https://nikhilxsunder.github.io/ns-sdn/"
