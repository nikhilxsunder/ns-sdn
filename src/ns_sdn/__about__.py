# Copyright (c) 2026 Nikhil Sunder
# SPDX-License-Identifier: MIT
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
