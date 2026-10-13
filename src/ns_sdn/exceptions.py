# Copyright (c) 2026 Nikhil Sunder
# SPDX-License-Identifier: MIT
"""The error hierarchy every module raises from.

Three subclasses, one per kind of fix. ``DimensionError`` means the
geometry of an input is wrong (reshape the data); ``SpecificationError``
means the content of a choice is wrong (change the argument);
``NumericalError`` means the arithmetic could not be carried out (look at
the numbers). Each also derives from the matching builtin, so code written
against ``ValueError`` and ``RuntimeError`` keeps working. The shape
mirrors ``cultivars.exceptions`` so a caller handling one package's errors
handles the other's without a second ``except`` clause.

Example:
    >>> import numpy as np
    >>> from ns_sdn.spectral_decomposition import NSSDN
    >>> try:
    ...     NSSDN(np.arange(50.0), k=0)
    ... except NSSDNError as error:
    ...     print(type(error).__name__)
    SpecificationError
"""

__all__ = ["DimensionError", "NSSDNError", "NumericalError", "SpecificationError"]


class NSSDNError(Exception):
    """Base class for every exception raised by ns-sdn. Never raised directly."""


class DimensionError(NSSDNError, ValueError):
    """An input has the wrong rank, length, or alignment."""


class SpecificationError(NSSDNError, ValueError):
    """A specification choice is outside its admissible set."""


class NumericalError(NSSDNError, RuntimeError):
    """A computation could not be carried out: non-finite data, a diverged fit."""
