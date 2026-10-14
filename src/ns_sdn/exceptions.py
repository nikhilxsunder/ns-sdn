# filepath: /src/ns_sdn/exceptions.py
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
"""The error hierarchy every module raises from.

Three subclasses, one per kind of fix. :class:`DimensionError` means the
geometry of an input is wrong (reshape the data);
:class:`SpecificationError` means the content of a choice is wrong
(change the argument); :class:`NumericalError` means the arithmetic could
not be carried out (look at the numbers). Each also derives from the
matching builtin, so code written against ``ValueError`` and
``RuntimeError`` keeps working, and all share :class:`NSSDNError`, so one
``except`` clause catches anything the package raises. The shape mirrors
``cultivars.exceptions`` so a caller handling one package's errors
handles the other's without a second clause.

Every message names the argument, states the rule it broke and quotes the
offending value, in that order, so a traceback is enough to fix the call:
``k must be at least 1; got 0.``

+----------------------------+--------------------+-------------------------------------------+
| Exception                  | Builtin base       | Raised when                               |
+============================+====================+===========================================+
| :class:`DimensionError`    | ``ValueError``     | A series is not one-dimensional or is too |
|                            |                    | short; covariates, a time grid or future  |
|                            |                    | covariates have the wrong length or width.|
+----------------------------+--------------------+-------------------------------------------+
| :class:`SpecificationError`| ``ValueError``     | An option is outside its ``Literal``; a   |
|                            |                    | count, rate, threshold or seed is of the  |
|                            |                    | wrong type or sign; a schedule or         |
|                            |                    | penalties record is malformed; a time grid|
|                            |                    | is not strictly increasing.               |
+----------------------------+--------------------+-------------------------------------------+
| :class:`NumericalError`    | ``RuntimeError``   | Data contain ``nan`` or ``inf``; the      |
|                            |                    | training loss becomes non-finite.         |
+----------------------------+--------------------+-------------------------------------------+

Example:
    >>> import numpy as np
    >>> from ns_sdn.spectral_decomposition import NSSDN
    >>> try:
    ...     NSSDN(np.arange(50.0), k=0)
    ... except NSSDNError as error:
    ...     print(type(error).__name__)
    SpecificationError

    The builtin bases are kept, so generic handlers still work:

    >>> issubclass(DimensionError, ValueError), issubclass(NumericalError, RuntimeError)
    (True, True)
    >>> try:
    ...     NSSDN(np.ones((4, 2)))
    ... except ValueError as error:
    ...     print(error)
    endog must be one-dimensional; got shape (4, 2).
"""

__all__ = ["DimensionError", "NSSDNError", "NumericalError", "SpecificationError"]


class NSSDNError(Exception):
    """Base class for every exception raised by ns-sdn. Never raised directly.

    Catch this to handle anything the package refuses in one clause;
    catch a subclass to handle one kind of fix.

    Example:
        >>> import numpy as np
        >>> from ns_sdn.spectral_decomposition import NSSDN
        >>> try:
        ...     NSSDN(np.array([1.0, np.nan, 3.0, 4.0]))
        ... except NSSDNError as error:
        ...     print(type(error).__name__, "-", error)
        NumericalError - endog contains non-finite values; missing data is not supported.
    """


class DimensionError(NSSDNError, ValueError):
    """An input has the wrong rank, length, or alignment.

    The data are the right kind of thing in the wrong shape: a series
    with more than one column, fewer observations than the model needs,
    covariates or a time grid whose length differs from the series, or
    future covariates whose shape differs from what a forecast of that
    horizon requires. The fix is to reshape or realign the input; the
    message quotes both shapes.

    Example:
        >>> import numpy as np
        >>> from ns_sdn.spectral_decomposition import NSSDN
        >>> NSSDN(np.arange(10.0), exog=np.ones((9, 2)))
        Traceback (most recent call last):
            ...
        ns_sdn.exceptions.DimensionError: exog has 9 rows but the series has 10.
    """


class SpecificationError(NSSDNError, ValueError):
    """A specification choice is outside its admissible set.

    An option that is not one of its ``Literal`` values, a count that is
    not a positive integer, a rate, threshold or penalty weight that is
    negative or non-finite, a seed that is not an integer, a schedule
    that is empty or holds something other than ``Stage`` records, a
    time grid that is not strictly increasing, or a comparison across
    results that do not share an effective sample. The fix is to change
    the argument; the message lists the admissible values where there is
    a finite set.

    Example:
        >>> import numpy as np
        >>> from ns_sdn.spectral_decomposition import NSSDN
        >>> NSSDN(np.arange(10.0), state_dim=2.5)
        Traceback (most recent call last):
            ...
        ns_sdn.exceptions.SpecificationError: state_dim must be an integer; got 2.5.
    """


class NumericalError(NSSDNError, RuntimeError):
    """A computation could not be carried out: non-finite data, a diverged fit.

    Raised before training when a series, covariate panel or time grid
    contains ``nan`` or ``inf`` (missing data is not imputed), and during
    training when the loss becomes non-finite at some step, in which case
    the message names the stage and step and suggests a lower learning
    rate or scaling the series. The fix is to look at the numbers.

    Example:
        >>> import numpy as np
        >>> from ns_sdn.spectral_decomposition import NSSDN
        >>> NSSDN(np.arange(10.0), time=np.r_[np.arange(9.0), np.inf])
        Traceback (most recent call last):
            ...
        ns_sdn.exceptions.NumericalError: time contains non-finite values.
    """
