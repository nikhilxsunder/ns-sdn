# filepath: /src/ns_sdn/engine/_core/_validators.py
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
"""Input validation primitives.

Every public constructor routes its argument checking through this module,
so a malformed input produces the same exception type and message shape
regardless of where it was raised. Each validator coerces as well as
checks: it returns the value in the form the rest of the package assumes
(a float64 copy, a Python ``int`` or ``float``, the string of a
``Literal``), so a caller can bind the return value and forget the input.

The division of labor follows :mod:`ns_sdn.exceptions`: geometry raises
:class:`~ns_sdn.exceptions.DimensionError`, content raises
:class:`~ns_sdn.exceptions.SpecificationError`, non-finite numbers raise
:class:`~ns_sdn.exceptions.NumericalError`. Messages name the argument,
state the rule and quote the value.

Example:
    >>> import numpy as np
    >>> _validate_endog(np.arange(5.0)).dtype, _validate_positive_int(3, "k")
    (dtype('float64'), 3)
"""

from __future__ import annotations

from typing import TypeAliasType, get_args

import numpy as np
import numpy.typing as npt

from ...exceptions import DimensionError, NumericalError, SpecificationError

__all__ = [
    "_validate_choice",
    "_validate_endog",
    "_validate_exog",
    "_validate_nonnegative",
    "_validate_positive_int",
    "_validate_time",
    "_validate_unit_interval",
]


def _validate_endog(endog: npt.ArrayLike, *, min_len: int = 3) -> npt.NDArray[np.float64]:
    """Coerce a univariate series to a finite float64 vector.

    A ``(n, 1)`` column is flattened, because that is how a single column
    of a frame or a matrix arrives; any other two-dimensional input is a
    multivariate series and is refused. The returned array is a copy, so a
    model never aliases the caller's data.

    Args:
        endog: One-dimensional array-like of numbers; a ``(n, 1)`` column
            is accepted. Anything :func:`numpy.asarray` turns into floats.
        min_len: Minimum admissible length: three for a specification
            (two observations for a one-step fit plus one for a variance),
            two for a prediction pass.

    Returns:
        A ``(n,)`` float64 copy.

    Raises:
        DimensionError: If the input is not one-dimensional (after
            flattening a column) or has fewer than ``min_len``
            observations.
        NumericalError: If it contains ``nan`` or ``inf``.

    Example:
        >>> _validate_endog([[1.0], [2.0], [3.0]]).shape
        (3,)
        >>> _validate_endog([1.0, 2.0])
        Traceback (most recent call last):
            ...
        ns_sdn.exceptions.DimensionError: endog must have at least 3 observations; got 2.
        >>> _validate_endog([1.0, 2.0], min_len=2)
        array([1., 2.])
    """
    y = np.asarray(endog, dtype=np.float64)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise DimensionError(f"endog must be one-dimensional; got shape {y.shape}.")
    if y.shape[0] < min_len:
        raise DimensionError(f"endog must have at least {min_len} observations; got {y.shape[0]}.")
    if not np.all(np.isfinite(y)):
        raise NumericalError("endog contains non-finite values; missing data is not supported.")
    return y.copy()


def _validate_exog(
    exog: npt.ArrayLike | None, nobs: int, *, name: str = "exog"
) -> npt.NDArray[np.float64] | None:
    """Coerce an optional exogenous panel aligned with the series.

    A one-dimensional input is promoted to a single column. The panel is
    used as given: no scaling, no intercept, no lagging, so a covariate
    that must be known at :math:`t` to be legitimate is the caller's
    responsibility to lag.

    Args:
        exog: ``(n,)`` or ``(n, p)`` array-like, or ``None`` for no
            covariates.
        nobs: Required number of rows, the length of the series it
            accompanies.
        name: Label for error messages, so a future panel can say
            ``future_exog``.

    Returns:
        A ``(n, p)`` float64 copy, or ``None`` when ``exog`` is ``None``.

    Raises:
        DimensionError: If the rank is above two or the rows do not match
            ``nobs``.
        NumericalError: If it contains ``nan`` or ``inf``.

    Example:
        >>> _validate_exog([1.0, 2.0, 3.0], 3).shape
        (3, 1)
        >>> _validate_exog(None, 3) is None
        True
        >>> _validate_exog(np.ones((4, 2)), 3)
        Traceback (most recent call last):
            ...
        ns_sdn.exceptions.DimensionError: exog has 4 rows but the series has 3.
    """
    if exog is None:
        return None
    x = np.asarray(exog, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise DimensionError(f"{name} must be one- or two-dimensional; got shape {x.shape}.")
    if x.shape[0] != nobs:
        raise DimensionError(f"{name} has {x.shape[0]} rows but the series has {nobs}.")
    if not np.all(np.isfinite(x)):
        raise NumericalError(f"{name} contains non-finite values.")
    return x.copy()


def _validate_time(time: npt.ArrayLike | None, nobs: int) -> npt.NDArray[np.float64]:
    r"""Coerce an optional time grid; default to the integers.

    The grid is what the phase recursion integrates over,
    :math:`\theta_{t,k} = \theta_{t-1,k} + \omega_{t,k}(t_t - t_{t-1})`,
    so it must be strictly increasing: a repeated time would freeze the
    phase and a decreasing one would run it backwards. Units are the
    caller's; frequencies are reported in radians per unit of this grid.

    Args:
        time: ``(n,)`` strictly increasing array-like, or ``None`` for
            :math:`0, 1, \ldots, n - 1`. Any shape is flattened.
        nobs: Required length, the length of the series.

    Returns:
        A ``(n,)`` float64 grid (a copy when one was given).

    Raises:
        DimensionError: If the length does not match the series.
        SpecificationError: If the grid is not strictly increasing.
        NumericalError: If it contains ``nan`` or ``inf``.

    Example:
        >>> _validate_time(None, 3)
        array([0., 1., 2.])
        >>> _validate_time([0.0, 0.5, 0.5], 3)
        Traceback (most recent call last):
            ...
        ns_sdn.exceptions.SpecificationError: time must be strictly increasing.
    """
    if time is None:
        return np.arange(nobs, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64).reshape(-1)
    if t.shape[0] != nobs:
        raise DimensionError(f"time has {t.shape[0]} entries but the series has {nobs}.")
    if not np.all(np.isfinite(t)):
        raise NumericalError("time contains non-finite values.")
    if nobs > 1 and not np.all(np.diff(t) > 0.0):
        raise SpecificationError("time must be strictly increasing.")
    return t.copy()


def _validate_positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    """Require an integer at least ``minimum`` and return it as a Python ``int``.

    Booleans are refused even though they subclass ``int``, so
    ``k=True`` does not silently mean one component. NumPy integer
    scalars are accepted and converted.

    Args:
        value: The candidate.
        name: Label for the error message.
        minimum: Smallest admissible value.

    Returns:
        ``int(value)``.

    Raises:
        SpecificationError: If the value is not an integer or is below
            ``minimum``.

    Example:
        >>> _validate_positive_int(4, "k")
        4
        >>> _validate_positive_int(np.int64(2), "steps", minimum=2)
        2
        >>> _validate_positive_int(True, "k")
        Traceback (most recent call last):
            ...
        ns_sdn.exceptions.SpecificationError: k must be an integer; got True.
    """
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise SpecificationError(f"{name} must be an integer; got {value!r}.")
    if int(value) < minimum:
        raise SpecificationError(f"{name} must be at least {minimum}; got {value}.")
    return int(value)


def _validate_nonnegative(value: object, name: str) -> float:
    """Require a finite number at least zero and return it as a Python ``float``.

    The check behind every penalty weight, learning rate and threshold.
    Booleans are refused; integers and NumPy scalars are accepted and
    converted.

    Args:
        value: The candidate.
        name: Label for the error message.

    Returns:
        ``float(value)``.

    Raises:
        SpecificationError: If the value is not a number, is negative, or
            is not finite.

    Example:
        >>> _validate_nonnegative(0, "amplitude")
        0.0
        >>> _validate_nonnegative(-1e-3, "frequency")
        Traceback (most recent call last):
            ...
        ns_sdn.exceptions.SpecificationError: frequency must be non-negative and finite; got -0.001.
    """
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise SpecificationError(f"{name} must be a number; got {value!r}.")
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise SpecificationError(f"{name} must be non-negative and finite; got {value}.")
    return out


def _validate_unit_interval(value: object, name: str) -> float:
    r"""Require a number in the closed interval :math:`[0, 1]` and return it as a ``float``.

    The check behind the spectral-state ``active_threshold``, a
    probability-weight floor.

    Args:
        value: The candidate.
        name: Label for the error message.

    Returns:
        ``float(value)``.

    Raises:
        SpecificationError: If the value is not a finite number in
            :math:`[0, 1]`.

    Example:
        >>> _validate_unit_interval(0.05, "active_threshold")
        0.05
        >>> _validate_unit_interval(1.5, "active_threshold")
        Traceback (most recent call last):
            ...
        ns_sdn.exceptions.SpecificationError: active_threshold must lie in [0, 1]; got 1.5.
    """
    out = _validate_nonnegative(value, name)
    if out > 1.0:
        raise SpecificationError(f"{name} must lie in [0, 1]; got {value}.")
    return out


def _validate_choice(value: object, alias: TypeAliasType, name: str) -> str:
    """Require a value from a ``Literal`` alias and return it as a ``str``.

    Reads the admissible set off the alias at call time, so the message
    lists exactly the values the annotation admits and the two cannot
    disagree.

    Args:
        value: The candidate.
        alias: A PEP 695 alias over :data:`typing.Literal`, one of
            :mod:`ns_sdn.typing`.
        name: Label for the error message.

    Returns:
        The validated string.

    Raises:
        SpecificationError: If the value is not one of the alias's members.

    Example:
        >>> from ._types import Transition
        >>> _validate_choice("gru", Transition, "transition")
        'gru'
        >>> _validate_choice("GRU", Transition, "transition")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ns_sdn.exceptions.SpecificationError: transition must be one of 'gru', 'timeaware'; got ...
    """
    choices = get_args(alias.__value__)
    if value not in choices:
        listed = ", ".join(repr(c) for c in choices)
        raise SpecificationError(f"{name} must be one of {listed}; got {value!r}.")
    return str(value)
