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
regardless of where it was raised.
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

    Args:
        endog: One-dimensional array-like; a ``(n, 1)`` column is accepted.
        min_len: Minimum admissible length.

    Returns:
        A ``(n,)`` float64 copy.

    Raises:
        DimensionError: If the input is not one-dimensional or is too short.
        NumericalError: If it contains non-finite values.

    Example:
        >>> _validate_endog([[1.0], [2.0], [3.0]]).shape
        (3,)
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

    Args:
        exog: ``(n,)`` or ``(n, p)`` array-like, or ``None``.
        nobs: Required number of rows.
        name: Label for error messages.

    Returns:
        A ``(n, p)`` float64 copy, or ``None``.

    Raises:
        DimensionError: If the rank is wrong or the rows do not match.
        NumericalError: If it contains non-finite values.
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
    """Coerce an optional time grid; default to the integers.

    Args:
        time: ``(n,)`` strictly increasing array-like, or ``None`` for
            ``0, 1, ..., n - 1``.
        nobs: Required length.

    Returns:
        A ``(n,)`` float64 grid.

    Raises:
        DimensionError: If the length does not match the series.
        SpecificationError: If the grid is not strictly increasing.
        NumericalError: If it contains non-finite values.

    Example:
        >>> _validate_time(None, 3)
        array([0., 1., 2.])
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
    """Require an integer at least ``minimum``.

    Raises:
        SpecificationError: If the value is not an integer or is too small.

    Example:
        >>> _validate_positive_int(4, "k")
        4
    """
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise SpecificationError(f"{name} must be an integer; got {value!r}.")
    if int(value) < minimum:
        raise SpecificationError(f"{name} must be at least {minimum}; got {value}.")
    return int(value)


def _validate_nonnegative(value: object, name: str) -> float:
    """Require a finite float at least zero.

    Raises:
        SpecificationError: If the value is not a non-negative finite number.
    """
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise SpecificationError(f"{name} must be a number; got {value!r}.")
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise SpecificationError(f"{name} must be non-negative and finite; got {value}.")
    return out


def _validate_unit_interval(value: object, name: str) -> float:
    """Require a float in the closed interval ``[0, 1]``.

    Raises:
        SpecificationError: If the value lies outside ``[0, 1]``.
    """
    out = _validate_nonnegative(value, name)
    if out > 1.0:
        raise SpecificationError(f"{name} must lie in [0, 1]; got {value}.")
    return out


def _validate_choice(value: object, alias: TypeAliasType, name: str) -> str:
    """Require a value from a ``Literal`` alias.

    Args:
        value: The candidate.
        alias: A PEP 695 alias over ``typing.Literal``.
        name: Label for the error message.

    Returns:
        The validated string.

    Raises:
        SpecificationError: If the value is not one of the alias's members.

    Example:
        >>> from ._types import Transition
        >>> _validate_choice("gru", Transition, "transition")
        'gru'
    """
    choices = get_args(alias.__value__)
    if value not in choices:
        listed = ", ".join(repr(c) for c in choices)
        raise SpecificationError(f"{name} must be one of {listed}; got {value!r}.")
    return str(value)
