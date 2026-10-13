# Copyright (c) 2026 Nikhil Sunder
# SPDX-License-Identifier: MIT
"""Categorical option aliases for the public constructors.

Each name is a PEP 695 alias over a ``Literal``: an annotation for the
static checker and the argument the runtime validator reads. A name here is
a specification choice with modeling content, never a display option.

Example:
    >>> from ns_sdn.typing import Alignment
    >>> from typing import get_args
    >>> get_args(Alignment.__value__)
    ('filtered', 'predicted')
"""

from __future__ import annotations

from .engine._core._types import Alignment, Objective, Scaling, Transition

__all__ = ["Alignment", "Objective", "Scaling", "Transition"]
