# Copyright (c) 2026 Nikhil Sunder
# SPDX-License-Identifier: MIT
"""Primitive layer: constants, value objects, validators, and type aliases.

Nothing in this subpackage knows what a model is, and nothing here imports
``torch``. Every symbol has more than one consumer; mono-consumer mechanism
lives in ``_internals``.
"""

from __future__ import annotations

from . import _containers, _defaults, _types, _validators

__all__ = ["_containers", "_defaults", "_types", "_validators"]
