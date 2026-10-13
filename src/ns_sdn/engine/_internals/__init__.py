# Copyright (c) 2026 Nikhil Sunder
# SPDX-License-Identifier: MIT
"""Mechanism layer: the network, the trainer, the predictor, the family base.

Torch is confined to ``_modules``, ``_objectives``, ``_trainers`` and
``_predictors``; the fit record and the public result see numpy only.
"""

from __future__ import annotations

from . import _fits, _mixins, _models, _modules, _objectives, _predictors, _trainers

__all__ = [
    "_fits",
    "_mixins",
    "_models",
    "_modules",
    "_objectives",
    "_predictors",
    "_trainers",
]
