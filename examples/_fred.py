# filepath: /examples/_fred.py
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
"""Shared FRED access for the example gallery.

The DGS10 example pulls its data live from FRED through ``fedfred``. This
helper does the two things it would otherwise repeat: resolve the API key
with an actionable error when it is missing, and place a series on the
calendar-day grid of the papers, with gaps of up to a week filled forward.

The leading underscore keeps the module out of the gallery
(``ignore_pattern`` in ``conf.py``); sphinx-gallery puts the examples
directory on ``sys.path`` while a script runs, so ``from _fred import
fetch_series`` resolves there exactly as it does from a shell started in
this directory.

Example:
    >>> from _fred import fetch_series  # doctest: +SKIP
    >>> log_yield = fetch_series("DGS10", start="2000-02-01", units="log")  # doctest: +SKIP
"""

from __future__ import annotations

import fedfred
import pandas as pd

_KEY_HELP = (
    "No FRED API key is configured. The examples download their data live, "
    "so set the FRED_API_KEY environment variable (a free key is issued at "
    "https://fred.stlouisfed.org/docs/api/api_key.html) or call "
    "fedfred.set_api_key(...) before importing this module's helpers."
)


def client(*, cache: bool = True) -> fedfred.FredAPI:
    """Build a ``fedfred`` client from the configured key.

    Args:
        cache: Whether to memoize responses for the life of the client, so a
            notebook that re-runs a cell does not re-download the series.

    Returns:
        A ready ``FredAPI`` instance.

    Raises:
        RuntimeError: If neither ``fedfred.set_api_key`` nor the
            ``FRED_API_KEY`` environment variable supplies a key.
    """
    key = fedfred.get_api_key()
    if not key:
        raise RuntimeError(_KEY_HELP)
    return fedfred.FredAPI(api_key=key, cache_mode=cache)


def fetch_series(
    series_id: str,
    *,
    start: str,
    end: str | None = None,
    units: str | None = None,
    fill_limit: int = 7,
) -> pd.Series:
    """Download one FRED series on the papers' daily grid.

    The observations are placed on a calendar-day index and gaps of at most
    ``fill_limit`` days are filled forward, which is how the IEEE-ORSS run
    prepared DGS10: weekends and holidays carry the last quote, longer
    outages are dropped.

    Args:
        series_id: The FRED series id, for example ``"DGS10"``.
        start: First observation date, ``"YYYY-MM-DD"``.
        end: Last observation date; ``None`` means the latest available.
        units: A FRED units code (``"lin"``, ``"log"``, ``"chg"``, ``"pch"``,
            ...); ``None`` keeps the series' native units.
        fill_limit: Longest gap, in days, to fill forward.

    Returns:
        A float series indexed by date, with no missing values.

    Raises:
        RuntimeError: If no API key is configured.
        ValueError: If the series is empty over the window.
    """
    frame = client().get_series_observations(
        series_id, observation_start=start, observation_end=end, units=units
    )
    values = pd.to_numeric(frame["value"], errors="coerce")
    values.index = pd.to_datetime(frame.index)
    values = values.sort_index().dropna().asfreq("D").ffill(limit=fill_limit).dropna()
    if values.empty:
        raise ValueError(f"no observations of {series_id} between {start} and {end or 'today'}")
    values.index.name = "date"
    return values.astype(float).rename(series_id)
