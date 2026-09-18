"""Colour-scale parameters, with the all-NaN slice the Postgres migration made common.

Since the migration the resolution table also carries resolutions adopted without a vote and by
non-recorded vote, neither of which has a consensus score — about 72% of rows. A narrow enough
date range therefore selects rows that are *entirely* unscored, and both the choropleth and the
word cloud used to hand that straight to the adaptive colour scale.
"""

import math

import pandas as pd
import pytest

from app.features.color_utils import _compute_scale_params, make_adaptive_colorscale_plotly

ALL_NAN = pd.Series([float("nan")] * 5)


def test_all_nan_scores_yield_a_finite_scale():
    """NaN here reaches plotly as `tickvals=[0, nan, 1]` and renders an axis label reading 'nan'."""
    lo, avg, hi, midpoint_frac = _compute_scale_params(ALL_NAN)

    assert all(math.isfinite(v) for v in (lo, avg, hi, midpoint_frac))
    assert lo <= avg <= hi
    assert 0.0 <= midpoint_frac <= 1.0


def test_all_nan_scores_respect_an_explicit_range():
    """The choropleth pins lo/hi to the full [0, 1] agreement range; only avg is derived."""
    lo, avg, hi, _ = _compute_scale_params(ALL_NAN, lo=0.0, hi=1.0)

    assert (lo, hi) == (0.0, 1.0)
    assert math.isfinite(avg)


def test_empty_series_yields_a_finite_scale():
    lo, avg, hi, midpoint_frac = _compute_scale_params(pd.Series([], dtype="float64"))

    assert all(math.isfinite(v) for v in (lo, avg, hi, midpoint_frac))


def test_adaptive_colorscale_survives_an_all_nan_slice():
    """The public entry point both callers use."""
    colorscale, lo, avg, hi = make_adaptive_colorscale_plotly(ALL_NAN, "RdYlBu", lo=0.0, hi=1.0)

    assert all(math.isfinite(v) for v in (lo, avg, hi))
    assert colorscale
    assert all(math.isfinite(stop) for stop, _ in colorscale)


def test_scored_series_still_anchors_on_its_own_mean():
    """The guard above must not disturb the normal path."""
    scores = pd.Series([0.2, 0.4, 0.6, 0.8, float("nan")])

    _, avg, _, _ = _compute_scale_params(scores, lo=0.0, hi=1.0)

    assert avg == pytest.approx(0.5)
