"""Shared colour-scale utilities used by both the choropleth and the word cloud."""

import pandas as pd
import plotly.colors as pc


def make_adaptive_colorscale_plotly(
    scores: pd.Series,
    base_colorscale,
    n: int = 256,
    lo: float = None,
    hi: float = None,
) -> tuple:
    """Build a Plotly colorscale from a consensus-score series.

    Anchors the neutral midpoint colour at the mean of scores. By default, the
    colour range spans the 1st/99th percentile of scores. Pass explicit lo/hi to
    fix the range (e.g. lo=0.0, hi=1.0) while still anchoring the midpoint at
    the data mean.

    Returns (colorscale, lo, avg, hi) where:
      - colorscale: list of (position, color) tuples for color_continuous_scale
      - lo, hi: colour range endpoints used
      - avg: global mean (for colorbar tick label)
    """
    lo, avg, hi, midpoint_frac = _compute_scale_params(scores, lo=lo, hi=hi)
    colors = pc.sample_colorscale(base_colorscale, n)
    colorscale = [(_remap_t(i / (n - 1), midpoint_frac), color) for i, color in enumerate(colors)]
    return colorscale, lo, avg, hi


def _compute_scale_params(
        scores: pd.Series,
        lo: float = None,
        hi: float = None
) -> tuple[float, float, float, float]:
    """Compute colour-scale parameters from a consensus-score series.

    lo/hi default to the 1st/99th percentile (clamped to [0, 1]); pass explicit
    values to fix the range. avg is always the data mean; midpoint_frac is avg's
    position within [lo, hi].
    """
    clean = scores.dropna()
    lo = float(max(clean.quantile(0.01), 0.0)) if lo is None else lo
    hi = float(min(clean.quantile(0.99), 1.0)) if hi is None else hi
    avg = float(clean.mean())
    midpoint_frac = (avg - lo) / (hi - lo) if hi > lo else 0.5
    return lo, avg, hi, midpoint_frac


def _remap_t(t: float, midpoint_frac: float) -> float:
    """Map t in [0, 1] so the scale midpoint (t=0.5) appears at midpoint_frac.

    Linearly stretches the lower half of the scale over [0, midpoint_frac] and the
    upper half over [midpoint_frac, 1], preserving the endpoints.
    """
    if t <= 0.5:
        return t * 2 * midpoint_frac
    else:
        return midpoint_frac + (t - 0.5) * 2 * (1 - midpoint_frac)
