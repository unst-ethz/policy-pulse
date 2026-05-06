"""Shared colour-scale utilities used by both the choropleth and the word cloud."""

import pandas as pd
import plotly.colors as pc


def make_adaptive_colorscale_plotly(
    scores: pd.Series,
    base_colorscale,
    n: int = 256,
) -> tuple:
    """Build a Plotly colorscale from a consensus-score series.

    Computes the 1st/99th percentile of scores as the colour range and anchors
    the neutral midpoint colour at the mean, so the full gradient reflects the
    actual spread of the data and the average sits visually in the centre.

    Returns (colorscale, lo, avg, hi) where:
      - colorscale: list of (position, color) tuples for color_continuous_scale
      - lo, hi: colour range endpoints (1st/99th percentile, clamped to [0, 1])
      - avg: global mean (for colorbar tick label)
    """
    lo, avg, hi, midpoint_frac = _compute_scale_params(scores)
    colors = pc.sample_colorscale(base_colorscale, n)
    colorscale = [(_remap_t(i / (n - 1), midpoint_frac), color) for i, color in enumerate(colors)]
    return colorscale, lo, avg, hi


def _compute_scale_params(scores: pd.Series) -> tuple[float, float, float, float]:
    """Compute colour-scale parameters from a consensus-score series.

    Returns (lo, avg, hi, midpoint_frac) where lo/hi are the 1st/99th percentile
    (clamped to [0, 1]), avg is the mean, and midpoint_frac is avg's position in [lo, hi].
    """
    clean = scores.dropna()
    lo = float(max(clean.quantile(0.01), 0.0))
    hi = float(min(clean.quantile(0.99), 1.0))
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
