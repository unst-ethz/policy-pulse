"""Keep missing observations distinct from numeric zero at the HTTP boundary."""

import math
from datetime import date, datetime

import numpy as np
import pandas as pd


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()[:10]
    return value


def records(frame: pd.DataFrame) -> list[dict]:
    return clean(frame.to_dict("records"))
