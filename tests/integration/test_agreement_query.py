"""Integration tests for correctness and performance of query_agreement_between_countries.

--- Bilateral agreement score ---
For a single resolution and two countries A and B, the bilateral agreement score
is defined as:

    score(A, B) = 1 - |v_A - v_B| / 2

where v_A, v_B ∈ {1 (yes), 0 (abstained), -1 (no)}. A score of NaN is produced
whenever either country did not vote (X or missing). Because the vote values are
integers, the score can only ever be exactly 0.0, 0.5, or 1.0.

Across a set of resolutions the score is the nanmean of per-resolution scores.

--- Correctness tests ---
Tests use a reference implementation (_reference_agreement) that derives expected
values directly from raw vote strings in resolution_table. It is intentionally
naive — simple enough to be obviously correct — and shares no code with the query
engine. Tests assert that the engine matches this reference rather than
hard-coded 'golden values', so correctness is re-verified against the live
dataset on every run.

--- Performance tests ---
query_agreement_between_countries is called in two modes:

    average=True  — choropleth map: one mean score per country, triggered on
                    every filter change; the most latency-sensitive path.
    average=False — profile page and agreement-by-subject: full (R' × C)
                    DataFrame needed for year-based group-bys or subject filtering.

Each mode is tested at three resolution-count levels that reflect real user
date-filter choices:

    full      (+5'000 res) — no date filter, entire history from 1946
    post-2000 (+2'000 res) — a common "last 25 years" filter
    decade    (~800 res)   — a tighter filter such as 2010–2019

Performance budgets are set as regression guards: tight enough that a
naive loop-based implementation cannot meet them, with comfortable
headroom above what the vectorised implementation actually takes.
"""

import time

import numpy as np
import pandas as pd
import pytest

SAMPLE_SIZE_CORRECTNESS = 500
C1 = "USA"
C2 = "CHN"

# Time budgets in seconds (vectorised implementation targets)
TIME_BUDGETS = {
    "choropleth_full":     0.5,   # 5'000+ resolutions, average=True
    "choropleth_post2000": 0.2,   # 2'000+ resolutions, average=True
    "choropleth_decade":   0.1,   #  ~800 resolutions, average=True
    "bilateral_full":      0.75,  # 5'000+ resolutions, average=False
    "bilateral_post2000":  0.25,  # 2'000+ resolutions, average=False
}


def _data_available() -> bool:
    try:
        from app import data as app_data
        return not app_data.query_engine.query_resolutions().empty
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _data_available(), reason="Local resolution data files not available"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    from app import data as app_data
    return app_data.query_engine


@pytest.fixture(scope="module")
def resolution_table(engine):
    return engine.query_resolutions()


@pytest.fixture(scope="module")
def sample_res_ids(resolution_table):
    return resolution_table["undl_id"].iloc[:SAMPLE_SIZE_CORRECTNESS].tolist()


@pytest.fixture(scope="module")
def all_res_ids(resolution_table):
    return resolution_table["undl_id"].tolist()


@pytest.fixture(scope="module")
def post2000_res_ids(resolution_table):
    return resolution_table[resolution_table["date"] >= "2000-01-01"]["undl_id"].tolist()


@pytest.fixture(scope="module")
def decade_res_ids(resolution_table):
    mask = (resolution_table["date"] >= "2010-01-01") & (resolution_table["date"] < "2020-01-01")
    return resolution_table[mask]["undl_id"].tolist()


def _reference_agreement(resolution_table: pd.DataFrame, c1: str, c2: str, ids) -> pd.Series:
    """Ground-truth bilateral agreement derived from raw vote strings."""
    vote_map = {"Y": 1.0, "A": 0.0, "N": -1.0}
    df = resolution_table[resolution_table["undl_id"].isin(set(ids))].set_index("undl_id")
    v1 = df[c1].map(vote_map)
    v2 = df[c2].map(vote_map)
    return 1.0 - (v1 - v2).abs() / 2.0  # Agreement score formula (pandas implementation)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_result_is_dataframe_with_undl_id(engine, sample_res_ids):
    result = engine.query_agreement_between_countries(
        country_code=C1, resolution_ids=sample_res_ids, average=False
    )
    assert isinstance(result, pd.DataFrame)
    assert "undl_id" in result.columns


def test_country_columns_present(engine, sample_res_ids):
    result = engine.query_agreement_between_countries(
        country_code=C1, resolution_ids=sample_res_ids, average=False
    )
    assert C2 in result.columns, f"{C2} column missing from result"
    assert C1 not in result.columns or result[C1].isna().all(), (
        f"{C1} self-agreement column should be NaN"
    )


def test_values_are_discrete(engine, sample_res_ids):
    """Agreement scores must be exactly 0.0, 0.5, or 1.0 (or NaN)."""
    result = engine.query_agreement_between_countries(
        country_code=C1, resolution_ids=sample_res_ids, average=False
    )
    col = result[C2].dropna()
    allowed = {0.0, 0.5, 1.0}
    unexpected = set(col.round(6).unique()) - allowed
    assert not unexpected, f"Non-discrete values found: {unexpected}"


def test_values_in_range(engine, sample_res_ids):
    result = engine.query_agreement_between_countries(
        country_code=C1, resolution_ids=sample_res_ids, average=False
    )
    col = result[C2].dropna()
    assert (col >= 0.0).all() and (col <= 1.0).all()


def test_symmetry(engine, sample_res_ids):
    """agree(A→B, r) must equal agree(B→A, r) for every resolution."""
    df_c1 = engine.query_agreement_between_countries(
        country_code=C1, resolution_ids=sample_res_ids, average=False
    ).set_index("undl_id")
    df_c2 = engine.query_agreement_between_countries(
        country_code=C2, resolution_ids=sample_res_ids, average=False
    ).set_index("undl_id")

    common_ids = df_c1.index.intersection(df_c2.index)
    a_from_c1 = df_c1.loc[common_ids, C2]
    a_from_c2 = df_c2.loc[common_ids, C1]

    diff = (a_from_c1 - a_from_c2).abs().dropna()
    assert (diff < 1e-6).all(), f"Symmetry violated; max diff = {diff.max():.2e}"


def test_matches_reference_computation(engine, resolution_table, sample_res_ids):
    """Per-resolution scores must match the reference implementation to float32 precision."""
    result = engine.query_agreement_between_countries(
        country_code=C1, resolution_ids=sample_res_ids, average=False
    ).set_index("undl_id")

    ref = _reference_agreement(resolution_table, C1, C2, sample_res_ids)

    common = result.index.intersection(ref.index)
    r_vals = result.loc[common, C2]
    e_vals = ref.loc[common]

    pd.testing.assert_series_equal(
        r_vals.isna(), e_vals.isna(), check_names=False,
        obj=f"NaN pattern mismatch for {C1}–{C2}",
    )

    mask = r_vals.notna() & e_vals.notna()
    np.testing.assert_allclose(
        r_vals[mask].to_numpy(dtype=float),
        e_vals[mask].to_numpy(dtype=float),
        atol=1e-5,
        err_msg=f"Per-resolution scores differ from reference for {C1}–{C2}",
    )


def test_average_true_matches_nanmean(engine, resolution_table, sample_res_ids):
    """average=True result must equal nanmean of per-resolution reference scores."""
    result_avg = engine.query_agreement_between_countries(
        country_code=C1, resolution_ids=sample_res_ids, average=True
    )
    assert isinstance(result_avg, (pd.Series, pd.DataFrame))

    engine_mean = result_avg[C2] if isinstance(result_avg, pd.Series) else result_avg[C2].iloc[0]

    ref = _reference_agreement(resolution_table, C1, C2, sample_res_ids)
    ref_mean = float(np.nanmean(ref.to_numpy(dtype=float)))

    assert abs(engine_mean - ref_mean) < 1e-4, (
        f"average=True mismatch: engine={engine_mean:.6f}, reference={ref_mean:.6f}"
    )


def test_pre_membership_resolutions_are_nan(engine):
    """Countries that had not yet joined the UN yet must have NaN agreement, not 0 or 1."""
    # Switzerland (CHE) joined the UN in 2002; resolutions before that should be NaN
    all_res = engine.query_resolutions()
    pre_membership_ids = all_res[all_res["date"] < "2002-01-01"]["undl_id"].tolist()

    if not pre_membership_ids:
        pytest.skip("No pre-2002 resolutions in dataset")

    result = engine.query_agreement_between_countries(
        country_code="CHE", resolution_ids=pre_membership_ids, average=False
    ).set_index("undl_id")

    check_col = "USA"
    if check_col not in result.columns:
        pytest.skip(f"{check_col} column not available")

    non_nan = result[check_col].dropna()
    assert len(non_nan) == 0, (
        f"CHE has {len(non_nan)} non-NaN agreement scores before UN membership"
    )


# ---------------------------------------------------------------------------
# resolution_table dtype tests
# ---------------------------------------------------------------------------

def test_vote_columns_are_categorical(engine):
    """Vote columns must be loaded as CategoricalDtype, not object.

    Regression guard: if the two-pass CSV read in _read_resolution_table is
    ever removed or bypassed, this test fails immediately rather than silently
    restoring ~50 MB of unnecessary object-array overhead.
    """
    import pandas as pd
    rt = engine.resolution_table
    from app.un_data_stream.data.repository import DataRepository
    vote_cols = [c for c in rt.columns if c not in DataRepository._RESOLUTION_META_COLS]
    assert vote_cols, "No vote columns found in resolution_table"
    non_categorical = [c for c in vote_cols if not isinstance(rt[c].dtype, pd.CategoricalDtype)]
    assert not non_categorical, (
        f"{len(non_categorical)} vote columns have non-categorical dtype: {non_categorical[:5]}"
    )


def test_vote_values_in_expected_set(engine):
    """Every non-NaN value in every vote column must be one of Y / N / A / X.

    Catches upstream data changes that introduce a new vote code which would
    silently map to NaN in the categorical dtype and corrupt agreement scores.
    """
    from app.un_data_stream.data.repository import DataRepository
    rt = engine.resolution_table
    vote_cols = [c for c in rt.columns if c not in DataRepository._RESOLUTION_META_COLS]
    allowed = {"Y", "N", "A", "X"}
    for col in vote_cols:
        observed = set(rt[col].dropna().unique())
        unexpected = observed - allowed
        assert not unexpected, (
            f"Column '{col}' contains unexpected vote values: {unexpected}"
        )


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------

def test_choropleth_full_within_budget(engine, all_res_ids):
    t0 = time.perf_counter()
    result = engine.query_agreement_between_countries(
        country_code=C1, resolution_ids=all_res_ids, average=True
    )
    elapsed = time.perf_counter() - t0
    assert result is not None
    budget = TIME_BUDGETS["choropleth_full"]
    assert elapsed < budget, (
        f"choropleth full ({len(all_res_ids)} res): {elapsed:.3f}s > {budget}s budget"
    )


def test_choropleth_post2000_within_budget(engine, post2000_res_ids):
    t0 = time.perf_counter()
    result = engine.query_agreement_between_countries(
        country_code=C1, resolution_ids=post2000_res_ids, average=True
    )
    elapsed = time.perf_counter() - t0
    assert result is not None
    budget = TIME_BUDGETS["choropleth_post2000"]
    assert elapsed < budget, (
        f"choropleth post-2000 ({len(post2000_res_ids)} res): {elapsed:.3f}s > {budget}s budget"
    )


def test_choropleth_decade_within_budget(engine, decade_res_ids):
    t0 = time.perf_counter()
    result = engine.query_agreement_between_countries(
        country_code=C1, resolution_ids=decade_res_ids, average=True
    )
    elapsed = time.perf_counter() - t0
    assert result is not None
    budget = TIME_BUDGETS["choropleth_decade"]
    assert elapsed < budget, (
        f"choropleth 2010-2019 ({len(decade_res_ids)} res): {elapsed:.3f}s > {budget}s budget"
    )


def test_bilateral_full_within_budget(engine, all_res_ids):
    t0 = time.perf_counter()
    result = engine.query_agreement_between_countries(
        country_code=C1, resolution_ids=all_res_ids, average=False
    )
    elapsed = time.perf_counter() - t0
    assert result is not None
    budget = TIME_BUDGETS["bilateral_full"]
    assert elapsed < budget, (
        f"bilateral full ({len(all_res_ids)} res): {elapsed:.3f}s > {budget}s budget"
    )


def test_bilateral_post2000_within_budget(engine, post2000_res_ids):
    t0 = time.perf_counter()
    result = engine.query_agreement_between_countries(
        country_code=C1, resolution_ids=post2000_res_ids, average=False
    )
    elapsed = time.perf_counter() - t0
    assert result is not None
    budget = TIME_BUDGETS["bilateral_post2000"]
    assert elapsed < budget, (
        f"bilateral post-2000 ({len(post2000_res_ids)} res): {elapsed:.3f}s > {budget}s budget"
    )


def test_summary(engine, all_res_ids, post2000_res_ids, decade_res_ids, capsys):
    """Print a timing table for all scenarios; always passes."""
    scenarios = [
        ("choropleth full", all_res_ids, True, TIME_BUDGETS["choropleth_full"]),
        ("choropleth post-2000", post2000_res_ids, True, TIME_BUDGETS["choropleth_post2000"]),
        ("choropleth 2010-19", decade_res_ids, True, TIME_BUDGETS["choropleth_decade"]),
        ("bilateral full", all_res_ids, False, TIME_BUDGETS["bilateral_full"]),
        ("bilateral post-2000", post2000_res_ids, False, TIME_BUDGETS["bilateral_post2000"]),
    ]

    lines = [f"{'Scenario':<26} {'Res':>6}  {'Time (ms)':>10}  {'Budget (ms)':>12}  {'Pass?':>6}"]
    lines.append("-" * 68)

    for label, ids, avg, budget in scenarios:
        t0 = time.perf_counter()
        engine.query_agreement_between_countries(
            country_code=C1, resolution_ids=ids, average=avg
        )
        elapsed = time.perf_counter() - t0
        ok = "YES" if elapsed < budget else "NO"
        lines.append(
            f"{label:<26} {len(ids):>6}  {elapsed * 1000:>9.1f}ms  {budget * 1000:>10.0f}ms  {ok:>6}"
        )

    with capsys.disabled():
        print("\n" + "\n".join(lines))
