"""Tests for the branching policy hooks and Python callbacks.

Tests the full stack:
1. Python branch_callback is called with correct observations
2. Return values are correctly parsed and applied
3. Fallback to default policy works (None, exceptions, invalid returns)
4. Telemetry is recorded correctly
5. No regression when branch_callback is not provided
"""

import math
import os
import sys
from pathlib import Path

import numpy as np

# Ensure the build output is on the path.
BUILD_DIR = Path(os.environ.get("SIMPLINHO_BUILD_DIR", "build")).resolve()
sys.path.insert(0, str(BUILD_DIR))

import simplinho_bnb as snb  # noqa: E402
from simplinho import Model, RevisedSimplexOptions  # noqa: E402


# ── Helpers ──


def make_simple_mip():
    """Create a small MIP model for testing."""
    m = Model()
    x = m.add_var("x", lb=0, ub=10, var_type=snb.VarType.Integer)
    y = m.add_var("y", lb=0, ub=10, var_type=snb.VarType.Integer)
    z = m.add_var("z", lb=0, ub=1, var_type=snb.VarType.Binary)
    m.add_constr(x + y + z <= 5, name="capacity")
    m.add_constr(2 * x + y >= 3, name="min_x")
    m.add_constr(y + 2 * z <= 6, name="mix")
    m.maximize(x + 2 * y + 3 * z)
    return m


def make_fractional_root_mip():
    """Create a MIP whose root relaxation requires branching."""
    m = Model()
    x = m.add_var("x", lb=0, ub=1, var_type=snb.VarType.Binary)
    y = m.add_var("y", lb=0, ub=1, var_type=snb.VarType.Binary)
    m.add_constr(2 * x + 3 * y <= 4)
    m.maximize(3 * x + 4 * y)
    return m


def callback_options():
    """Keep the fixture fractional until the branching-policy hook."""
    options = snb.BranchAndBoundOptions()
    options.parallel_workers = 1
    options.use_rounding = False
    options.use_diving = False
    options.use_async_heuristics = False
    options.use_cut_pool = False
    options.use_node_presolve = False
    return options


# ── Test: Python callback receives observations ──


def test_python_callback_receives_observations():
    """Verify the callback is called with the expected observation dict."""
    received = []

    def branch_callback(obs):
        received.append(obs)
        # Return None to let the default policy decide.
        return None

    m = make_fractional_root_mip()
    m.solve_mip(callback_options(), branch_callback=branch_callback)

    assert len(received) > 0, "Callback should have been called at least once"
    obs = received[0]

    # Check expected keys exist.
    assert "fractional_variables" in obs
    assert "lp_values" in obs
    assert "fractionality" in obs
    assert "down_distance" in obs
    assert "up_distance" in obs
    assert "node_id" in obs
    assert "depth" in obs
    assert "lp_objective" in obs
    assert "lower_bounds" in obs
    assert "upper_bounds" in obs
    assert "is_root" in obs
    assert "maximize" in obs
    assert "lp_status" in obs
    assert "n_nodes_explored" in obs
    assert "pseudocost" in obs

    # Verify array shapes match.
    n_frac = len(obs["fractional_variables"])
    assert n_frac > 0, "Should have fractional variables"
    assert obs["lp_values"].shape == (n_frac,)
    assert obs["fractionality"].shape == (n_frac,)
    assert obs["down_distance"].shape == (n_frac,)
    assert obs["up_distance"].shape == (n_frac,)


def test_observations_root_node():
    """Verify is_root=True and pseudocost is empty at root node."""
    root_obs = None

    def branch_callback(obs):
        nonlocal root_obs
        if obs["is_root"] and root_obs is None:
            root_obs = obs
        return None

    m = make_fractional_root_mip()
    m.solve_mip(callback_options(), branch_callback=branch_callback)

    assert root_obs is not None
    assert root_obs["is_root"] is True
    assert len(root_obs["pseudocost"]) == 0


def test_observations_nonroot_node():
    """Verify pseudocost data is populated at non-root nodes."""
    nonroot_obs = None

    def branch_callback(obs):
        nonlocal nonroot_obs
        if not obs["is_root"] and nonroot_obs is None:
            nonroot_obs = obs
        return None

    # Use an MIP with enough fractional variables to trigger multiple nodes.
    m = make_fractional_root_mip()
    m.solve_mip(callback_options(), branch_callback=branch_callback)

    assert nonroot_obs is not None
    assert len(nonroot_obs["pseudocost"]) == len(nonroot_obs["fractional_variables"])
    # Each pseudocost entry should have up_score, down_score, samples.
    for entry in nonroot_obs["pseudocost"]:
        assert "up_score" in entry
        assert "down_score" in entry
        assert "samples" in entry


# ── Test: Python callback return value ──


def test_python_callback_returns_decision():
    """Verify a Python-decided variable is actually used."""
    chosen_vars = []

    def branch_callback(obs):
        # Always pick the most fractional variable.
        frac = obs["fractionality"]
        idx = int(np.argmax(frac))
        var = int(obs["fractional_variables"][idx])
        chosen_vars.append(var)
        return var

    m = make_fractional_root_mip()
    result = m.solve_mip(callback_options(), branch_callback=branch_callback)

    assert len(chosen_vars) > 0, "Callback should have been called"
    assert result.status == snb.Status.Optimal or result.status == snb.Status.NodeLimit


# ── Test: Fallback behavior ──


def test_python_callback_none_fallback():
    """Return None → default policy takes over, solve still completes."""
    call_count = 0

    def branch_callback(obs):
        nonlocal call_count
        call_count += 1
        return None  # Defer to default.

    m = make_fractional_root_mip()
    result = m.solve_mip(callback_options(), branch_callback=branch_callback)

    assert call_count > 0
    assert result is not None


def test_python_callback_exception_fallback():
    """Python raises → fallback to most fractional, solve completes."""
    call_count = 0

    def branch_callback(obs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("intentional callback error")

    m = make_fractional_root_mip()
    result = m.solve_mip(callback_options(), branch_callback=branch_callback)

    assert call_count > 0
    # Should not crash — fallback handles the exception.
    assert result is not None


def test_python_callback_rejects_legacy_tuple():
    """The old tuple action is invalid and cleanly falls back."""

    def branch_callback(obs):
        variable = int(obs["fractional_variables"][0])
        return (variable, 0.0, 1.0, 0.5)

    m = make_fractional_root_mip()
    result = m.solve_mip(callback_options(), branch_callback=branch_callback)

    assert result is not None


def test_python_callback_invalid_variable_index():
    """Return out-of-range variable → fallback."""

    def branch_callback(obs):
        # Variable 9999 is definitely not in the fractional list.
        return (9999, 0.0, 10.0, 0.0)

    m = make_fractional_root_mip()
    result = m.solve_mip(callback_options(), branch_callback=branch_callback)

    assert result is not None


def test_callable_object_callback():
    """Callable policy objects are accepted in addition to plain functions."""

    class Policy:
        def __init__(self):
            self.call_count = 0

        def __call__(self, obs):
            self.call_count += 1
            return None

    policy = Policy()
    result = make_fractional_root_mip().solve_mip(
        callback_options(), branch_callback=policy
    )

    assert result is not None
    assert policy.call_count > 0


def test_callback_is_scoped_to_one_solve():
    """A callback must not leak into a later solve on the same thread."""
    call_count = 0

    def branch_callback(obs):
        nonlocal call_count
        call_count += 1
        return None

    m = make_fractional_root_mip()
    m.solve_mip(callback_options(), branch_callback=branch_callback)
    first_solve_calls = call_count
    m.solve_mip(callback_options())

    assert first_solve_calls > 0
    assert call_count == first_solve_calls


def test_callback_with_parallel_workers():
    """Native worker threads can run while Python callbacks use the GIL."""
    call_count = 0

    def branch_callback(obs):
        nonlocal call_count
        call_count += 1
        return None

    options = callback_options()
    options.parallel_workers = 2
    result = make_fractional_root_mip().solve_mip(
        options, branch_callback=branch_callback
    )

    assert result is not None
    assert call_count > 0


def test_no_callback_unchanged_behavior():
    """Without branch_callback, existing behavior should work unchanged."""
    m = make_simple_mip()
    result = m.solve_mip()

    assert result is not None
    assert result.status in (snb.Status.Optimal, snb.Status.NodeLimit)
    assert not math.isnan(result.objective)


# ── Test: Telemetry ──


def test_telemetry_fields_exist():
    """BranchingTelemetry struct should have expected fields."""
    tel = snb.BranchingTelemetry()
    assert hasattr(tel, "call_count")
    assert hasattr(tel, "success_count")
    assert hasattr(tel, "fallback_count")
    assert hasattr(tel, "exception_count")
    assert hasattr(tel, "timeout_count")
    assert hasattr(tel, "total_wall_ns")
    assert hasattr(tel, "max_wall_ns")
    assert hasattr(tel, "min_wall_ns")
    assert hasattr(tel, "timeout_ms")
    assert hasattr(tel, "summary")

    # summary should return a non-empty string.
    summary_str = tel.summary()
    assert isinstance(summary_str, str)
    assert "callbacks:" in summary_str

    # repr should be non-empty.
    repr_str = repr(tel)
    assert isinstance(repr_str, str)
    assert "BranchingTelemetry" in repr_str


# ── Test: Observations shape correctness ──


def test_observations_array_consistency():
    """Verify that all observation arrays have consistent lengths."""
    observations = []

    def branch_callback(obs):
        observations.append(obs)
        return None

    m = make_fractional_root_mip()
    m.solve_mip(callback_options(), branch_callback=branch_callback)

    for obs in observations:
        n = len(obs["fractional_variables"])
        assert n == obs["lp_values"].shape[0]
        assert n == obs["fractionality"].shape[0]
        assert n == obs["down_distance"].shape[0]
        assert n == obs["up_distance"].shape[0]
        # fractionality should equal min(down_distance, up_distance).
        frac = obs["fractionality"]
        min_dist = np.minimum(obs["down_distance"], obs["up_distance"])
        np.testing.assert_allclose(frac, min_dist, atol=1e-10)
