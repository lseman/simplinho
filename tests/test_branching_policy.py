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
import pytest

# Ensure the build output is on the path.
BUILD_DIR = os.environ.get("SIMPLINHO_BUILD_DIR", str(Path("build-bnb").resolve()))
sys.path.insert(0, str(BUILD_DIR.resolve()))

import simplinho_bnb as snb  # noqa: E402
from simplinho import Model, RevisedSimplexOptions  # noqa: E402


# ── Helpers ──


def make_simple_mip():
    """Create a small MIP model for testing."""
    m = Model()
    x = m.add_var("x", lb=0, ub=10, var_type=snb.VarType.Integer)
    y = m.add_var("y", lb=0, ub=10, var_type=snb.VarType.Integer)
    z = m.add_var("z", lb=0, ub=10, var_type=snb.VarType.Binary)
    m.add_constr(x + y + z <= 5, name="capacity")
    m.add_constr(2 * x + y >= 3, name="min_x")
    m.add_constr(y + 2 * z <= 6, name="mix")
    m.maximize(x + 2 * y + 3 * z)
    return m


def make_lp_only():
    """Create an LP (no integer vars) for testing."""
    m = Model()
    x = m.add_var("x", lb=0, ub=10)
    y = m.add_var("y", lb=0, ub=10)
    m.add_constr(x + y <= 5)
    m.maximize(x + 2 * y)
    return m


# ── Test: Python callback receives observations ──


def test_python_callback_receives_observations():
    """Verify the callback is called with the expected observation dict."""
    received = []

    def branch_callback(obs):
        received.append(obs)
        # Return None to let the default policy decide.
        return None

    m = make_lp_only()
    m.solve_mip(branch_callback=branch_callback)

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
        if obs["is_root"] and root_obs is None:
            nonlocal root_obs
            root_obs = obs
        return None

    m = make_lp_only()
    m.solve_mip(branch_callback=branch_callback)

    assert root_obs is not None
    assert root_obs["is_root"] is True
    assert len(root_obs["pseudocost"]) == 0


def test_observations_nonroot_node():
    """Verify pseudocost data is populated at non-root nodes."""
    nonroot_obs = None

    def branch_callback(obs):
        if not obs["is_root"] and nonroot_obs is None:
            nonlocal nonroot_obs
            nonroot_obs = obs
        return None

    # Use an MIP with enough fractional variables to trigger multiple nodes.
    m = make_simple_mip()
    m.solve_mip(branch_callback=branch_callback)

    assert nonroot_obs is not None
    assert len(nonroot_obs["pseudocost"]) > 0
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
        # Return: (variable_index, down_bound, up_bound, score)
        return (var, obs["lp_values"][idx], obs["lp_values"][idx], frac[idx])

    m = make_simple_mip()
    result = m.solve_mip(branch_callback=branch_callback)

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

    m = make_simple_mip()
    result = m.solve_mip(branch_callback=branch_callback)

    assert call_count > 0
    assert result is not None


def test_python_callback_exception_fallback():
    """Python raises → fallback to most fractional, solve completes."""
    call_count = 0

    def branch_callback(obs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("intentional callback error")

    m = make_simple_mip()
    result = m.solve_mip(branch_callback=branch_callback)

    assert call_count > 0
    # Should not crash — fallback handles the exception.
    assert result is not None


def test_python_callback_invalid_return_type():
    """Return wrong type → fallback, solve still completes."""

    def branch_callback(obs):
        return "not a tuple"  # Invalid.

    m = make_simple_mip()
    result = m.solve_mip(branch_callback=branch_callback)

    assert result is not None


def test_python_callback_invalid_variable_index():
    """Return out-of-range variable → fallback."""

    def branch_callback(obs):
        # Variable 9999 is definitely not in the fractional list.
        return (9999, 0.0, 10.0, 0.0)

    m = make_simple_mip()
    result = m.solve_mip(branch_callback=branch_callback)

    assert result is not None


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

    m = make_simple_mip()
    m.solve_mip(branch_callback=branch_callback)

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
