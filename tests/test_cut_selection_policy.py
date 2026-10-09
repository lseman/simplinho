"""End-to-end tests for the Python cut-selection callback."""

import os
import sys
from pathlib import Path

import numpy as np

BUILD_DIR = Path(os.environ.get("SIMPLINHO_BUILD_DIR", "build")).resolve()
sys.path.insert(0, str(BUILD_DIR))

import simplinho_bnb as snb  # noqa: E402
from simplinho import Model  # noqa: E402


def make_cut_mip():
    model = Model()
    variables = [
        model.add_var(f"x{i}", lb=0, ub=1, var_type=snb.VarType.Binary)
        for i in range(4)
    ]
    # Its LP relaxation has value 2.5, while the integer optimum is 2. The
    # enabled separators reliably produce violated root cuts for this fixture.
    model.add_constr(sum(2 * variable for variable in variables) <= 5)
    model.maximize(sum(variables))
    return model


def callback_options():
    options = snb.BranchAndBoundOptions()
    options.parallel_workers = 1
    options.use_rounding = False
    options.use_diving = False
    options.use_async_heuristics = False
    options.use_node_presolve = False
    options.max_root_cut_rounds = 3
    return options


def test_cut_callback_receives_batched_observations():
    received = []

    def cut_callback(obs):
        received.append(obs)
        return np.asarray(obs["efficacy"], dtype=np.float64)

    result = make_cut_mip().solve_mip(
        callback_options(), cut_callback=cut_callback
    )

    assert result.status == snb.Status.Optimal
    assert result.objective == 2.0
    assert received

    obs = received[0]
    candidate_count = len(obs["pool_indices"])
    assert candidate_count > 0
    for key in (
        "violation",
        "efficacy",
        "active_efficacy",
        "density_adjusted_efficacy",
        "dynamism",
        "fractional_focus",
        "objective_parallelism",
        "strength",
        "age",
        "times_used",
        "nnz",
        "cut_type",
        "rhs",
        "sense",
    ):
        assert len(obs[key]) == candidate_count

    assert obs["is_root"] is True
    assert obs["depth"] == 0
    assert obs["round"] >= 0
    assert len(obs["row_starts"]) == candidate_count + 1
    assert obs["row_starts"][0] == 0
    assert obs["row_starts"][-1] == len(obs["column_indices"])
    assert len(obs["column_indices"]) == len(obs["coefficients"])


def test_cut_callback_none_and_invalid_scores_fall_back():
    callbacks = [
        lambda obs: None,
        lambda obs: np.zeros(len(obs["efficacy"]) + 1),
        lambda obs: np.full(len(obs["efficacy"]), np.nan),
        lambda obs: "invalid",
    ]

    for callback in callbacks:
        result = make_cut_mip().solve_mip(
            callback_options(), cut_callback=callback
        )
        assert result.status == snb.Status.Optimal
        assert result.objective == 2.0


def test_cut_callback_exception_falls_back():
    call_count = 0

    def cut_callback(obs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("intentional cut callback error")

    result = make_cut_mip().solve_mip(
        callback_options(), cut_callback=cut_callback
    )

    assert call_count > 0
    assert result.status == snb.Status.Optimal
    assert result.objective == 2.0


def test_cut_callback_must_be_callable():
    try:
        make_cut_mip().solve_mip(callback_options(), cut_callback=42)
    except TypeError as error:
        assert "cut_callback must be callable" in str(error)
    else:
        raise AssertionError("non-callable cut_callback should be rejected")
