# %% [markdown]
# # Compare `simplinho` MIPs vs PuLP + HiGHS
#
# This notebook compares `simplinho` against PuLP with the HiGHS backend on two groups of models:
#
# - small hand-built MIPs for correctness
# - larger synthetic binary knapsack-style instances with `100`, `250`, `500`, and `1000` binary variables
#
# The larger section is useful for stress-testing search, heuristics, and cuts.

# %% [markdown]
# If you see an import error or a missing `BranchAndBoundOptions`, restart the kernel and rerun from the top. This notebook loads the local compiled extension directly from `build-local` / `build` / `build-verify` and avoids stale installed copies.

# %%
from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import pulp
except ImportError as exc:
    raise ImportError(
        "This notebook needs PuLP and HiGHS support. Install with `pip install pulp highspy pandas`."
    ) from exc


ROOT = Path.cwd().parent if Path.cwd().name == "tests" else Path.cwd()


def import_simplinho():
    required_attrs = ("BranchAndBoundOptions", "MIPSolution", "MIPStatus")
    errors = []

    for build_dir in ("build-local", "build", "build-verify"):
        candidate = ROOT / build_dir
        if not candidate.exists():
            continue

        module_files = sorted(candidate.glob("simplinho*.so"))
        if not module_files:
            continue

        module_path = module_files[0]

        try:
            sys.modules.pop("simplinho", None)
            spec = importlib.util.spec_from_file_location("simplinho", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not create import spec for {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules["simplinho"] = module
            spec.loader.exec_module(module)
            if all(hasattr(module, attr) for attr in required_attrs):
                return module, candidate
            errors.append(f"{build_dir}: loaded {module_path} but MIP API is missing")
        except Exception as exc:
            errors.append(f"{build_dir}: {type(exc).__name__}: {exc}")
        finally:
            sys.modules.pop("simplinho", None)

    raise ImportError(
        "could not find a built simplinho module with MIP bindings. Tried: "
        + "; ".join(errors)
    )


simplinho, BUILD_DIR = import_simplinho()

print("root:", ROOT)
print("build dir:", BUILD_DIR)
print("simplinho module:", simplinho.__file__)
print("has MIP API:", hasattr(simplinho, "BranchAndBoundOptions"))
print("PuLP version:", pulp.__version__)


# %%
def make_bnb_options(**overrides):
    options = simplinho.BranchAndBoundOptions()
    options.node_selection = simplinho.NodeSelectionStrategy.DepthFirst
    options.branching_strategy = simplinho.BranchingStrategy.StrongBranching
    options.diving_strategy = simplinho.DivingStrategy.Fractional
    options.use_feasibility_pump = True
    options.use_rens = True
    options.use_rins = True
    options.use_local_search = True
    options.use_cut_pool = True
    options.use_gomory_cuts = True
    options.use_cover_cuts = True
    options.max_cut_rounds_per_node = 10
    options.max_cuts_added_per_round = 20
    options.heuristic_subproblem_max_nodes = 128
    options.max_nodes = 1000000
    for key, value in overrides.items():
        setattr(options, key, value)
    return options


DEFAULT_OPTIONS = make_bnb_options()
LARGE_OPTIONS = make_bnb_options(
    branching_strategy=simplinho.BranchingStrategy.StrongBranching,
    heuristic_subproblem_max_nodes=96,
    max_cuts_added_per_round=12,
    verbose=False,
)


def mip_case(name, sense, variables, constraints):
    return {
        "name": name,
        "sense": sense,
        "variables": variables,
        "constraints": constraints,
    }


def status_name(status):
    try:
        return simplinho.mip_status_to_string(status)
    except Exception:
        return str(status)


def build_simplinho_model(case):
    model = simplinho.Model()
    vars_by_name = {}

    for spec in case["variables"]:
        var_type = spec.get("type", "continuous")
        lb = spec.get("lb", 0.0)
        ub = spec.get("ub", math.inf)
        obj = spec.get("obj", 0.0)

        if var_type == "binary":
            var = model.add_binary_var(spec["name"], obj=obj)
        elif var_type == "integer":
            var = model.add_integer_var(spec["name"], lb=lb, ub=ub, obj=obj)
        else:
            var = model.add_var(spec["name"], lb=lb, ub=ub, obj=obj)

        vars_by_name[spec["name"]] = var

    for row in case["constraints"]:
        expr = 0.0
        for name, coeff in row["coeffs"].items():
            expr = expr + coeff * vars_by_name[name]

        if row["sense"] == "<=":
            model.add_constr(expr <= row["rhs"], name=row["name"])
        elif row["sense"] == ">=":
            model.add_constr(expr >= row["rhs"], name=row["name"])
        else:
            model.add_constr(expr == row["rhs"], name=row["name"])

    objective = 0.0
    for spec in case["variables"]:
        if abs(spec.get("obj", 0.0)) > 0.0:
            objective = objective + spec["obj"] * vars_by_name[spec["name"]]

    if case["sense"] == "max":
        model.maximize(objective)
    else:
        model.minimize(objective)

    return model, vars_by_name


def solve_with_simplinho(case, options=DEFAULT_OPTIONS):
    model, vars_by_name = build_simplinho_model(case)
    start = time.perf_counter()
    solution = model.solve_mip(options)
    elapsed = time.perf_counter() - start
    values = {}
    if solution.has_solution:
        for name in vars_by_name:
            values[name] = float(solution.value(name))

    return {
        "status": status_name(solution.status),
        "obj": float(solution.obj) if solution.has_solution else math.nan,
        "values": values,
        "solve_seconds": elapsed,
        "node_count": int(solution.node_count),
        "lp_iterations": int(solution.lp_iterations),
        "heuristic_successes": int(solution.heuristic_successes),
        "cuts_generated": int(solution.cuts_generated),
        "cuts_applied": int(solution.cuts_applied),
        "cut_pool_size": int(solution.cut_pool_size),
        "raw": solution,
    }


def solve_with_pulp_highs(case, msg=False):
    prob_sense = pulp.LpMaximize if case["sense"] == "max" else pulp.LpMinimize
    prob = pulp.LpProblem(case["name"], prob_sense)
    vars_by_name = {}

    for spec in case["variables"]:
        var_type = spec.get("type", "continuous")
        cat = {
            "continuous": pulp.LpContinuous,
            "integer": pulp.LpInteger,
            "binary": pulp.LpBinary,
        }[var_type]
        lb = spec.get("lb", 0.0)
        ub = spec.get("ub", None)
        low_bound = None if lb is None or math.isinf(lb) and lb < 0 else float(lb)
        up_bound = None if ub is None or math.isinf(ub) else float(ub)
        vars_by_name[spec["name"]] = pulp.LpVariable(
            spec["name"],
            lowBound=low_bound,
            upBound=up_bound,
            cat=cat,
        )

    prob += pulp.lpSum(
        float(spec.get("obj", 0.0)) * vars_by_name[spec["name"]]
        for spec in case["variables"]
    )

    for row in case["constraints"]:
        expr = pulp.lpSum(
            float(coeff) * vars_by_name[name] for name, coeff in row["coeffs"].items()
        )
        if row["sense"] == "<=":
            prob += expr <= float(row["rhs"]), row["name"]
        elif row["sense"] == ">=":
            prob += expr >= float(row["rhs"]), row["name"]
        else:
            prob += expr == float(row["rhs"]), row["name"]

    start = time.perf_counter()
    prob.solve(pulp.HiGHS(msg=msg))
    elapsed = time.perf_counter() - start

    return {
        "status": pulp.LpStatus[prob.status],
        "obj": float(pulp.value(prob.objective)) if prob.status == 1 else math.nan,
        "values": {name: float(pulp.value(var)) for name, var in vars_by_name.items()},
        "solve_seconds": elapsed,
        "raw": prob,
    }


def compare_case(case, atol=1e-7, compare_variables=True, options=DEFAULT_OPTIONS):
    simplinho_res = solve_with_simplinho(case, options=options)
    pulp_res = solve_with_pulp_highs(case)

    all_names = [spec["name"] for spec in case["variables"]]
    max_var_diff = math.nan
    close_variables = None

    if compare_variables:
        max_var_diff = 0.0
        for name in all_names:
            sx = simplinho_res["values"].get(name, math.nan)
            px = pulp_res["values"].get(name, math.nan)
            if math.isfinite(sx) and math.isfinite(px):
                max_var_diff = max(max_var_diff, abs(sx - px))
        close_variables = max_var_diff <= atol

    obj_close = (
        math.isfinite(simplinho_res["obj"])
        and math.isfinite(pulp_res["obj"])
        and abs(simplinho_res["obj"] - pulp_res["obj"]) <= atol
    )

    return {
        "case": case["name"],
        "sense": case["sense"],
        "num_vars": len(case["variables"]),
        "num_constraints": len(case["constraints"]),
        "simplinho_status": simplinho_res["status"],
        "pulp_status": pulp_res["status"],
        "simplinho_obj": simplinho_res["obj"],
        "pulp_obj": pulp_res["obj"],
        "obj_abs_diff": abs(simplinho_res["obj"] - pulp_res["obj"])
        if math.isfinite(simplinho_res["obj"]) and math.isfinite(pulp_res["obj"])
        else math.nan,
        "max_var_abs_diff": max_var_diff,
        "simplinho_seconds": simplinho_res["solve_seconds"],
        "pulp_seconds": pulp_res["solve_seconds"],
        "node_count": simplinho_res["node_count"],
        "lp_iterations": simplinho_res["lp_iterations"],
        "heuristic_successes": simplinho_res["heuristic_successes"],
        "cuts_generated": simplinho_res["cuts_generated"],
        "cuts_applied": simplinho_res["cuts_applied"],
        "same_status": simplinho_res["status"].lower() == pulp_res["status"].lower(),
        "close_objective": obj_close,
        "close_variables": close_variables,
        "simplinho_result": simplinho_res,
        "pulp_result": pulp_res,
    }


def make_large_binary_knapsack_case(n):
    variables = []
    coeffs_1 = {}
    coeffs_2 = {}
    coeffs_3 = {}
    coeffs_4 = {}
    weight_sum_1 = 0.0
    weight_sum_2 = 0.0
    weight_sum_3 = 0.0

    for i in range(n):
        name = f"x_{i}"
        profit = float(((17 * i + 13) % 97) + 10)
        w1 = float(((11 * i + 7) % 29) + 1)
        w2 = float(((19 * i + 5) % 31) + 1)
        w3 = float(((23 * i + 3) % 37) + 1)
        g = i % 10

        variables.append({"name": name, "type": "binary", "obj": profit})
        coeffs_1[name] = w1
        coeffs_2[name] = w2
        coeffs_3[name] = w3
        coeffs_4[name] = 1.0 if g in (0, 1, 2, 3) else 0.0
        weight_sum_1 += w1
        weight_sum_2 += w2
        weight_sum_3 += w3

    constraints = [
        {
            "name": "cap_1",
            "coeffs": coeffs_1,
            "sense": "<=",
            "rhs": 0.35 * weight_sum_1,
        },
        {
            "name": "cap_2",
            "coeffs": coeffs_2,
            "sense": "<=",
            "rhs": 0.33 * weight_sum_2,
        },
        {
            "name": "cap_3",
            "coeffs": coeffs_3,
            "sense": "<=",
            "rhs": 0.31 * weight_sum_3,
        },
        {
            "name": "group_cap",
            "coeffs": coeffs_4,
            "sense": "<=",
            "rhs": max(8.0, 0.18 * n),
        },
    ]

    return mip_case(
        name=f"large_binary_multiknapsack_{n}",
        sense="max",
        variables=variables,
        constraints=constraints,
    )


# %% [markdown]
# ## Small Correctness Cases

# %%
# small_cases = [
#     mip_case(
#         name="binary_cover_knapsack",
#         sense="max",
#         variables=[
#             {"name": "x1", "type": "binary", "obj": 10.0},
#             {"name": "x2", "type": "binary", "obj": 7.0},
#             {"name": "x3", "type": "binary", "obj": 5.0},
#         ],
#         constraints=[
#             {"name": "capacity", "coeffs": {"x1": 4.0, "x2": 3.0, "x3": 2.0}, "sense": "<=", "rhs": 5.0},
#         ],
#     ),
#     mip_case(
#         name="integer_production",
#         sense="max",
#         variables=[
#             {"name": "x", "type": "integer", "lb": 0.0, "ub": 10.0, "obj": 5.0},
#             {"name": "y", "type": "integer", "lb": 0.0, "ub": 10.0, "obj": 4.0},
#         ],
#         constraints=[
#             {"name": "labor", "coeffs": {"x": 6.0, "y": 4.0}, "sense": "<=", "rhs": 24.0},
#             {"name": "material", "coeffs": {"x": 1.0, "y": 2.0}, "sense": "<=", "rhs": 6.0},
#         ],
#     ),
#     mip_case(
#         name="set_packing_ring",
#         sense="max",
#         variables=[
#             {"name": "a", "type": "binary", "obj": 8.0},
#             {"name": "b", "type": "binary", "obj": 6.0},
#             {"name": "c", "type": "binary", "obj": 5.0},
#             {"name": "d", "type": "binary", "obj": 4.0},
#         ],
#         constraints=[
#             {"name": "ab", "coeffs": {"a": 1.0, "b": 1.0}, "sense": "<=", "rhs": 1.0},
#             {"name": "bc", "coeffs": {"b": 1.0, "c": 1.0}, "sense": "<=", "rhs": 1.0},
#             {"name": "cd", "coeffs": {"c": 1.0, "d": 1.0}, "sense": "<=", "rhs": 1.0},
#             {"name": "da", "coeffs": {"d": 1.0, "a": 1.0}, "sense": "<=", "rhs": 1.0},
#         ],
#     ),
#     mip_case(
#         name="mixed_integer_with_continuous",
#         sense="max",
#         variables=[
#             {"name": "x", "type": "integer", "lb": 0.0, "ub": 4.0, "obj": 6.0},
#             {"name": "y", "type": "binary", "obj": 4.0},
#             {"name": "z", "type": "continuous", "lb": 0.0, "ub": 3.0, "obj": 1.5},
#         ],
#         constraints=[
#             {"name": "c1", "coeffs": {"x": 2.0, "y": 1.0, "z": 1.0}, "sense": "<=", "rhs": 7.0},
#             {"name": "c2", "coeffs": {"x": 1.0, "z": 1.0}, "sense": "<=", "rhs": 5.0},
#         ],
#     ),
# ]

# len(small_cases)


# %%
# small_results = [compare_case(case, compare_variables=True, options=DEFAULT_OPTIONS) for case in small_cases]

# small_summary = pd.DataFrame(
#     [
#         {
#             "case": row["case"],
#             "num_vars": row["num_vars"],
#             "num_constraints": row["num_constraints"],
#             "simplinho_status": row["simplinho_status"],
#             "pulp_status": row["pulp_status"],
#             "simplinho_obj": row["simplinho_obj"],
#             "pulp_obj": row["pulp_obj"],
#             "obj_abs_diff": row["obj_abs_diff"],
#             "max_var_abs_diff": row["max_var_abs_diff"],
#             "simplinho_seconds": row["simplinho_seconds"],
#             "pulp_seconds": row["pulp_seconds"],
#             "node_count": row["node_count"],
#             "lp_iterations": row["lp_iterations"],
#             "heuristic_successes": row["heuristic_successes"],
#             "cuts_generated": row["cuts_generated"],
#             "cuts_applied": row["cuts_applied"],
#             "same_status": row["same_status"],
#             "close_objective": row["close_objective"],
#             "close_variables": row["close_variables"],
#         }
#         for row in small_results
#     ]
# )

# small_summary


# %%
# # assert small_summary["same_status"].all()
# # assert small_summary["close_objective"].all()
# # assert small_summary["close_variables"].all()

# small_summary[["case", "node_count", "heuristic_successes", "cuts_generated", "cuts_applied"]]


# %%
# detail_case = small_results[0]

# print("case:", detail_case["case"])
# print("simplinho:", detail_case["simplinho_result"])
# print("pulp/highs:", detail_case["pulp_result"])


# %% [markdown]
# ## Large Binary Benchmarks
#
# These instances are deterministic multi-constraint binary knapsack models. For large cases we compare status and objective, and we track runtime and search statistics. We do not require exact variable-by-variable agreement because multiple optima can appear in larger combinatorial models.

# %%
large_sizes = [100]
large_cases = [make_large_binary_knapsack_case(n) for n in large_sizes]

[
    (case["name"], len(case["variables"]), len(case["constraints"]))
    for case in large_cases
]


# %%
large_results = [
    compare_case(case, compare_variables=False, options=LARGE_OPTIONS)
    for case in large_cases
]

large_summary = pd.DataFrame(
    [
        {
            "case": row["case"],
            "num_vars": row["num_vars"],
            "num_constraints": row["num_constraints"],
            "simplinho_status": row["simplinho_status"],
            "pulp_status": row["pulp_status"],
            "simplinho_obj": row["simplinho_obj"],
            "pulp_obj": row["pulp_obj"],
            "obj_abs_diff": row["obj_abs_diff"],
            "simplinho_seconds": row["simplinho_seconds"],
            "pulp_seconds": row["pulp_seconds"],
            "speed_ratio_pulp_over_simplinho": row["pulp_seconds"]
            / row["simplinho_seconds"]
            if row["simplinho_seconds"] > 0
            else math.nan,
            "node_count": row["node_count"],
            "lp_iterations": row["lp_iterations"],
            "heuristic_successes": row["heuristic_successes"],
            "cuts_generated": row["cuts_generated"],
            "cuts_applied": row["cuts_applied"],
            "same_status": row["same_status"],
            "close_objective": row["close_objective"],
        }
        for row in large_results
    ]
)

print(large_summary)


# %%
# assert large_summary["same_status"].all()
# assert large_summary["close_objective"].all()

large_summary[
    [
        "num_vars",
        "simplinho_seconds",
        "pulp_seconds",
        "node_count",
        "cuts_generated",
        "heuristic_successes",
    ]
]
