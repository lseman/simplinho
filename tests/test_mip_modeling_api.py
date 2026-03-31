import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def import_simplinho():
    for build_dir in ("build-local", "build", "build-verify"):
        candidate = ROOT / build_dir
        if not candidate.exists():
            continue
        if not any(candidate.glob("*.so")):
            continue
        sys.path.insert(0, str(candidate))
        try:
            import simplinho

            return simplinho
        except ImportError:
            continue
    raise ImportError("could not find a built simplinho module")


try:
    simplinho = import_simplinho()
    HAS_SIMPLINHO = True
except ImportError:
    simplinho = None
    HAS_SIMPLINHO = False


@unittest.skipUnless(HAS_SIMPLINHO, "requires a locally built simplinho module")
class MIPModelingApiTests(unittest.TestCase):
    def test_integer_and_binary_variable_types(self):
        model = simplinho.Model()
        x = model.add_var("x", lb=0.0, ub=5.0, var_type=simplinho.VarType.Integer)
        y = model.add_binary_var("y", obj=1.0)

        self.assertEqual(x.type, simplinho.VarType.Integer)
        self.assertEqual(y.type, simplinho.VarType.Binary)
        self.assertTrue(math.isclose(y.lb, 0.0, rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(y.ub, 1.0, rel_tol=0.0, abs_tol=1e-12))

        z = model.add_var("z", lb=0.0, ub=1.0)
        z.type = simplinho.VarType.Binary
        self.assertEqual(z.type, simplinho.VarType.Binary)

        with self.assertRaises(ValueError):
            model.add_var("bad", lb=0.0, ub=2.0, var_type=simplinho.VarType.Binary)

    def _build_binary_branching_model(self):
        model = simplinho.Model()
        x = model.add_binary_var("x")
        y = model.add_binary_var("y")
        model.add_constr(2.0 * x + 2.0 * y <= 3.0, name="cap")
        model.maximize(x + y)
        return model, x, y

    def _build_integer_pair_branching_model(self):
        model = simplinho.Model()
        x = model.add_integer_var("x", lb=0.0, ub=10.0)
        y = model.add_integer_var("y", lb=0.0, ub=10.0)
        model.add_constr(2.0 * x + 2.0 * y <= 5.0, name="cap")
        model.maximize(x + y)
        return model, x, y

    def _build_large_binary_multiknapsack_model(self, n=25):
        model = simplinho.Model()
        vars_ = []
        coeffs_1 = []
        coeffs_2 = []
        coeffs_3 = []
        coeffs_4 = []

        for i in range(n):
            name = f"x_{i}"
            profit = float(((17 * i + 13) % 97) + 10)
            w1 = float(((11 * i + 7) % 29) + 1)
            w2 = float(((19 * i + 5) % 31) + 1)
            w3 = float(((23 * i + 3) % 37) + 1)
            group = 1.0 if i % 10 in (0, 1, 2, 3) else 0.0

            vars_.append(model.add_binary_var(name, obj=profit))
            coeffs_1.append(w1)
            coeffs_2.append(w2)
            coeffs_3.append(w3)
            coeffs_4.append(group)

        model.add_constr(
            sum(coeffs_1[i] * vars_[i] for i in range(n)) <= 0.35 * sum(coeffs_1),
            name="cap_1",
        )
        model.add_constr(
            sum(coeffs_2[i] * vars_[i] for i in range(n)) <= 0.33 * sum(coeffs_2),
            name="cap_2",
        )
        model.add_constr(
            sum(coeffs_3[i] * vars_[i] for i in range(n)) <= 0.31 * sum(coeffs_3),
            name="cap_3",
        )
        model.add_constr(
            sum(coeffs_4[i] * vars_[i] for i in range(n)) <= max(8.0, 0.18 * n),
            name="group_cap",
        )
        model.maximize(
            sum(float(((17 * i + 13) % 97) + 10) * vars_[i] for i in range(n))
        )
        return model, vars_

    def test_solve_mip_depth_first(self):
        model, x, y = self._build_binary_branching_model()
        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.DepthFirst

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution.best_bound, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(
            math.isclose(
                solution.root_relaxation_objective, 1.5, rel_tol=0.0, abs_tol=1e-8
            )
        )
        self.assertGreaterEqual(solution.node_count, 3)
        self.assertTrue(
            math.isclose(solution.value(x), round(solution.value(x)), rel_tol=0.0, abs_tol=1e-8)
        )
        self.assertTrue(
            math.isclose(solution.value(y), round(solution.value(y)), rel_tol=0.0, abs_tol=1e-8)
        )
        self.assertGreaterEqual(len(solution.tree_nodes), 3)
        self.assertEqual(solution.tree_nodes[0].parent_id, -1)

    def test_solve_mip_hybrid_node_selection(self):
        model, _, _ = self._build_binary_branching_model()
        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.Hybrid
        options.hybrid_depth_bias = 2

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.node_count, 1)
        self.assertTrue(math.isfinite(solution.tree_nodes[0].estimate))

    def test_solve_mip_best_estimate_node_selection(self):
        model, _, _ = self._build_binary_branching_model()
        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestEstimate
        options.branching_strategy = simplinho.BranchingStrategy.PseudoCost

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.node_count, 1)
        self.assertTrue(math.isfinite(solution.tree_nodes[0].estimate))

    def test_solve_mip_with_progress_logging_enabled(self):
        model, _, _ = self._build_binary_branching_model()
        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.verbose = True
        options.log_frequency = 1

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))

    def test_solve_mip_parallel_workers(self):
        model, _, _ = self._build_integer_pair_branching_model()
        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestEstimate
        options.branching_strategy = simplinho.BranchingStrategy.PseudoCost
        options.parallel_workers = 2

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 2.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isfinite(solution.tree_nodes[0].estimate))

    def test_solve_mip_branching_strategies(self):
        for branching_strategy in (
            simplinho.BranchingStrategy.MostFractional,
            simplinho.BranchingStrategy.PseudoCost,
            simplinho.BranchingStrategy.StrongBranching,
        ):
            with self.subTest(branching_strategy=branching_strategy):
                model, x, y = self._build_integer_pair_branching_model()

                options = simplinho.BranchAndBoundOptions()
                options.node_selection = simplinho.NodeSelectionStrategy.BestBound
                options.branching_strategy = branching_strategy

                solution = model.solve_mip(options)

                self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
                self.assertTrue(solution.has_solution)
                self.assertTrue(math.isclose(solution.obj, 2.0, rel_tol=0.0, abs_tol=1e-8))
                self.assertTrue(
                    math.isclose(solution.best_bound, 2.0, rel_tol=0.0, abs_tol=1e-8)
                )
                self.assertTrue(
                    math.isclose(
                        solution.root_relaxation_objective, 2.5, rel_tol=0.0, abs_tol=1e-8
                    )
                )
                self.assertTrue(
                    math.isclose(
                        solution.value(x) + solution.value(y),
                        2.0,
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    )
                )
                self.assertGreaterEqual(len(solution.tree_nodes), 1)

    def test_branch_and_bound_node_limit_alias(self):
        model, _, _ = self._build_integer_pair_branching_model()
        options = simplinho.BranchAndBoundOptions()
        options.node_limit = 1

        self.assertEqual(options.max_nodes, 1)
        self.assertEqual(options.node_limit, 1)

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.NodeLimit)

    def test_branch_and_bound_cut_engine_options_round_trip(self):
        options = simplinho.BranchAndBoundOptions()
        options.use_conflict_cuts = False
        options.max_conflict_cuts_per_round = 2
        options.max_cuts_per_type = 3
        options.cut_max_parallelism = 0.9

        self.assertFalse(options.use_conflict_cuts)
        self.assertEqual(options.max_conflict_cuts_per_round, 2)
        self.assertEqual(options.max_cuts_per_type, 3)
        self.assertTrue(math.isclose(options.cut_max_parallelism, 0.9, rel_tol=0.0, abs_tol=1e-12))

    def test_strong_branching_changes_root_branch_choice(self):
        model = simplinho.Model()
        vars_ = [model.add_integer_var(f"x_{i}", lb=0.0, ub=5.0) for i in range(4)]
        model.add_constr(
            3.0 * vars_[0] + 1.0 * vars_[1] + 3.0 * vars_[2] + 5.0 * vars_[3] <= 4.0,
            name="c0",
        )
        model.add_constr(
            1.0 * vars_[0] + 3.0 * vars_[1] + 2.0 * vars_[2] + 2.0 * vars_[3] <= 8.0,
            name="c1",
        )
        model.add_constr(
            3.0 * vars_[0] + 3.0 * vars_[1] + 4.0 * vars_[2] + 5.0 * vars_[3] <= 6.0,
            name="c2",
        )
        model.maximize(5.0 * vars_[0] + 2.0 * vars_[1] + 8.0 * vars_[2] + 4.0 * vars_[3])

        def solve_with(branching_strategy):
            options = simplinho.BranchAndBoundOptions()
            options.node_selection = simplinho.NodeSelectionStrategy.BestBound
            options.branching_strategy = branching_strategy
            options.parallel_workers = 1
            options.use_cut_pool = False
            options.use_feasibility_pump = False
            options.use_rens = False
            options.use_rins = False
            options.use_local_search = False
            options.use_local_branching = False
            return model.solve_mip(options)

        most_fractional = solve_with(simplinho.BranchingStrategy.MostFractional)
        strong_branching = solve_with(simplinho.BranchingStrategy.StrongBranching)

        self.assertEqual(most_fractional.status, simplinho.MIPStatus.Optimal)
        self.assertEqual(strong_branching.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(most_fractional.has_solution)
        self.assertTrue(strong_branching.has_solution)
        self.assertGreaterEqual(len(most_fractional.tree_nodes), 1)
        self.assertGreaterEqual(len(strong_branching.tree_nodes), 1)
        self.assertEqual(most_fractional.tree_nodes[0].branch_var, 1)
        self.assertEqual(strong_branching.tree_nodes[0].branch_var, 0)

    def test_mip_node_presolve_tightens_binary_bounds(self):
        model = simplinho.Model()
        x = model.add_binary_var("x")
        model.add_constr(x >= 0.6, name="force_one")
        model.minimize(x)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(
            math.isclose(
                solution.root_relaxation_objective, 1.0, rel_tol=0.0, abs_tol=1e-8
            )
        )
        self.assertEqual(solution.node_count, 1)
        self.assertTrue(math.isclose(solution.value(x), 1.0, rel_tol=0.0, abs_tol=1e-8))

    def test_mip_node_presolve_closes_reverse_propagation_chain(self):
        model = simplinho.Model()
        x1 = model.add_binary_var("x1")
        x2 = model.add_binary_var("x2")
        x3 = model.add_binary_var("x3")
        x4 = model.add_binary_var("x4")

        # Add the dependent rows in reverse order so node propagation needs
        # multiple rounds to reach a fixed point.
        model.add_constr(x4 - x3 >= 0.0, name="link_43")
        model.add_constr(x3 - x2 >= 0.0, name="link_32")
        model.add_constr(x2 - x1 >= 0.0, name="link_21")
        model.add_constr(x1 >= 0.6, name="force_x1")
        model.minimize(x4)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(
            math.isclose(
                solution.root_relaxation_objective, 1.0, rel_tol=0.0, abs_tol=1e-8
            )
        )
        self.assertEqual(solution.node_count, 1)
        self.assertTrue(math.isclose(solution.value(x1), 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution.value(x2), 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution.value(x3), 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution.value(x4), 1.0, rel_tol=0.0, abs_tol=1e-8))

    def test_root_mip_presolve_detects_infeasible_integer_bounds(self):
        model = simplinho.Model()
        x = model.add_integer_var("x", lb=0.2, ub=0.8)
        model.minimize(0.0 * x)

        solution = model.solve_mip()

        self.assertEqual(solution.status, simplinho.MIPStatus.Infeasible)
        self.assertFalse(solution.has_solution)
        self.assertEqual(solution.node_count, 0)

    def test_root_mip_presolve_aggregates_implied_free_continuous_variable(self):
        model = simplinho.Model()
        x = model.add_var("x", lb=0.0, ub=1.0)
        y = model.add_binary_var("y")
        z = model.add_binary_var("z")
        model.add_constr(x + y == 1.0, name="link")
        model.add_constr(x + 2.0 * z <= 1.5, name="cap")
        model.maximize(y + z)

        solution = model.solve_mip()

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.root_presolve_aggregations, 1)
        self.assertGreaterEqual(solution.root_presolve_removed_coeffs, 1)
        self.assertTrue(math.isclose(solution.value(y), 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution.value(z), 0.0, rel_tol=0.0, abs_tol=1e-8))

    def test_solve_mip_diving_strategies(self):
        for diving_strategy in (
            simplinho.DivingStrategy.Fractional,
            simplinho.DivingStrategy.VectorLength,
            simplinho.DivingStrategy.ObjectiveValue,
            simplinho.DivingStrategy.Coefficient,
            simplinho.DivingStrategy.Guided,
            simplinho.DivingStrategy.Adaptive,
        ):
            with self.subTest(diving_strategy=diving_strategy):
                model, x, y = self._build_binary_branching_model()

                options = simplinho.BranchAndBoundOptions()
                options.node_selection = simplinho.NodeSelectionStrategy.BestBound
                options.diving_strategy = diving_strategy
                options.max_dive_depth = 8
                options.max_dive_lp_solves = 16

                solution = model.solve_mip(options)

                self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
                self.assertTrue(solution.has_solution)
                self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
                self.assertTrue(solution.heuristic_successes >= 1)
                self.assertTrue(solution.heuristic_lp_iterations >= 0)
                self.assertTrue(
                    math.isclose(
                        solution.value(x) + solution.value(y),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    )
                )

    def test_solve_mip_rins_and_local_search(self):
        model, _, _ = self._build_binary_branching_model()

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.diving_strategy = simplinho.DivingStrategy.Fractional
        options.use_rins = True
        options.rins_fix_ratio = 0.5
        options.use_local_search = True
        options.local_search_iterations = 4
        options.local_search_max_free_vars = 1
        options.heuristic_subproblem_max_nodes = 16

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.rins_successes, 1)
        self.assertGreaterEqual(solution.local_search_successes, 1)
        self.assertGreaterEqual(solution.heuristic_successes, 1)

    def test_solve_mip_local_branching(self):
        model = simplinho.Model()
        x1 = model.add_binary_var("x1")
        x2 = model.add_binary_var("x2")
        x3 = model.add_binary_var("x3")
        model.add_constr(5.0 * x1 + 4.0 * x2 + 3.0 * x3 <= 8.0, name="cap")
        model.maximize(8.0 * x1 + 7.0 * x2 + 6.0 * x3)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.Hybrid
        options.hybrid_depth_bias = 2
        options.diving_strategy = simplinho.DivingStrategy.Guided
        options.use_local_branching = True
        options.local_branching_neighborhood_ratio = 0.5
        options.local_branching_min_radius = 1
        options.local_branching_max_radius = 2
        options.heuristic_subproblem_max_nodes = 16

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 14.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.local_branching_successes, 0)

    def test_solve_mip_feasibility_pump_and_rens(self):
        model, x, y = self._build_binary_branching_model()

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.use_feasibility_pump = True
        options.feasibility_pump_iterations = 4
        options.feasibility_pump_fix_ratio = 0.5
        options.use_rens = True
        options.rens_fix_ratio = 0.5
        options.heuristic_subproblem_max_nodes = 16

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.feasibility_pump_successes, 1)
        self.assertGreaterEqual(solution.rens_successes, 1)
        self.assertGreaterEqual(solution.heuristic_successes, 1)
        self.assertTrue(
            math.isclose(
                solution.value(x) + solution.value(y), 1.0, rel_tol=0.0, abs_tol=1e-8
            )
        )

    def test_solve_mip_feasibility_jump(self):
        model, x, y = self._build_binary_branching_model()

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.diving_strategy = simplinho.DivingStrategy.Disabled
        options.use_feasibility_jump = True
        options.feasibility_jump_iterations = 8
        options.feasibility_jump_max_free_vars = 1
        options.use_feasibility_pump = False
        options.use_rens = False
        options.use_rins = False
        options.use_local_search = False
        options.use_local_branching = False
        options.heuristic_subproblem_max_nodes = 8

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.feasibility_jump_successes, 1)
        self.assertGreaterEqual(solution.heuristic_successes, 1)
        self.assertTrue(
            math.isclose(
                solution.value(x) + solution.value(y), 1.0, rel_tol=0.0, abs_tol=1e-8
            )
        )

    def test_heuristic_submips_can_use_diving_and_cuts(self):
        model = simplinho.Model()
        x = model.add_integer_var("x", lb=0.0, ub=10.0)
        y = model.add_binary_var("y")
        z = model.add_binary_var("z")
        model.add_constr(2.0 * x + 3.0 * y + 2.0 * z <= 7.0, name="cap")
        model.add_constr(y + z <= 1.0, name="pack")
        model.maximize(x + 2.0 * y + 2.0 * z)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.diving_strategy = simplinho.DivingStrategy.Fractional
        options.use_feasibility_pump = True
        options.feasibility_pump_iterations = 4
        options.use_rens = True
        options.rens_fix_ratio = 0.5
        options.use_cut_pool = True
        options.use_cover_cuts = True
        options.use_gomory_cuts = False
        options.max_cut_rounds_per_node = 2
        options.max_cuts_added_per_round = 4
        options.heuristic_subproblem_max_nodes = 16

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 4.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.heuristic_successes, 1)

    def test_solve_mip_cover_cuts_at_root(self):
        model, x, y = self._build_binary_branching_model()

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.use_cut_pool = True
        options.use_cover_cuts = True
        options.use_gomory_cuts = False
        options.max_cut_rounds_per_node = 2
        options.max_cuts_added_per_round = 4

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution.best_bound, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.cuts_generated, 1)
        self.assertGreaterEqual(solution.cuts_applied, 1)
        self.assertGreaterEqual(solution.cut_pool_size, 1)
        self.assertTrue(math.isclose(solution.value(x) + solution.value(y), 1.0, abs_tol=1e-8))

    def test_solve_mip_gomory_cuts(self):
        model = simplinho.Model()
        x = model.add_integer_var("x", lb=0.0, ub=10.0)
        y = model.add_binary_var("y")
        z = model.add_binary_var("z")
        model.add_constr(2.0 * x + 3.0 * y + 2.0 * z <= 7.0, name="cap")
        model.add_constr(y + z <= 1.0, name="pack")
        model.maximize(x + 2.0 * y + 2.0 * z)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.use_cut_pool = True
        options.use_gomory_cuts = True
        options.use_cover_cuts = False
        options.max_cut_rounds_per_node = 2
        options.max_cuts_added_per_round = 4

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 4.0, rel_tol=0.0, abs_tol=1e-6))
        self.assertGreaterEqual(solution.cuts_generated, 1)
        self.assertGreaterEqual(solution.cuts_applied, 1)
        self.assertGreaterEqual(solution.cut_pool_size, 1)
        self.assertTrue(math.isclose(solution.value(x), 2.0, rel_tol=0.0, abs_tol=1e-8))

    def test_solve_mip_implied_bound_cuts(self):
        model = simplinho.Model()
        x = model.add_integer_var("x", lb=0.0, ub=5.0)
        y = model.add_binary_var("y")
        model.add_constr(2.0 * x + 4.0 * y <= 12.0, name="cap")
        model.maximize(x + 1.5 * y)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.use_cut_pool = True
        options.use_gomory_cuts = False
        options.use_cover_cuts = False
        options.use_implied_bound_cuts = True
        options.max_cut_rounds_per_node = 2
        options.max_cuts_added_per_round = 4

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 5.5, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.cuts_generated, 1)
        self.assertGreaterEqual(solution.cuts_applied, 1)
        self.assertTrue(math.isclose(solution.value(x), 4.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution.value(y), 1.0, rel_tol=0.0, abs_tol=1e-8))

    def test_solve_mip_clique_cuts(self):
        model = simplinho.Model()
        x = model.add_binary_var("x")
        y = model.add_binary_var("y")
        z = model.add_binary_var("z")
        model.add_constr(2.0 * x + 2.0 * y + 2.0 * z <= 3.0, name="pack")
        model.maximize(x + y + z)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.use_cut_pool = True
        options.use_gomory_cuts = False
        options.use_cover_cuts = False
        options.use_implied_bound_cuts = False
        options.use_clique_cuts = True
        options.max_cut_rounds_per_node = 2
        options.max_cuts_added_per_round = 4

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution.best_bound, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.cuts_generated, 1)
        self.assertGreaterEqual(solution.cuts_applied, 1)
        self.assertTrue(
            math.isclose(
                solution.value(x) + solution.value(y) + solution.value(z),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        )

    def test_solve_mip_probing_implied_bound_cuts(self):
        model = simplinho.Model()
        x = model.add_integer_var("x", lb=0.0, ub=10.0)
        y = model.add_binary_var("y")
        z = model.add_binary_var("z")
        model.add_constr(x + 10.0 * z <= 15.0, name="cap")
        model.add_constr(y <= z, name="link")
        model.maximize(x + 3.0 * y)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.use_cut_pool = True
        options.use_gomory_cuts = False
        options.use_cover_cuts = False
        options.use_implied_bound_cuts = False
        options.use_clique_cuts = False
        options.use_probing_implications = True
        options.probing_max_candidates = 4
        options.max_cut_rounds_per_node = 2
        options.max_cuts_added_per_round = 4

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 10.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution.best_bound, 10.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.cuts_generated, 1)
        self.assertGreaterEqual(solution.cuts_applied, 1)
        self.assertTrue(math.isclose(solution.value(x), 10.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution.value(y), 0.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution.value(z), 0.0, rel_tol=0.0, abs_tol=1e-8))

    def test_cover_cuts_do_not_prune_global_optimum(self):
        model = simplinho.Model()
        profits = [23.0, 40.0, 57.0, 74.0, 91.0, 11.0, 28.0, 45.0, 62.0, 79.0]
        w1 = [8.0, 19.0, 1.0, 12.0, 23.0, 5.0, 16.0, 27.0, 9.0, 20.0]
        w2 = [6.0, 25.0, 13.0, 1.0, 20.0, 8.0, 27.0, 15.0, 3.0, 22.0]
        w3 = [4.0, 27.0, 13.0, 36.0, 22.0, 8.0, 31.0, 17.0, 3.0, 26.0]
        group = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        vars_ = [model.add_binary_var(f"x_{i}", obj=profits[i]) for i in range(10)]
        model.add_constr(sum(w1[i] * vars_[i] for i in range(10)) <= 49.0, name="cap_1")
        model.add_constr(sum(w2[i] * vars_[i] for i in range(10)) <= 46.2, name="cap_2")
        model.add_constr(sum(w3[i] * vars_[i] for i in range(10)) <= 57.97, name="cap_3")
        model.add_constr(sum(group[i] * vars_[i] for i in range(10)) <= 8.0, name="group_cap")
        model.maximize(sum(profits[i] * vars_[i] for i in range(10)))

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.branching_strategy = simplinho.BranchingStrategy.PseudoCost
        options.use_cut_pool = True
        options.use_gomory_cuts = True
        options.use_cover_cuts = True
        options.max_cut_rounds_per_node = 2
        options.max_cuts_added_per_round = 12

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 233.0, rel_tol=0.0, abs_tol=1e-8))

    def test_cover_cuts_match_multiknapsack_optimum(self):
        model, vars_ = self._build_large_binary_multiknapsack_model(25)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.branching_strategy = simplinho.BranchingStrategy.MostFractional
        options.diving_strategy = simplinho.DivingStrategy.Disabled
        options.parallel_workers = 1
        options.use_cut_pool = True
        options.use_gomory_cuts = False
        options.use_cover_cuts = True
        options.use_feasibility_pump = False
        options.use_rens = False
        options.use_rins = False
        options.use_local_search = False
        options.use_local_branching = False
        options.max_cut_rounds_per_node = 2
        options.max_cuts_added_per_round = 12

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 797.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution.best_bound, 797.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.cuts_generated, 1)
        self.assertGreaterEqual(solution.cuts_applied, 1)
        chosen = [i for i, var in enumerate(vars_) if math.isclose(solution.value(var), 1.0, rel_tol=0.0, abs_tol=1e-8)]
        self.assertEqual(chosen, [0, 2, 4, 8, 10, 13, 15, 16, 18, 21, 22])

    def test_cover_cuts_large_multiknapsack_35_does_not_raise(self):
        model, _ = self._build_large_binary_multiknapsack_model(35)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.branching_strategy = simplinho.BranchingStrategy.MostFractional
        options.diving_strategy = simplinho.DivingStrategy.Disabled
        options.parallel_workers = 1
        options.use_cut_pool = True
        options.use_gomory_cuts = False
        options.use_cover_cuts = True
        options.use_feasibility_pump = False
        options.use_rens = False
        options.use_rins = False
        options.use_local_search = False
        options.use_local_branching = False
        options.max_cut_rounds_per_node = 2
        options.max_cuts_added_per_round = 12

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1117.0, rel_tol=0.0, abs_tol=1e-8))


if __name__ == "__main__":
    unittest.main()
