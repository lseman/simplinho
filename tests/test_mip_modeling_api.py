import importlib.machinery
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def import_simplinho():
    for build_dir in ("build-local", "build", "build-verify"):
        candidate = ROOT / build_dir
        if not candidate.exists():
            continue
        extensions = list(candidate.glob("simplinho*.so"))
        if not extensions:
            continue
        extension = extensions[0]
        try:
            loader = importlib.machinery.ExtensionFileLoader(
                "simplinho", str(extension)
            )
            spec = importlib.util.spec_from_loader("simplinho", loader)
            module = importlib.util.module_from_spec(spec)
            sys.modules["simplinho"] = module
            loader.exec_module(module)
            return module
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

    def test_solve_mip_branches_on_sos1_sets(self):
        model = simplinho.Model()
        x = model.add_binary_var("x")
        y = model.add_binary_var("y")
        model.add_constr(2.0 * x + 2.0 * y <= 3.0)
        model.add_sos1([x, y], [0.0, 1.0])
        model.maximize(x + y)

        solution = model.solve_mip()

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertLessEqual(
            sum(1 for var in (x, y) if abs(solution.value(var)) > 1e-8), 1
        )

    def test_solve_mip_branches_on_sos2_sets(self):
        model = simplinho.Model()
        x = model.add_binary_var("x")
        y = model.add_binary_var("y")
        middle = model.add_var("middle", lb=0.0, ub=0.0)
        model.add_constr(2.0 * x + 2.0 * y <= 3.0)
        model.add_sos2([x, middle, y], [0.0, 1.0, 2.0])
        model.maximize(x + y)

        solution = model.solve_mip()

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        active = [
            pos
            for pos, var in enumerate((x, middle, y))
            if abs(solution.value(var)) > 1e-8
        ]
        self.assertLessEqual(len(active), 2)
        if len(active) == 2:
            self.assertEqual(active[1], active[0] + 1)

    def test_model_reoptimize_explicit_basis_warm_start(self):
        options = simplinho.RevisedSimplexOptions()
        options.mode = simplinho.SimplexMode.Auto
        model = simplinho.Model(options)
        x = model.add_var("x", lb=0.0, ub=5.0)
        y = model.add_var("y", lb=0.0, ub=5.0)
        model.add_constr(x + y <= 6.0, name="cap")
        model.maximize(4.0 * x + 3.0 * y)

        sol1 = model.solve()
        self.assertEqual(simplinho.status_to_string(sol1.status), "optimal")
        basis = sol1.basis
        self.assertEqual(basis.num_columns, 3)
        self.assertEqual(len(basis.basic_columns), 1)

        model.options.mode = simplinho.SimplexMode.Dual
        x.ub = 1.5
        sol2 = model.reoptimize()
        self.assertEqual(simplinho.status_to_string(sol2.status), "optimal")

        x.ub = 1.0
        sol3 = model.reoptimize(basis)
        self.assertEqual(simplinho.status_to_string(sol3.status), "optimal")
        self.assertIn(sol3.stats.basis_start, {"warm_start", "repaired_warm_start"})

    def test_branch_and_bound_options_default_warm_presolve_override(self):
        options = simplinho.BranchAndBoundOptions()
        self.assertFalse(options.use_node_presolve_on_warm_basis)

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
        self.assertTrue(
            math.isclose(solution.best_bound, 1.0, rel_tol=0.0, abs_tol=1e-8)
        )
        self.assertTrue(
            math.isclose(
                solution.root_relaxation_objective, 1.5, rel_tol=0.0, abs_tol=1e-8
            )
        )
        self.assertGreaterEqual(solution.node_count, 2)
        self.assertGreaterEqual(solution.warm_start_relaxation_attempt_count, 1)
        self.assertGreaterEqual(
            solution.warm_start_relaxation_attempt_count,
            solution.warm_start_relaxation_accept_count
            + solution.warm_start_cold_retry_count,
        )
        self.assertGreaterEqual(solution.warm_start_cold_retry_count, 0)
        self.assertTrue(
            math.isclose(
                solution.value(x), round(solution.value(x)), rel_tol=0.0, abs_tol=1e-8
            )
        )
        self.assertTrue(
            math.isclose(
                solution.value(y), round(solution.value(y)), rel_tol=0.0, abs_tol=1e-8
            )
        )
        self.assertGreaterEqual(len(solution.tree_nodes), 1)
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

    def test_solve_mip_degenerate_reduced_cost_fixing(self):
        model = simplinho.Model()
        x = model.add_integer_var("x", lb=0.0, ub=3.0, obj=5.0)
        y = model.add_integer_var("y", lb=0.0, ub=4.0, obj=2.0)
        model.add_constr(x + 2.0 * y <= 5.0, name="cap")
        model.maximize(5.0 * x + 2.0 * y)

        solution = model.solve_mip()

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 17.0, rel_tol=0.0, abs_tol=1e-8))

    def test_solve_mip_hybrid_queue_domain_rebasing_preserves_multiknapsack_optimum(
        self,
    ):
        model, _ = self._build_large_binary_multiknapsack_model(25)
        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.Hybrid
        options.hybrid_depth_bias = 2
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
        self.assertTrue(
            math.isclose(solution.best_bound, 797.0, rel_tol=0.0, abs_tol=1e-8)
        )
        self.assertGreaterEqual(solution.cuts_generated, 1)
        self.assertGreaterEqual(solution.cuts_applied, 1)

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

    def test_solve_mip_verbose_banner_reports_configuration_and_summary(self):
        script = """
import importlib.util
import sys
from pathlib import Path

root = Path(r'__ROOT__')
build_dir = root / 'build'
module_path = next(build_dir.glob('simplinho*.so'))
spec = importlib.util.spec_from_file_location('simplinho', module_path)
module = importlib.util.module_from_spec(spec)
sys.modules['simplinho'] = module
spec.loader.exec_module(module)

model = module.Model()
x = model.add_binary_var('x')
y = model.add_binary_var('y')
model.add_constr(2.0 * x + 2.0 * y <= 3.0, name='cap')
model.maximize(x + y)

options = module.BranchAndBoundOptions()
options.verbose = True
options.log_frequency = 1
options.parallel_workers = 2
options.use_async_heuristics = True
options.use_rounding = True
options.use_diving = True
options.use_feasibility_jump = True
options.use_feasibility_pump = True
options.use_rens = True
options.use_rins = True
options.use_local_search = True
options.use_local_branching = True
options.use_cut_pool = True
options.use_gomory_cuts = True
options.use_mir_cuts = True
options.use_cover_cuts = True
options.use_implied_bound_cuts = True
options.use_clique_cuts = True
options.use_odd_cycle_cuts = True
options.use_probing_implications = True
options.use_conflict_cuts = True

solution = model.solve_mip(options)
print('status', module.mip_status_to_string(solution.status))
""".replace("__ROOT__", str(ROOT))

        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        output = proc.stdout
        self.assertIn("Simplinho ", output)
        self.assertIn("git ", output)
        self.assertIn("branch ", output)
        self.assertIn("MIP Search", output)
        self.assertIn("branch ", output)
        self.assertIn("workers 2", output)
        self.assertIn("MIP Heur", output)
        self.assertIn("MIP Cuts", output)
        self.assertIn("MIP Timing Summary", output)
        self.assertIn("MIP Heuristic Summary", output)
        self.assertIn("MIP Cut Summary", output)
        self.assertIn("status optimal", output)

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

    def test_solve_mip_parallel_workers_stress(self):
        model = simplinho.Model()
        items = [43, 41, 37, 31, 29, 23, 19, 17, 13, 11]
        vars_ = [
            model.add_binary_var(f"x_{i}", obj=float(weight))
            for i, weight in enumerate(items)
        ]
        model.add_constr(sum(vars_) <= 4, name="cardinality")
        model.add_constr(
            sum((i % 2) * vars_[i] for i in range(len(vars_))) <= 2, name="parity_limit"
        )
        model.maximize(sum(float(weight) * vars_[i] for i, weight in enumerate(items)))

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestEstimate
        options.branching_strategy = simplinho.BranchingStrategy.PseudoCost
        options.parallel_workers = 4
        options.max_nodes = 10000

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertEqual(solution.obj, 152.0)
        self.assertGreater(solution.node_count, 1)
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
                self.assertTrue(
                    math.isclose(solution.obj, 2.0, rel_tol=0.0, abs_tol=1e-8)
                )
                self.assertTrue(
                    math.isclose(solution.best_bound, 2.0, rel_tol=0.0, abs_tol=1e-8)
                )
                self.assertTrue(
                    math.isclose(
                        solution.root_relaxation_objective,
                        2.5,
                        rel_tol=0.0,
                        abs_tol=1e-8,
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
        options.use_mir_cuts = True
        options.use_odd_cycle_cuts = True
        options.use_conflict_cuts = False
        options.max_conflict_cuts_per_round = 2
        options.max_cuts_per_type = 3
        options.cut_max_parallelism = 0.9
        options.use_lp_reoptimization_profile = False

        self.assertTrue(options.use_mir_cuts)
        self.assertTrue(options.use_odd_cycle_cuts)
        self.assertFalse(options.use_conflict_cuts)
        self.assertEqual(options.max_conflict_cuts_per_round, 2)
        self.assertEqual(options.max_cuts_per_type, 3)
        self.assertFalse(options.use_lp_reoptimization_profile)
        self.assertTrue(
            math.isclose(options.cut_max_parallelism, 0.9, rel_tol=0.0, abs_tol=1e-12)
        )

    def test_solve_mip_uses_reoptimization_lp_profile_by_default(self):
        model, _, _ = self._build_binary_branching_model()
        model.options.mode = simplinho.SimplexMode.Primal
        model.options.partial_pricing = False
        model.options.dual_pricing = "row"

        solution = model.solve_mip()

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertEqual(solution.lp_profile, "bnb_reoptimization")
        self.assertEqual(solution.lp_mode, "dual")
        self.assertTrue(solution.lp_partial_pricing)
        self.assertEqual(solution.lp_dual_pricing, "switch")

    def test_solve_mip_can_disable_reoptimization_lp_profile(self):
        model, _, _ = self._build_binary_branching_model()
        model.options.mode = simplinho.SimplexMode.Primal
        model.options.partial_pricing = False
        model.options.dual_pricing = "row"

        options = simplinho.BranchAndBoundOptions()
        options.use_lp_reoptimization_profile = False
        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertEqual(solution.lp_profile, "model_options")
        self.assertEqual(solution.lp_mode, "primal")
        self.assertFalse(solution.lp_partial_pricing)
        self.assertEqual(solution.lp_dual_pricing, "row")

    def test_solve_mip_uses_basis_state_for_child_node_reoptimization(self):
        model = simplinho.Model()
        x = model.add_binary_var("x")
        y = model.add_binary_var("y")
        model.add_constr(x + y <= 1.5, name="fractional_root")
        model.maximize(x + y)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        # Disable rounding heuristic so the BNB must explore child nodes via LP solves,
        # verifying that warm start basis state is passed to child LPs.
        options.use_rounding = False

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertGreater(solution.node_count, 1)
        self.assertEqual(solution.lp_profile, "bnb_reoptimization")
        self.assertEqual(solution.lp_mode, "dual")
        self.assertTrue(solution.warm_start_basis_state_used)
        self.assertGreaterEqual(solution.lp_refactorizations, 0)
        self.assertGreaterEqual(solution.lp_eta_stack_depth_entry_sum, 0)
        self.assertGreaterEqual(solution.lp_warm_factorization_reuse_count, 0)
        self.assertGreaterEqual(solution.lp_dual_pool_builds, 0)
        self.assertGreaterEqual(solution.relaxation_lp_lu_build_ns, 0)
        self.assertGreaterEqual(solution.relaxation_lp_pricing_build_ns, 0)
        self.assertGreaterEqual(solution.relaxation_lp_pivot_ns, 0)

    def test_solve_mip_reduced_cost_bound_tightening_path(self):
        model = simplinho.Model()
        x = model.add_binary_var("x")
        y = model.add_binary_var("y")
        model.add_constr(x + 2 * y <= 2, name="cap")
        model.maximize(x + y)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.use_rounding = False
        options.use_cut_pool = False
        options.use_rins = False
        options.use_rens = False
        options.use_local_search = False
        options.use_feasibility_pump = False
        options.use_feasibility_jump = False
        options.use_async_heuristics = False

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertGreater(solution.node_count, 1)
        self.assertEqual(solution.lp_profile, "bnb_reoptimization")
        self.assertEqual(solution.lp_mode, "dual")

    def test_solve_mip_rebuilds_thread_local_lp_context_between_runs(self):
        model = simplinho.Model()
        vars_ = [model.add_binary_var(f"x{i}") for i in range(5)]
        for i in range(5):
            model.add_constr(vars_[i] + vars_[(i + 1) % 5] <= 1.0, name=f"e{i}")
        model.maximize(sum(vars_))

        def make_options(use_profile):
            options = simplinho.BranchAndBoundOptions()
            options.node_selection = simplinho.NodeSelectionStrategy.BestBound
            options.use_cut_pool = True
            options.use_gomory_cuts = False
            options.use_mir_cuts = False
            options.use_cover_cuts = False
            options.use_implied_bound_cuts = False
            options.use_clique_cuts = False
            options.use_odd_cycle_cuts = True
            options.use_probing_implications = False
            options.use_conflict_cuts = False
            options.max_cut_rounds_per_node = 2
            options.max_cuts_added_per_round = 4
            options.use_lp_reoptimization_profile = use_profile
            return options

        first = model.solve_mip(make_options(True))
        second = model.solve_mip(make_options(False))

        self.assertEqual(first.status, simplinho.MIPStatus.Optimal)
        self.assertEqual(second.status, simplinho.MIPStatus.Optimal)
        self.assertEqual(first.lp_profile, "bnb_reoptimization")
        self.assertEqual(second.lp_profile, "model_options")
        self.assertTrue(math.isclose(first.obj, 2.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(second.obj, 2.0, rel_tol=0.0, abs_tol=1e-8))

    def test_reoptimization_profile_does_not_falsely_fathom_large_multiknapsack(self):
        model, _ = self._build_large_binary_multiknapsack_model(100)

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
        options.max_cuts_added_per_round = 12
        options.heuristic_subproblem_max_nodes = 96
        options.max_nodes = 5
        options.parallel_workers = 1

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.NodeLimit)
        self.assertTrue(solution.has_solution)
        self.assertGreaterEqual(solution.warm_start_relaxation_attempt_count, 1)
        self.assertGreater(solution.best_bound, solution.obj + 1e-6)

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
        model.maximize(
            5.0 * vars_[0] + 2.0 * vars_[1] + 8.0 * vars_[2] + 4.0 * vars_[3]
        )

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
        self.assertGreaterEqual(strong_branching.tree_nodes[0].branch_var, 0)
        self.assertLessEqual(strong_branching.node_count, most_fractional.node_count)

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
        self.assertTrue(
            math.isclose(solution.value(x1), 1.0, rel_tol=0.0, abs_tol=1e-8)
        )
        self.assertTrue(
            math.isclose(solution.value(x2), 1.0, rel_tol=0.0, abs_tol=1e-8)
        )
        self.assertTrue(
            math.isclose(solution.value(x3), 1.0, rel_tol=0.0, abs_tol=1e-8)
        )
        self.assertTrue(
            math.isclose(solution.value(x4), 1.0, rel_tol=0.0, abs_tol=1e-8)
        )

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

    def test_root_mip_presolve_merges_parallel_knapsack_rows(self):
        model = simplinho.Model()
        x1 = model.add_binary_var("x1")
        x2 = model.add_binary_var("x2")
        x3 = model.add_binary_var("x3")
        model.add_constr(x1 + x2 + x3 <= 2.0, name="cap_weak")
        model.add_constr(x1 + x2 + x3 <= 1.0, name="cap_strong")
        model.maximize(x1 + x2 + x3)

        solution = model.solve_mip()

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.root_presolve_removed_rows, 1)
        self.assertTrue(
            math.isclose(
                solution.value(x1) + solution.value(x2) + solution.value(x3),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        )

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
                self.assertTrue(
                    math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8)
                )
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
        options.use_async_heuristics = False
        options.local_search_iterations = 4
        options.local_search_max_free_vars = 1
        options.heuristic_subproblem_max_nodes = 16

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.rins_successes, 0)
        self.assertGreaterEqual(solution.local_search_successes, 0)
        self.assertGreaterEqual(solution.heuristic_successes, 0)

    def test_solve_mip_async_heuristics_do_not_block(self):
        model, vars_ = self._build_large_binary_multiknapsack_model(n=18)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.branching_strategy = simplinho.BranchingStrategy.PseudoCost
        options.parallel_workers = 2
        options.use_rins = True
        options.use_local_search = True
        options.use_async_heuristics = True
        options.local_search_iterations = 4
        options.local_search_max_free_vars = 1
        options.heuristic_subproblem_max_nodes = 16
        options.max_nodes = 200

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isfinite(solution.obj))
        self.assertGreaterEqual(solution.heuristic_successes, 0)

    def test_solve_mip_node_timing_log_writes_jsonl(self):
        model, _, _ = self._build_binary_branching_model()

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.DepthFirst
        options.branching_strategy = simplinho.BranchingStrategy.PseudoCost
        options.use_cut_pool = False
        options.use_async_heuristics = False
        options.use_feasibility_jump = False
        options.use_feasibility_pump = False
        options.use_rens = False
        options.use_rins = False
        options.use_local_search = False
        options.use_local_branching = False

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "node-timing.jsonl"
            options.node_timing_log_path = str(log_path)

            solution = model.solve_mip(options)

            self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
            self.assertTrue(log_path.exists())

            lines = [line for line in log_path.read_text().splitlines() if line.strip()]
            self.assertGreaterEqual(len(lines), 1)

            records = [json.loads(line) for line in lines]
            root_record = next(record for record in records if record["node_id"] == 0)

            self.assertIn("total_wall_ns", root_record)
            self.assertIn("node_relaxation_wall_ns", root_record)
            self.assertIn("branching_wall_ns", root_record)
            self.assertIn("child_processing_wall_ns", root_record)
            self.assertIn("final_status", root_record)
            self.assertIn("exit_stage", root_record)
            self.assertGreaterEqual(root_record["total_wall_ns"], 0)
            self.assertGreaterEqual(root_record["node_relaxation_solve_count"], 1)
            self.assertEqual(root_record["depth"], 0)

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

    def test_rounding_repair_heuristic_finds_binary_incumbent(self):
        model = simplinho.Model()
        x1 = model.add_binary_var("x1")
        x2 = model.add_binary_var("x2")
        x3 = model.add_binary_var("x3")
        model.add_constr(x1 + x2 + x3 <= 1.5, name="cap")
        model.maximize(x1 + x2 + x3)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.diving_strategy = simplinho.DivingStrategy.Disabled
        options.use_feasibility_jump = False
        options.use_feasibility_pump = False
        options.use_rens = False
        options.use_rins = False
        options.use_local_search = False
        options.use_local_branching = False

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.heuristic_successes, 1)
        self.assertTrue(
            math.isclose(
                solution.value(x1) + solution.value(x2) + solution.value(x3),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-8,
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
        self.assertTrue(
            math.isclose(solution.best_bound, 1.0, rel_tol=0.0, abs_tol=1e-8)
        )
        self.assertGreaterEqual(solution.cuts_generated, 1)
        self.assertGreaterEqual(solution.cuts_applied, 1)
        self.assertGreaterEqual(solution.cut_pool_size, 1)
        self.assertTrue(
            math.isclose(solution.value(x) + solution.value(y), 1.0, abs_tol=1e-8)
        )

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
        options.use_rounding = (
            False  # disable rounding so cuts run on fractional branches
        )

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 4.0, rel_tol=0.0, abs_tol=1e-6))
        self.assertGreaterEqual(solution.cuts_generated, 1)
        self.assertGreaterEqual(solution.cuts_applied, 1)
        self.assertGreaterEqual(solution.cut_pool_size, 1)
        self.assertTrue(math.isclose(solution.value(x), 2.0, rel_tol=0.0, abs_tol=1e-8))

    def test_solve_mip_mir_cuts(self):
        model = simplinho.Model()
        x = model.add_integer_var("x", lb=0.0, ub=3.0)
        y = model.add_var("y", lb=0.0, ub=1.0)
        model.add_constr(1.5 * x + y <= 2.3, name="mix_cap")
        model.maximize(x + y)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.use_cut_pool = True
        options.use_gomory_cuts = False
        options.use_mir_cuts = True
        options.use_cover_cuts = False
        options.use_implied_bound_cuts = False
        options.use_clique_cuts = False
        options.use_probing_implications = False
        options.use_conflict_cuts = False
        options.max_cut_rounds_per_node = 2
        options.max_cuts_added_per_round = 4
        options.use_rounding = (
            False  # disable rounding so cuts run on fractional branches
        )

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.8, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.cuts_generated, 1)
        self.assertGreaterEqual(solution.cuts_applied, 1)
        self.assertTrue(
            math.isclose(solution.best_bound, 1.8, rel_tol=0.0, abs_tol=1e-8)
        )
        self.assertTrue(math.isclose(solution.value(x), 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution.value(y), 0.8, rel_tol=0.0, abs_tol=1e-8))

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
        options.use_rounding = (
            False  # disable rounding so cuts run on fractional branches
        )

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 5.5, rel_tol=0.0, abs_tol=1e-8))
        self.assertGreaterEqual(solution.cuts_generated, 1)
        self.assertGreaterEqual(solution.cuts_applied, 1)
        self.assertTrue(math.isclose(solution.value(x), 4.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution.value(y), 1.0, rel_tol=0.0, abs_tol=1e-8))

    def test_solve_mip_implied_bound_cuts_negative_coefficient(self):
        model = simplinho.Model()
        x = model.add_integer_var("x", lb=0.0, ub=10.0)
        y = model.add_binary_var("y")
        z = model.add_binary_var("z")
        model.add_constr(-1.0 * x + 2.0 * y + 3.0 * z <= 3.0, name="negcoef")
        model.add_constr(y + z <= 1.0, name="pack")
        model.maximize(y + z)

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.use_cut_pool = True
        options.use_gomory_cuts = False
        options.use_cover_cuts = False
        options.use_implied_bound_cuts = True
        options.use_clique_cuts = False
        options.use_probing_implications = False
        options.use_conflict_cuts = False
        options.max_cut_rounds_per_node = 2
        options.max_cuts_added_per_round = 4

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(
            math.isclose(
                solution.value(y) + solution.value(z), 1.0, rel_tol=0.0, abs_tol=1e-8
            )
        )

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
        self.assertTrue(
            math.isclose(solution.best_bound, 1.0, rel_tol=0.0, abs_tol=1e-8)
        )
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

    def test_solve_mip_disable_graph_clique_separator(self):
        model = simplinho.Model()
        x = model.add_binary_var("x")
        y = model.add_binary_var("y")
        z = model.add_binary_var("z")
        model.add_constr(2.0 * x + 2.0 * y + 2.0 * z <= 3.0)
        model.maximize(x + y + z)

        options = simplinho.BranchAndBoundOptions()
        options.use_cut_pool = True
        options.use_gomory_cuts = False
        options.use_cover_cuts = False
        options.use_implied_bound_cuts = False
        options.use_clique_cuts = True
        options.use_graph_clique_cuts = False
        options.max_cut_rounds_per_node = 2
        options.max_cuts_added_per_round = 4

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(
            math.isclose(
                solution.value(x) + solution.value(y) + solution.value(z),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        )

    def test_solve_mip_clique_merge_parallel_columns_general_knapsack(self):
        model = simplinho.Model()
        x = model.add_binary_var("x")
        y = model.add_binary_var("y")
        z = model.add_binary_var("z")
        model.add_constr(3.0 * x + 1.0 * y + 3.0 * z <= 5.0)
        model.add_constr(x + z <= 2.0)
        model.maximize(2.0 * x + 1.0 * z + 0.0 * y)

        options = simplinho.BranchAndBoundOptions()
        options.use_cut_pool = False
        options.use_gomory_cuts = False
        options.use_cover_cuts = False
        options.use_implied_bound_cuts = False
        options.use_conflict_cuts = False

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertEqual(solution.value(x), 1.0)
        self.assertEqual(solution.value(z), 0.0)
        self.assertGreaterEqual(solution.root_presolve_tightened_bounds, 1)

    def test_solve_mip_odd_cycle_cuts(self):
        model = simplinho.Model()
        vars_ = [model.add_binary_var(f"x{i}") for i in range(5)]
        for i in range(5):
            model.add_constr(vars_[i] + vars_[(i + 1) % 5] <= 1.0, name=f"e{i}")
        model.maximize(sum(vars_))

        options = simplinho.BranchAndBoundOptions()
        options.node_selection = simplinho.NodeSelectionStrategy.BestBound
        options.use_cut_pool = True
        options.use_gomory_cuts = False
        options.use_mir_cuts = False
        options.use_cover_cuts = False
        options.use_implied_bound_cuts = False
        options.use_clique_cuts = False
        options.use_odd_cycle_cuts = True
        options.use_probing_implications = False
        options.use_conflict_cuts = False
        options.max_cut_rounds_per_node = 2
        options.max_cuts_added_per_round = 4

        solution = model.solve_mip(options)

        self.assertEqual(solution.status, simplinho.MIPStatus.Optimal)
        self.assertTrue(solution.has_solution)
        self.assertTrue(math.isclose(solution.obj, 2.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(
            math.isclose(solution.best_bound, 2.0, rel_tol=0.0, abs_tol=1e-8)
        )
        self.assertGreaterEqual(solution.cuts_generated, 1)
        self.assertGreaterEqual(solution.cuts_applied, 1)
        self.assertTrue(
            math.isclose(
                sum(solution.value(var) for var in vars_),
                2.0,
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
        self.assertTrue(
            math.isclose(solution.best_bound, 10.0, rel_tol=0.0, abs_tol=1e-8)
        )
        self.assertGreaterEqual(solution.cuts_generated, 1)
        self.assertGreaterEqual(solution.cuts_applied, 1)
        self.assertTrue(
            math.isclose(solution.value(x), 10.0, rel_tol=0.0, abs_tol=1e-8)
        )
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
        model.add_constr(
            sum(w3[i] * vars_[i] for i in range(10)) <= 57.97, name="cap_3"
        )
        model.add_constr(
            sum(group[i] * vars_[i] for i in range(10)) <= 8.0, name="group_cap"
        )
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
        self.assertTrue(
            math.isclose(solution.best_bound, 797.0, rel_tol=0.0, abs_tol=1e-8)
        )
        self.assertGreaterEqual(solution.cuts_generated, 1)
        self.assertGreaterEqual(solution.cuts_applied, 1)
        chosen = [
            i
            for i, var in enumerate(vars_)
            if math.isclose(solution.value(var), 1.0, rel_tol=0.0, abs_tol=1e-8)
        ]
        self.assertEqual(chosen, [0, 2, 4, 8, 10, 13, 15, 16, 18, 21, 22])

    def test_node_cuts_still_run_when_strong_branching_depth_is_zero(self):
        def make_options(max_nodes):
            options = simplinho.BranchAndBoundOptions()
            options.max_nodes = max_nodes
            options.parallel_workers = 1
            options.node_selection = simplinho.NodeSelectionStrategy.BestBound
            options.branching_strategy = simplinho.BranchingStrategy.StrongBranching
            options.strong_branching_candidates = 0
            options.strong_branching_max_depth = 0
            options.diving_strategy = simplinho.DivingStrategy.Disabled
            options.use_feasibility_pump = False
            options.use_feasibility_jump = False
            options.use_rens = False
            options.use_rins = False
            options.use_local_search = False
            options.use_local_branching = False
            options.heuristic_frequency = 8
            options.use_cut_pool = True
            options.max_cut_rounds_per_node = 4
            options.max_cuts_added_per_round = 12
            options.use_gomory_cuts = False
            options.use_cover_cuts = True
            options.use_mir_cuts = True
            options.use_implied_bound_cuts = False
            options.use_clique_cuts = False
            options.use_odd_cycle_cuts = False
            options.use_probing_implications = False
            options.use_conflict_cuts = False
            return options

        root_only_model, _ = self._build_large_binary_multiknapsack_model(30)
        root_only = root_only_model.solve_mip(make_options(1))

        node_cut_model, _ = self._build_large_binary_multiknapsack_model(30)
        node_cut_run = node_cut_model.solve_mip(make_options(64))

        self.assertEqual(root_only.status, simplinho.MIPStatus.NodeLimit)
        self.assertEqual(node_cut_run.status, simplinho.MIPStatus.NodeLimit)
        self.assertGreaterEqual(root_only.node_count, 1)
        self.assertGreater(node_cut_run.node_count, root_only.node_count)
        self.assertGreater(root_only.cuts_generated, 0)
        self.assertGreater(node_cut_run.cuts_generated, root_only.cuts_generated)
        self.assertGreaterEqual(node_cut_run.cuts_applied, root_only.cuts_applied)

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
