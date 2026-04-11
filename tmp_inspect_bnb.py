import importlib.util
from pathlib import Path

build = Path("build")
mod_path = build / "simplinho.cpython-313-x86_64-linux-gnu.so"
spec = importlib.util.spec_from_file_location("simplinho", mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

n = 100
coeffs_1 = []
coeffs_2 = []
coeffs_3 = []
coeffs_4 = []
case = {"sense": "max", "variables": [], "constraints": []}
for i in range(n):
    profit = float(((17 * i + 13) % 97) + 10)
    w1 = float(((11 * i + 7) % 29) + 1)
    w2 = float(((19 * i + 5) % 31) + 1)
    w3 = float(((23 * i + 3) % 37) + 1)
    group = 1.0 if i % 10 in (0, 1, 2, 3) else 0.0
    case["variables"].append({"name": f"x_{i}", "type": "binary", "obj": profit})
    coeffs_1.append(w1)
    coeffs_2.append(w2)
    coeffs_3.append(w3)
    coeffs_4.append(group)
case["constraints"] = [
    {
        "name": "cap_1",
        "coeffs": {f"x_{i}": coeffs_1[i] for i in range(n)},
        "sense": "<=",
        "rhs": 0.35 * sum(coeffs_1),
    },
    {
        "name": "cap_2",
        "coeffs": {f"x_{i}": coeffs_2[i] for i in range(n)},
        "sense": "<=",
        "rhs": 0.33 * sum(coeffs_2),
    },
    {
        "name": "cap_3",
        "coeffs": {f"x_{i}": coeffs_3[i] for i in range(n)},
        "sense": "<=",
        "rhs": 0.31 * sum(coeffs_3),
    },
    {
        "name": "group_cap",
        "coeffs": {f"x_{i}": coeffs_4[i] for i in range(n)},
        "sense": "<=",
        "rhs": max(8.0, 0.18 * n),
    },
]

model = mod.Model()
vars_by_name = {}
for spec in case["variables"]:
    vars_by_name[spec["name"]] = model.add_binary_var(spec["name"], obj=spec["obj"])
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
model.maximize(
    sum(spec["obj"] * vars_by_name[spec["name"]] for spec in case["variables"])
)

options = mod.BranchAndBoundOptions()
options.max_nodes = 10000
options.node_selection = mod.NodeSelectionStrategy.BestBound
options.parallel_workers = 1
options.use_async_heuristics = False
options.use_cut_pool = False
options.use_gomory_cuts = False
options.use_cover_cuts = False
options.use_feasibility_pump = False
options.use_rens = False
options.use_rins = False
options.use_local_search = False
options.use_local_branching = False

res = model.solve_mip(options)
print("mip", res.status, res.obj, res.best_bound, res.node_count)
print("tree_nodes", len(res.tree_nodes))
for i, node in enumerate(res.tree_nodes):
    print(
        i,
        node.parent_id,
        node.depth,
        node.status,
        node.branch_var,
        node.branch_value,
        node.bound,
    )
