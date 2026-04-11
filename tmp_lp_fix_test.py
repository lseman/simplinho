import pulp

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


def solve_lp(fix0=None):
    prob = pulp.LpProblem("test", pulp.LpMaximize)
    vars_p = {}
    for spec in case["variables"]:
        vars_p[spec["name"]] = pulp.LpVariable(
            spec["name"], lowBound=0, upBound=1, cat=pulp.LpContinuous
        )
    if fix0 is not None:
        prob += vars_p["x_0"] == fix0
    for row in case["constraints"]:
        expr = pulp.lpSum(coeff * vars_p[name] for name, coeff in row["coeffs"].items())
        if row["sense"] == "<=":
            prob += expr <= row["rhs"]
        elif row["sense"] == ">=":
            prob += expr >= row["rhs"]
        else:
            prob += expr == row["rhs"]
    prob += pulp.lpSum(spec["obj"] * vars_p[spec["name"]] for spec in case["variables"])
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.LpStatus[status], pulp.value(prob.objective)


for fix in [None, 0, 1]:
    status, obj = solve_lp(fix)
    print("fix0", fix, status, obj)
