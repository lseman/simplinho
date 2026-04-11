import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

candidates = [
    ROOT / "build-perf",
    ROOT / "build",
    ROOT / "build-verify",
]
module_names = ["simplinho", "simplex"]
for candidate in candidates:
    if not candidate.exists():
        continue
    sys.path.insert(0, str(candidate))
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            print("loaded", module.__file__)
            import simplinho

            options = simplinho.RevisedSimplexOptions()
            options.mode = simplinho.SimplexMode.Auto
            model = simplinho.Model(options)
            x = model.addVar("x", lb=0.0)
            y = model.addVar("y", lb=0.0)
            model.addConstr(x + y <= 4.0, name="cap")
            model.maximize(x + 2.0 * y)
            sol1 = model.solve()
            print("sol1 status", simplinho.status_to_string(sol1.status))
            print("sol1 basis num_columns", sol1.basis.num_columns)
            print("sol1 basis basic_columns", sol1.basis.basic_columns)
            model.options.mode = simplinho.SimplexMode.Dual
            x.ub = 1.5
            sol2 = model.reoptimize()
            print("sol2 status", simplinho.status_to_string(sol2.status))
            x.ub = 1.0
            try:
                sol3 = model.reoptimize(sol1.basis)
                print("sol3 status", simplinho.status_to_string(sol3.status))
            except Exception as e:
                print(type(e).__name__, e)
            raise SystemExit(0)
        except ImportError as e:
            print("import failed", module_name, e)
        finally:
            sys.path.pop(0)
print("no module loaded")
