import importlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(".venv/lib/python3.13/site-packages").resolve()))
build = pathlib.Path("build").resolve()
sys.path.insert(0, str(build))
mod = importlib.import_module("simplinho")
print("module", mod.__file__)
options = mod.RevisedSimplexOptions()
options.mode = mod.SimplexMode.Auto
model = mod.Model(options)
x = model.addVar("x", lb=0.0)
y = model.addVar("y", lb=0.0)
model.addConstr(x + y <= 4.0, name="cap")
model.maximize(x + 2.0 * y)
sol1 = model.solve()
print("sol1 status", mod.status_to_string(sol1.status))
print("sol1 basis num_columns", sol1.basis.num_columns)
print("sol1 basis basic_columns", sol1.basis.basic_columns)
print("sol1 basis column_status", sol1.basis.column_status)
raw1 = sol1.raw
print("raw1 x size", len(raw1.x))
print("raw1 basis size", len(raw1.basis))
print("raw1 basis", raw1.basis)
print("raw1 basis_state size", len(raw1.basis_state.column_status))
print(
    "raw1 log contains solve start",
    "[solve] start" in raw1.log or "[solve] sparse start" in raw1.log,
)
print("raw1 log_lines", len(raw1.log_lines))
model.options.mode = mod.SimplexMode.Dual
x.ub = 1.5
sol2 = model.reoptimize()
print("sol2 status", mod.status_to_string(sol2.status))
print("sol2 basis_start", sol2.stats.basis_start)
x.ub = 1.0
try:
    sol3 = model.reoptimize(sol1.basis)
    print("sol3 status", mod.status_to_string(sol3.status))
except Exception as e:
    print(type(e).__name__, e)
