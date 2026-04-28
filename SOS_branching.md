# SOS1/SOS2 Branching in Simplinho

This document summarizes how `simplinho` implements branching on SOS1 and SOS2 constraints in the branch-and-bound solver.

## Representation

SOS constraints are represented as a structure with:

- `type`: `SOSType::SOS1` or `SOSType::SOS2`
- `variables`: ordered list of variable indices in the SOS set
- `weights`: ordering weights provided by the user (used to define the order)

The relevant branching code lives in `src/bnb/branching.cpp`, and the child state creation helper is in `include/bnb/diving.h`.

## SOS violation test

Given a node relaxation solution vector `primal` and a constraint `sos` with ordered positions `0..n-1`:

1. Build the active position list:

```text
A = [ pos | pos in 0..n-1, |primal[sos.variables[pos]]| > tol ]
```

2. Test violation:

- For SOS1:
  - violated if `|A| > 1`
- For SOS2:
  - violated if `|A| > 2`
  - or if `|A| == 2` and the two positions are not adjacent:
    - `A[1] != A[0] + 1`

The tolerance used is `max(feasibility_tol, 1e-9)`.

## Split index selection

If a violated SOS constraint is found, the solver selects a split index:

- SOS1:
  - `split = floor((first + last) / 2)`
- SOS2:
  - `split = max(first + 1, floor((first + last) / 2))`

where `first = A.front()` and `last = A.back()`.

## Branch construction

The branching decision is built around the split position.

Let:

- `left_zero` be the variables forced to zero in the up-child branch
- `right_zero` be the variables forced to zero in the down-child branch

For SOS1:

```text
left_zero  = { sos.variables[pos] | pos <= split }
right_zero = { sos.variables[pos] | pos > split }
```

For SOS2:

```text
left_zero  = { sos.variables[pos] | pos < split }
right_zero = { sos.variables[pos] | pos > split }
```

The branch decision then sets:

```text
decision.variable = sos.variables[split]
decision.value = primal[decision.variable]
decision.down_child.state = make_upper_zero_child_state(node, right_zero)
decision.up_child.state   = make_upper_zero_child_state(node, left_zero)
```

## Semantics of the two child branches

- `up_child` forces all variables in `left_zero` to zero.
- `down_child` forces all variables in `right_zero` to zero.

This effectively splits the ordered SOS set into two portions and preserves the SOS structure by excluding one side of the ordering in each child.

## Notes

- The brancher does not simply fix a single variable to 0 or 1.
- Instead, it creates child domains that enforce zeroing out an entire side of the SOS ordered set.
- The actual child state creation is performed by `make_upper_zero_child_state(...)` in `include/bnb/diving.h`.

## Python API

The Python modeling API exposes this functionality through:

- `Model.add_sos1(vars, weights)` / `Model.addSOS1(vars, weights)`
- `Model.add_sos2(vars, weights)` / `Model.addSOS2(vars, weights)`

These bindings are declared in `bindings/model_bindings.cpp`.

## Example

For an SOS2 constraint on variables `[x, y, z]`, a violated relaxation with active positions `[0, 2]` will split at `split = max(0+1, floor((0+2)/2)) = 1`.

- `left_zero = [x]`
- `right_zero = [z]`

This produces two child branches:

- branch 1: force `x = 0`
- branch 2: force `z = 0`
