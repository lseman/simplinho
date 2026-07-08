# Dual engine: two-sided (native) bound support — known gap

## Status

Reverted from this session's working tree. `include/simplex/engine/simplex_dual.h`
is back to its pre-session state: `choose_dual_leaving` only detects a basic
variable going **below its lower bound** (`yB(i) < -tol`). It never detects a
basic variable going **above its upper bound**, and can declare a
primal-infeasible basis "optimal" when that happens.

Because of this gap, the drivers (`simplex_sparse_impl.h`,
`simplex_dense_impl.h`) route any solve that might invoke the dual engine
(`opt_.mode == Dual` or `Auto`) on a problem with finite upper bounds through
the old standard-form reformulation instead of the new native-bounds path.
Only `opt_.mode == Primal` uses native bounds when upper bounds are present
(the primal engine's two-sided ratio test is verified correct — see below).
This is implemented as `has_upper_bounds` + `opt_.mode != SimplexMode::Primal`
in both drivers' `use_reformulation` gate.

**This doc exists so a future session can pick up the fix without re-deriving
the failure modes from scratch.**

## What already works (do not re-touch without reason)

- `include/simplex/engine/simplex_primal.h`: full rewrite to native bounds.
  Two-sided Harris ratio test, nonbasic-at-upper tracking (`at_upper`), bound
  flips, incremental `rhs_eff`. Verified against `scipy.optimize.linprog`
  (HiGHS) on 100+ randomized bounded LPs (dense and sparse, negative lower
  bounds, free variables via the reformulation fallback, unbounded/infeasible
  detection, warm-start reuse). No known bugs.
- `include/simplex/factorization/crash.h`: `evaluate_basis_quality_` and the
  whole basis-selection chain (`choose_initial_basis_`, `find_initial_basis_`,
  `build_basis_attempt_`, `improve_basis_by_swaps_`) now take `l`, `u` and
  score primal/dual feasibility with real two-sided bound checks instead of
  assuming standard form (`x >= 0`). This was a real, separate bug (basis
  repair logic would swap in a bound-infeasible basis and call it better).
  Fixed and verified.
- `include/simplex/engine/phase1.h`: `basis_is_primal_feasible_` also takes
  `l`, `u` now, two-sided.
- Driver phase-1 → phase-2 handoff: phase 1 may finish with nonbasics resting
  at their upper bound; both drivers now compute a `phase2_seed_status`
  (`LPBasisStatus::AtUpper`/`AtLower` per column, from phase 1's solution) and
  feed it to phase 2 so phase 2 doesn't restart those nonbasics at the wrong
  bound. This was necessary once phase 1 itself started respecting upper
  bounds (see `l_phase1`/`u_phase1` construction in both drivers).

## The dual-engine bug, in detail

### Symptom

Given a bounded LP where the optimal basis requires a basic variable to sit
at (or need to pass through) its **upper** bound at some point in the
solve, forced `Dual` mode either:

1. Returns a **wrong "Optimal"** result — a basis that is not actually
   primal-feasible gets accepted because `choose_dual_leaving` never flags
   the offending row as infeasible. Confirmed reproducer: the second problem
   in `tests/test_bug001_bound_reformulation.py`
   (`test_sparse_dual_reformulation_validates_mapped_primal_before_accepting_optimal`),
   solved via the *native* bounds path instead of reformulation — dual mode
   returns `obj=1.844` when the correct optimum is `obj=-2.0004`.
2. Or, if you patch the leaving-row detection to also catch above-upper rows
   naively, **hangs** — see "Attempted fix and why it failed" below.

The existing driver-level safety net (`primal_feasible_` check on the
returned `x` before accepting an `Optimal` dual result — see
`simplex_sparse_impl.h` around `st == LPSolution::Status::Optimal`) catches
case 1 for `Auto` mode and retries with primal, so `Auto` self-heals. It does
**not** help when the caller explicitly forces `Dual` mode, because there is
no fallback mode to retry with in that branch of the driver.

### Root cause

`RevisedSimplexDualEngine::run()` in `simplex_dual.h` maintains basic variable
values as `yB = B^{-1} * rhs_eff`, where every basic variable is always
anchored at its **lower** bound (`view[j] = BoundView::Lower` is an enforced
invariant for every `j` currently in the basis — see the `for (int j : basis)
view[j] = BoundView::Lower;` blocks). So `yB(i) = xB(i) - l(basis[i])`, and
feasibility means `0 <= yB(i) <= u(basis[i]) - l(basis[i])`.

The leaving-row selection only checks the lower half of that:

```cpp
// pricer.h, all three DualXxxPricer::choose_dual_leaving overloads:
if (yB(i) >= -tol) continue;   // only catches yB(i) < 0
```

There is no corresponding check for `yB(i) > u(basis[i]) - l(basis[i])`. A
basic variable can walk arbitrarily far above its upper bound during dual
pivoting and the engine will never select it as a leaving row, eventually
declaring optimality with a primal-infeasible basis.

### Attempted fix and why it failed (this session)

The fix attempted was: build a `yB_infeas` vector that folds *both* violation
kinds into "negative means infeasible" (matching what the pricers already
expect), by computing `range = u - l` per basic row and setting
`yB_infeas(i) = -(yB(i) - range)` when `yB(i) > range`. Then, once a leaving
row is chosen and found to be an above-upper case, **re-anchor that row's
view to `Upper` before computing anything else** (mirroring the existing
nonbasic bound-flip machinery: negate the column in `Ahat`/`chat`, shift
`rhs_eff` by the anchor delta), so that in the flipped frame the row becomes
an ordinary below-lower violation and every downstream step (pricing, BFRT,
pivot, the `yB` rank-1 update) needs no special-casing.

This is the textbook-correct approach and got close, but had two distinct
bugs, found in sequence:

1. **First bug**: flipping the sign of `Ahat`'s column for the row's
   variable while that variable was still **basic** only updated `Ahat`/
   `chat` — it did **not** update the live factorized `B` inside
   `nla`/`FTBasis`, which had been built from the original (unflipped)
   column. `solve_B`/`solve_BT` read the factorization, not `Ahat`, so `yB`
   kept being computed against a stale, inconsistent sign. Result: the
   feasibility read for that row flip-flopped every iteration
   (`yB(1)` oscillating between e.g. `3.76` and `2.49` forever), producing
   wild objective swings (`obj` jumping between `217`, `-7028`, `1310`) and
   eventually a spurious `Singular` from too many failed re-pivots. **Fix**:
   force `refactor_basis()` immediately after the flip, so `B` is rebuilt
   from the now-consistent `Ahat`. This resolved the oscillation.

2. **Second bug** (present after fixing #1): the anchor-flip lives inside the
   dual engine's **inner** `while (true) { ... continue; }` retry loop (the
   loop used for solve_B-failure retries, bound-view updates for nonbasics,
   etc.) — a loop whose exit conditions all assume every `continue` either
   makes forward progress or is bounded by the *outer* iteration counter. The
   flip's `continue` restarts the **inner** loop without incrementing
   `iters`, so `opt_.max_iters` never catches a case where the flip needs to
   fire again (e.g. a second basic variable also above its bound, discovered
   only after the refactor). Confirmed: minimal reproducer
   (`test_sparse_dual_reformulation_validates_mapped_primal_before_accepting_optimal`'s
   LP, run with `disable_presolve=True`) hangs indefinitely — traced to the
   solve never printing a second `[dual] iter=` line after `iter=1
   basis_after=[3, 4]`, i.e. stuck inside the inner loop before the outer
   iteration's `trace_line_` calls.

Both fixes are real and probably both still needed in the eventual correct
implementation, but the combination did not reach a terminating, always-
correct state within this session, so the whole change was reverted.

### What the correct fix likely needs

- The anchor-flip-and-refactor approach is sound in principle but needs to
  **not** live inside the inner retry `while(true)` loop, or needs the loop
  restructured so a flip always counts toward `iters` (or is otherwise
  provably non-repeating for the same row within one outer iteration).
- Consider whether refactoring on *every* above-upper flip is even
  acceptable performance-wise for BnB-style repeated small solves — a full
  refactor per flip is expensive. HiGHS-style implementations avoid this by
  maintaining basic variables' bound status as a signed multiplier folded
  into the FTRAN/BTRAN machinery rather than mutating `Ahat` in place; that
  is a bigger, cleaner rewrite than the patch attempted here.
- Whatever the mechanism, add a dedicated regression test with a small,
  hand-verifiable LP where the optimal basis has a basic variable that must
  end at its upper bound under `SimplexMode.Dual` specifically (not just
  Auto/Primal) — the existing
  `test_sparse_dual_reformulation_validates_mapped_primal_before_accepting_optimal`
  case is a good one; consider disabling presolve in a variant of it so the
  bug reproduces without depending on presolve's specific reductions.
- Once fixed, revert the `has_upper_bounds && opt_.mode != SimplexMode::Primal`
  reformulation-fallback gate added to both drivers this session (search for
  "the dual engine's leaving-variable selection is not yet upper-bound-aware"
  in `simplex_sparse_impl.h` and `simplex_dense_impl.h`) so `Dual`/`Auto`
  also get the native-bounds fast path.

## Separate, pre-existing bug (not caused by this session, do not conflate)

While chasing the above, a second, unrelated failure surfaced on large
randomized bounded LPs (m~20-40, n~30-80): both `use_reformulation=true`
paths (sparse and dense) return `NeedPhase1` where HiGHS finds a feasible
optimum. **Confirmed via two isolated `git worktree` builds at the
pre-session `HEAD` that this reproduces identically on the untouched
baseline** — it is not a regression from this session's work, just newly
visible because more bounded-LP traffic now flows through modes that fall
back to the reformulation path. Left untouched; worth its own investigation
session. Repro: any of the `neglb*` / `big_sparse*` cases in this session's
ad hoc stress tests (randomized dense `A`, mixed finite lower/upper bounds,
`scipy.optimize.linprog(method="highs")` as reference) reliably trigger it
under `SimplexMode.Auto` or `SimplexMode.Dual`.
