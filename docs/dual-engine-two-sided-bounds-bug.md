# Dual engine: two-sided (native) bound support — known gap

## Status (updated: guarded sparse path recovered, native gate remains)

Current public-path status:

- The driver-level native-dual gate is still required:
  `has_upper_bounds && opt_.mode != SimplexMode::Primal` must continue using
  the bound reformulation instead of native two-sided dual handling.
- The documented randomized 38x55 sparse LP (`rng = np.random.default_rng(99)`)
  now solves correctly through the guarded sparse path. The sparse
  reformulation can still fail internally with `Singular` on this instance,
  so the sparse driver falls back to the dense implementation and annotates
  the result with `sparse_dense_fallback=1`. Verified objective:
  `-9.657217259529745` vs HiGHS `-9.657217259529757`.
- Native dual two-sided bounds remain unsafe. A signed-row leaving-pivot
  attempt removed the old no-pivot view-flip storm, but the native path still
  fails on larger random LPs, so it stays fenced off.
- A separate row-rank gap found by the small sparse random sweep at `seed=3`
  is fixed: dependent equality rows are reduced before solve and the returned
  solution is checked against the original system before accepting optimality.

A second fix attempt (`normalize_basic_views` in
`include/simplex/engine/simplex_dual.h`) replaced the first attempt described
below. It **fixes the original hang/wrong-answer repro** — verified:
`test_sparse_dual_native_bounds_handles_basic_upper_violation_without_presolve`
and the pre-existing `test_sparse_dual_reformulation_validates_mapped_primal_before_accepting_optimal`
both pass, no hang, correct `obj=-2.000434092862235`. All 7 tests in
`test_bug001_bound_reformulation.py` pass.

**But it introduces a new bug: cycling without progress on larger problems.**
Confirmed on a randomized 38×55 sparse LP (`rng = np.random.default_rng(99)`,
first trial of the stress-test loop in this doc's history — see repro below):
the outer loop runs for the full `max_iters` budget, terminates cleanly at
`IterLimit` (so it's not an infinite loop — `iters` does increment now, that
part of the second attempt is correct), but **the basis never changes**
across iterations. Verbose trace shows the identical `basis=[...]` array on
every one of 30+ traced iterations, while `normalize_basic_views` reports
"basic bound-view flips=N" (N between 4 and 27) on *every single iteration*.
The same set of rows keeps flipping Lower↔Upper forever without ever reaching
`choose_dual_leaving` / an actual pivot that changes the basis.

Net effect: the LP that previously (pre-session baseline, pre-existing
separate bug — see bottom section) at least returned `NeedPhase1` quickly now
burns the full iteration budget and returns `IterLimit` instead — slower and
still wrong, just wrong in a different way. Not safe to ship as-is; the
`has_upper_bounds && opt_.mode != SimplexMode::Primal` reformulation-fallback
gate in both drivers should **stay in place** until this is fixed.

A third attempt changed native dual leaving-row selection so a basic variable
above its upper bound is priced as an infeasible row with the dual row sign
negated, and the leaving variable is put into the nonbasis at its upper bound
after the pivot. This removes the `normalize_basic_views` no-pivot flip storm
on the randomized 38x55 repro: the basis changes every iteration and there
are no `"basic bound-view flips"` trace lines. However, that native path still
does **not** solve the repro reliably; with a larger iteration budget it
eventually returns `Singular` / `"dual: no eligible entering"` with a
non-feasible returned vector, while HiGHS finds the optimum. Therefore the
driver-level `has_upper_bounds && opt_.mode != SimplexMode::Primal`
reformulation gate remains required for correctness.

**This doc exists so a future session can pick up the fix without re-deriving
the failure modes from scratch.**

## Repro for the new cycling bug

```python
import numpy as np, scipy.sparse as sp
import simplinho as sx

rng = np.random.default_rng(99)
m, n = rng.integers(15, 40), rng.integers(30, 80)   # first draw: m=38, n=55
density = rng.uniform(0.2, 0.6)
Ad = rng.normal(size=(m, n)) * (rng.uniform(size=(m, n)) < density)
l = np.where(rng.uniform(size=n) < 0.3, rng.uniform(-3, 0, n), 0.0)
u = l + rng.uniform(0.3, 4.0, n)
free_mask = rng.uniform(size=n) < 0.1
u[free_mask] = np.inf
x0 = l + np.where(np.isfinite(u), (u - l), 2.0) * rng.uniform(0.1, 0.9, n)
b = Ad @ x0
c = rng.normal(size=n)

o = sx.RevisedSimplexOptions()
o.mode = sx.SimplexMode.Dual
o.max_iters = 30       # small cap to inspect quickly; default 50000 just burns time
o.verbose = True
s = sx.RevisedSimplex(o).solve(sp.csc_matrix(Ad), b, c, l, u)
print(s.status)        # IterLimit
for ln in s.log.splitlines():
    print(ln)           # every "[dual] iter=N basic bound-view flips=K" line
                         # shows the SAME basis array, K in [4, 27], forever
```

Reference: `scipy.optimize.linprog(c, A_eq=Ad, b_eq=b, bounds=list(zip(l, u)),
method="highs")` finds `obj=-9.657217259529757` on this instance — so it is a
feasible, bounded LP; the dual engine should converge on it.

### Likely cause

`normalize_basic_views` recomputes `basic_above_range_rows(yB)` fresh each
outer-loop pass using whatever `yB` was refreshed at the top of that pass. If
the flip of a row's view causes `yB` (after refresh) to again read as
"above range" for that same row — e.g. because the anchor-shift math has an
off-by-range-sign error, or because `rebuild_nla()` + a stale `rhs_eff`
combination doesn't actually move the basic value across the boundary the
way a real pivot would — the row gets flagged and flipped again next
iteration, forever. A genuine dual pivot (entering/leaving variable swap via
`choose_dual_leaving` + BFRT) is required to actually change `xB`; a pure
anchor-view flip only changes which bound a basic variable is *measured
against* (`yB(i) := xB(i) - anchor`), it does not change `xB(i)` itself. So
if the row was above range before the flip, it is very likely to look
"above range" again immediately after — the flip alone doesn't fix
infeasibility, only pivoting the row out of the basis does.

This suggests the fundamental design gap: **`normalize_basic_views` treats
"basic variable above upper bound" as something fixable by re-anchoring
alone**, but re-anchoring a still-basic row doesn't change its numeric value
— it only relabels which bound violation direction future iterations will
report for it. The row still needs to actually **leave the basis** via a
real dual pivot before the infeasibility is resolved. Whether the flip should
happen at all before a pivot (vs. only as part of computing `w` for a
*chosen* leaving row, as the first attempt below tried) is the crux to
resolve.

### What to check next

- Instrument `basic_above_range_rows` to print `yB(i)` and `range` for one
  specific row across consecutive iterations — confirm whether the same row
  keeps reporting the same or a growing violation after each flip (would
  confirm the flip is not moving `xB` at all, just toggling the label).
- Compare against the first attempt's approach (below): that one flipped the
  view **only for the chosen leaving row**, immediately followed by a real
  pivot attempt in the same outer iteration (once the inner-loop-vs-iters bug
  was fixed) — never flipping N rows speculatively before any pivot. The
  new `normalize_basic_views` flips *all* above-range rows every iteration
  before pricing even runs, which may be why it never reaches a pivot: by the
  time flips are done, `yB` may already show a different set of rows over
  range (rounding / recompute drift), restarting the cycle.
- Consider bounding `normalize_basic_views` to fire at most once per outer
  iteration for a given row-set fingerprint, and forcing a real
  `choose_dual_leaving` + pivot attempt immediately after any flip round
  (not just a `continue` back to the top).

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

### First attempt and why it failed (superseded by the anchor-flip-based
    `normalize_basic_views` approach above, which fixes the hang but has its
    own cycling bug)

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
