# Presolve in Simplinho

These notes are written as a classroom lecture on presolve. The goal is not
only to describe what Simplinho does, but also to explain why the reductions are
valid and what mathematical invariant makes presolve safe.

We will move in four layers:

1. The optimization model and the main presolve invariant.
2. The mathematical reductions used in LP presolve.
3. The additional ideas needed for MIP presolve.
4. How the theory is reflected in Simplinho's implementation.

The examples are deliberately small. They are meant to be worked by hand before
looking at the code.

## Learning Goals

After studying this note, a student should be able to:

- compute row activity bounds for a linear constraint,
- decide when a row or column proves infeasibility or unboundedness,
- derive implied bounds from a constraint,
- explain why fixed-variable substitution preserves the original problem,
- understand why postsolve must run reductions in reverse order,
- distinguish LP presolve from MIP presolve,
- describe the cost tradeoff between cheap node presolve and stronger root
  presolve.

## The Model

The LP presolver works with the problem

$$
\min_x \; c^T x + c_0
$$

subject to row constraints

$$
a_i^T x \le b_i, \quad a_i^T x = b_i, \quad \text{or} \quad a_i^T x \ge b_i,
$$

and variable bounds

$$
l_j \le x_j \le u_j.
$$

In code, this is represented by `presolve::LP`:

- `A`: the constraint matrix.
- `b`: the right hand side vector.
- `sense`: one row sense per row, either `LE`, `EQ`, or `GE`.
- `c`: the linear objective coefficients.
- `l`, `u`: lower and upper bounds.
- `c0`: the constant objective shift.

We allow infinite bounds. For example, $u_j = +\infty$ means that the variable
has no explicit upper bound. This is important: replacing infinity by a very
large number changes the mathematical problem and can create numerical trouble.

## What Presolve Tries To Do

Presolve is the part of an optimizer that tries to simplify a model before the
main algorithm starts. A good presolver can:

- prove infeasibility before simplex or branch-and-bound begins,
- prove unboundedness early,
- remove redundant equations and variables,
- tighten variable bounds,
- reduce numerical scaling problems,
- leave enough information to reconstruct a solution of the original model.

The last item is the one that makes presolve subtle. A simplified model is only
useful if it remains connected to the original model.

## The Central Invariant

The central presolve invariant is:

> Every reduction must either prove the model status, preserve the model
> exactly, or preserve the optimal value while storing enough information to
> reconstruct an original optimal solution from a reduced optimal solution.

There are three common outcomes of a presolve step:

1. It proves infeasibility.
2. It proves unboundedness.
3. It produces a smaller or tighter equivalent problem plus a postsolve action.

The third case is the normal one.

### Formal View

Let the original problem be $P$ and let a reduction transform it into $P'$. A
safe reduction must satisfy one of the following:

- $P$ is infeasible, and the reduction proves that fact.
- $P$ is unbounded, and the reduction proves that fact.
- There is a mapping $R$ from optimal solutions of $P'$ back to optimal
  solutions of $P$ such that objective values are preserved after accounting
  for the objective shift.

For the last case, if $x'$ solves the reduced model to optimality, postsolve
computes

$$
x = R(x').
$$

If the reduction changed the constant objective term, we require

$$
c^T x + c_0 = {c'}^T x' + c_0'.
$$

This is why presolve stores actions.

## Why Postsolve Runs Backward

Suppose presolve fixes a variable:

$$
x_3 = 5.
$$

The reduced problem may no longer contain the original column for $x_3$, or the
column may be kept but zeroed with equal lower and upper bounds. After solving
the reduced problem, we still need a solution vector for the original
variables. Therefore presolve records an action:

```text
ActFixVar(index = 3, value = 5)
```

If presolve applies transformations

$$
P_1, P_2, \ldots, P_k,
$$

then postsolve must apply the inverse information in reverse order:

$$
P_k^{-1}, P_{k-1}^{-1}, \ldots, P_1^{-1}.
$$

This is the same rule used when undoing algebraic substitutions. The last
substitution performed is the first one that must be undone.

### Proof Sketch

Let $T_i$ be the map from problem $P_{i-1}$ to problem $P_i$. A solution of the
final problem $P_k$ lives after all transformations have been applied. To get
back to $P_0$, we must first undo $T_k$, then $T_{k-1}$, and so on:

$$
x_0 = T_1^{-1}(T_2^{-1}(\cdots T_k^{-1}(x_k)\cdots)).
$$

This composition is only valid in reverse order.

## High-Level LP Presolve Pipeline

The LP presolver in `include/simplex/presolver.h` follows this shape:

```text
function presolve_lp(P):
    validate dimensions

    optionally relax inactive huge bounds
    check l <= u
    detect simple unbounded rays

    optionally scale rows
    optionally scale columns

    optionally remove dependent equality rows

    build sparse row/column index

    for pass = 1 to max_passes:
        detect simple unbounded rays
        remove zero rows
        remove zero columns
        detect fixed variables
        eliminate singleton rows
        tighten bounds from row activity
        run domain propagation
        apply dual fixing by row locks
        optionally substitute singleton columns
        optionally run objective-guided probing
        merge duplicate or redundant rows
        optionally eliminate doubleton equalities

        stop early if no pass changed the model

    detect simple unbounded rays again
    remove remaining zero rows
    return reduced problem and postsolve stack
```

The exact set of active passes depends on `Presolver::Options`. Some stronger
passes, such as structural column substitution and doubleton equation
elimination, are opt-in.

The solver-facing default is conservative. The LP presolver is usually
non-destructive: it can zero columns, fix bounds, scale rows, and tighten
bounds, but it avoids structural column removal unless explicitly enabled. This
keeps basis handling and postsolve simpler.

## Bound Checks

The simplest infeasibility check is

$$
l_j > u_j.
$$

If this happens for any variable, the model has no feasible point.
Numerically, we test with a tolerance:

$$
u_j + \epsilon < l_j.
$$

### Proposition

If there exists an index $j$ such that $l_j > u_j$, then the feasible set is
empty.

### Proof

A feasible point must satisfy both $x_j \ge l_j$ and $x_j \le u_j$. If
$l_j > u_j$, no real number can satisfy both inequalities at the same time.
Therefore no feasible point exists.

### Example

Consider one variable with

$$
2 \le x \le 1.999999999.
$$

If the tolerance is $\epsilon = 10^{-9}$, this may be treated as a tiny
rounding issue. If the gap is larger, for example

$$
2 \le x \le 1.5,
$$

then presolve can immediately return infeasible.

## Row Activity

Many presolve reductions are based on the minimum and maximum possible activity
of a row. For a row

$$
a_i^T x = \sum_j a_{ij} x_j,
$$

with bounds $l_j \le x_j \le u_j$, define

$$
L_i = \min_{l \le x \le u} a_i^T x,
$$

and

$$
U_i = \max_{l \le x \le u} a_i^T x.
$$

Because the row is linear and the domain is a box, each variable contributes at
one of its bounds:

$$
L_i =
\sum_{a_{ij} \ge 0} a_{ij} l_j
+
\sum_{a_{ij} < 0} a_{ij} u_j,
$$

and

$$
U_i =
\sum_{a_{ij} \ge 0} a_{ij} u_j
+
\sum_{a_{ij} < 0} a_{ij} l_j.
$$

If an infinite bound is involved, the activity can be infinite. The
implementation tracks this carefully instead of pretending that infinity is a
large number.

### Proposition

For a linear function over a box, the minimum and maximum are attained at
bounds of the variables.

### Proof

Write the row as a sum of independent one-variable terms:

$$
\sum_j a_j x_j.
$$

For a fixed coefficient $a_j$, the term $a_j x_j$ is increasing in $x_j$ when
$a_j > 0$ and decreasing in $x_j$ when $a_j < 0$. Therefore its minimum is
obtained at $l_j$ for a positive coefficient and at $u_j$ for a negative
coefficient. The maximum uses the opposite bound. Since the variables are only
coupled through addition, summing these best individual choices gives the row
minimum and maximum.

### Example

Let

$$
2x_1 - 3x_2 \le 10,
$$

with

$$
0 \le x_1 \le 4, \quad 1 \le x_2 \le 5.
$$

The minimum row activity is

$$
L = 2 \cdot 0 - 3 \cdot 5 = -15.
$$

The maximum row activity is

$$
U = 2 \cdot 4 - 3 \cdot 1 = 5.
$$

Since $U = 5 \le 10$, the row is always satisfied and can be removed.

## Zero Rows

A zero row has all coefficients equal to zero:

$$
0^T x \le b_i, \quad 0^T x = b_i, \quad \text{or} \quad 0^T x \ge b_i.
$$

The row either proves infeasibility or is redundant.

For a `LE` row, the row is feasible if

$$
0 \le b_i.
$$

For an equality row, it is feasible if

$$
0 = b_i.
$$

For a `GE` row, it is feasible if

$$
0 \ge b_i.
$$

### Proof

A zero row has no variables left. Its truth value is just the truth value of a
constant statement. If that constant statement is true, every feasible point
satisfies the row, so it is redundant. If it is false, no feasible point can
satisfy the row, so the model is infeasible.

### Example

The row

$$
0x_1 + 0x_2 \le 7
$$

is redundant. The row

$$
0x_1 + 0x_2 = 3
$$

is impossible, so presolve can stop with infeasible.

## Zero Columns

A zero column means variable $x_j$ appears in no constraint. The variable only
matters through its objective coefficient and bounds:

$$
\min \; c_j x_j.
$$

If $c_j > 0$, minimizing wants the smallest possible value, so we choose

$$
x_j = l_j
$$

when $l_j$ is finite. If $l_j = -\infty$, the model is unbounded.

If $c_j < 0$, minimizing wants the largest possible value, so we choose

$$
x_j = u_j
$$

when $u_j$ is finite. If $u_j = +\infty$, the model is unbounded.

If $c_j = 0$, any value in the interval is optimal for that variable. The
presolver can pick a convenient value, usually one of the finite bounds or zero
if it is allowed.

### Proposition

For a zero column with a finite improving bound, fixing the variable at its
best bound preserves optimality. With no finite improving bound, the LP is
unbounded.

### Proof

Since the column appears in no row, changing $x_j$ cannot affect feasibility.
Only $c_j x_j$ changes. If $c_j > 0$, the expression is minimized by decreasing
$x_j$; if no lower bound exists, the objective tends to $-\infty$. If a finite
lower bound exists, $x_j = l_j$ is best. The case $c_j < 0$ is symmetric. If
$c_j = 0$, the objective is independent of $x_j$.

### Example

Minimize

$$
5x_1 - 2x_2
$$

where neither variable appears in any row and

$$
0 \le x_1 \le 10, \quad 0 \le x_2 \le 4.
$$

The best values are

$$
x_1 = 0, \quad x_2 = 4.
$$

The objective contribution is

$$
5 \cdot 0 - 2 \cdot 4 = -8.
$$

## Fixed Variables

If a variable has equal lower and upper bounds,

$$
l_j = u_j = \bar{x}_j,
$$

then the variable is fixed. Substitute it into every row:

$$
a_i^T x = a_{ij}\bar{x}_j + \sum_{k \ne j} a_{ik}x_k.
$$

Move the fixed contribution to the right hand side:

$$
\sum_{k \ne j} a_{ik}x_k \; \{\le,=,\ge\} \; b_i - a_{ij}\bar{x}_j.
$$

The objective shift changes by

$$
c_0 \leftarrow c_0 + c_j\bar{x}_j.
$$

In non-destructive LP presolve, Simplinho often keeps the variable position but
zeros the column and records the fixed value for postsolve.

### Proposition

Fixed-variable substitution preserves the feasible set projected onto the
remaining variables and preserves objective values after the objective shift.

### Proof

Every feasible solution of the original problem has $x_j = \bar{x}_j$ because
the lower and upper bounds are equal. Substituting this value into each row is
therefore just algebraic simplification of a condition that every feasible
solution already satisfies. The objective term $c_j x_j$ becomes the constant
$c_j \bar{x}_j$, so adding it to $c_0$ preserves the objective value.

### Example

Consider

$$
3x_1 + 2x_2 = 11,
$$

and suppose bounds imply

$$
x_2 = 4.
$$

Then

$$
3x_1 + 2 \cdot 4 = 11,
$$

so the reduced row is

$$
3x_1 = 3.
$$

Thus $x_1 = 1$.

## Singleton Rows

A singleton row has exactly one nonzero coefficient:

$$
a_{ij}x_j \; \{\le,=,\ge\} \; b_i.
$$

For an equality row,

$$
a_{ij}x_j = b_i,
$$

we get

$$
x_j = \frac{b_i}{a_{ij}}.
$$

For inequalities, we get a bound. The sign of $a_{ij}$ matters.

For

$$
a_{ij}x_j \le b_i,
$$

if $a_{ij} > 0$ then

$$
x_j \le \frac{b_i}{a_{ij}},
$$

but if $a_{ij} < 0$ then

$$
x_j \ge \frac{b_i}{a_{ij}}.
$$

For

$$
a_{ij}x_j \ge b_i,
$$

if $a_{ij} > 0$ then

$$
x_j \ge \frac{b_i}{a_{ij}},
$$

but if $a_{ij} < 0$ then

$$
x_j \le \frac{b_i}{a_{ij}}.
$$

### Proof

The result follows from dividing an inequality by $a_{ij}$. Division by a
positive number preserves the inequality direction. Division by a negative
number reverses it. Equality division is valid for any nonzero coefficient.

### Example

The row

$$
-2x_1 \le -6
$$

must be divided by a negative number, which reverses the inequality:

$$
x_1 \ge 3.
$$

If the old bounds were

$$
0 \le x_1 \le 10,
$$

the new bounds are

$$
3 \le x_1 \le 10.
$$

## General Implied Bound Tightening

Singleton rows are only the easiest case. A row with many variables can also
tighten bounds.

For a `LE` row,

$$
\sum_j a_jx_j \le b,
$$

isolate variable $x_k$:

$$
a_kx_k + \sum_{j \ne k} a_jx_j \le b.
$$

Let $L_{-k}$ be the minimum activity of all variables except $x_k$:

$$
L_{-k} =
\min_{l \le x \le u}
\sum_{j \ne k} a_jx_j.
$$

If $a_k > 0$, then the largest valid value of $x_k$ must satisfy

$$
a_kx_k + L_{-k} \le b,
$$

so

$$
x_k \le \frac{b - L_{-k}}{a_k}.
$$

If $a_k < 0$, dividing by a negative coefficient gives a lower bound:

$$
x_k \ge \frac{b - L_{-k}}{a_k}.
$$

For a `GE` row, use the maximum activity of the other variables:

$$
U_{-k} =
\max_{l \le x \le u}
\sum_{j \ne k} a_jx_j.
$$

If $a_k > 0$, then

$$
x_k \ge \frac{b - U_{-k}}{a_k}.
$$

If $a_k < 0$, then

$$
x_k \le \frac{b - U_{-k}}{a_k}.
$$

For equality rows, both sides are active. Equality rows can imply both lower
and upper bounds.

### Proposition

The implied bound formulas above are valid consequences of the original row:
they remove no feasible solution.

### Proof

Consider a `LE` row and coefficient $a_k > 0$. For every feasible assignment of
the other variables, their contribution is at least $L_{-k}$. Therefore every
feasible point satisfies

$$
a_kx_k + L_{-k} \le a_kx_k + \sum_{j \ne k}a_jx_j \le b.
$$

Thus $x_k \le (b - L_{-k})/a_k$. The proof for $a_k < 0$ is the same, except
division by a negative number reverses the inequality. The `GE` case uses
$U_{-k}$ because the other variables can contribute at most that value.

### Example: Tightening an Upper Bound

Suppose

$$
2x_1 + x_2 \le 10,
$$

with

$$
0 \le x_1 \le 100, \quad 3 \le x_2 \le 8.
$$

To tighten $x_1$, compute the minimum contribution of $x_2$:

$$
L_{-1} = 3.
$$

Then

$$
2x_1 + 3 \le 10,
$$

so

$$
x_1 \le 3.5.
$$

The new bound is

$$
0 \le x_1 \le 3.5.
$$

### Example: Tightening a Lower Bound

Suppose

$$
-3x_1 + x_2 \le 4,
$$

with

$$
0 \le x_1 \le 10, \quad 1 \le x_2 \le 5.
$$

For $x_1$, the coefficient is negative. Use the minimum contribution of
$x_2$:

$$
L_{-1} = 1.
$$

Then

$$
-3x_1 + 1 \le 4.
$$

This gives

$$
-3x_1 \le 3.
$$

Dividing by $-3$ reverses the inequality:

$$
x_1 \ge -1.
$$

This does not improve the old bound $x_1 \ge 0$, so presolve keeps the old
bound.

## Domain Propagation

Domain propagation repeatedly applies implied bound tightening. A new bound in
one row can trigger a stronger bound in another row.

```text
function propagate_bounds(rows, l, u):
    mark all rows dirty

    while there are dirty rows and pass limit is not reached:
        for each dirty row i:
            compute implied bounds from row i
            for each improved variable bound:
                update l or u
                mark neighboring rows dirty

            if some l[j] > u[j]:
                return infeasible

    return tightened bounds
```

### Fixed-Point Interpretation

Each row defines an operator that maps current bounds to tighter bounds. Domain
propagation applies these operators until no operator changes the bounds, or
until a pass limit is reached. A state where no bound changes is a local fixed
point of the propagation rules.

This is not the same as solving the LP. It is cheaper and weaker. It uses only
bound reasoning, not arbitrary linear combinations of rows.

### Example: A Propagation Chain

Consider

$$
x_1 + x_2 \le 5,
$$

$$
x_2 + x_3 \le 4,
$$

with

$$
x_1 \ge 3, \quad x_2 \ge 0, \quad x_3 \ge 0.
$$

From the first row:

$$
x_2 \le 5 - x_1 \le 2.
$$

Now the second row becomes stronger:

$$
x_3 \le 4 - x_2.
$$

Using $x_2 \ge 0$ gives $x_3 \le 4$. If later another row raises $x_2$ to
$1.5$, then propagation can improve this to

$$
x_3 \le 2.5.
$$

This is why presolve uses multiple passes.

## Redundant Rows

A row is redundant if every point satisfying the bounds also satisfies the row.

For a `LE` row,

$$
a_i^Tx \le b_i
$$

is redundant if

$$
U_i \le b_i.
$$

For a `GE` row, it is redundant if

$$
L_i \ge b_i.
$$

For an equality row, it is redundant if both

$$
L_i = b_i
$$

and

$$
U_i = b_i
$$

hold within tolerance.

Rows can also prove infeasibility:

- `LE`: infeasible if $L_i > b_i$.
- `GE`: infeasible if $U_i < b_i$.
- `EQ`: infeasible if $L_i > b_i$ or $U_i < b_i$.

### Proof

For a `LE` row, $U_i$ is the largest possible left hand side under the current
bounds. If even this largest value satisfies $U_i \le b_i$, then every bounded
point satisfies the row. If the smallest possible value is already larger than
$b_i$, then all points violate the row. The `GE` case is symmetric, and an
equality requires the entire possible activity interval to collapse to the
right hand side.

## Duplicate Rows

Two rows are duplicates if one is a scalar multiple of the other. For example,

$$
x_1 + 2x_2 \le 6
$$

and

$$
2x_1 + 4x_2 \le 12
$$

represent the same inequality.

If the right hand sides disagree,

$$
x_1 + 2x_2 = 6
$$

and

$$
2x_1 + 4x_2 = 10,
$$

then the second row means

$$
x_1 + 2x_2 = 5,
$$

which conflicts with the first. Presolve can report infeasibility.

### Theory

For equality rows, two rows

$$
a^T x = b, \quad \alpha a^T x = d
$$

with $\alpha \ne 0$ are consistent if and only if

$$
d = \alpha b.
$$

For inequality rows, duplicate detection must also respect the row sense and
the sign of the multiplier. Multiplying by a negative scalar reverses `LE` and
`GE`.

## Scaling

Scaling tries to improve numerical conditioning. The LP presolver can scale
rows so their largest coefficient is close to one. For a row scale $s_i > 0$,

$$
a_i^Tx \; \{\le,=,\ge\} \; b_i
$$

becomes

$$
\frac{1}{s_i}a_i^Tx \; \{\le,=,\ge\} \; \frac{b_i}{s_i}.
$$

The implementation chooses powers of two when practical. Powers of two are
friendly to binary floating point because they usually do not introduce extra
rounding in the mantissa.

### Proof of Equivalence

Because $s_i > 0$, dividing both sides by $s_i$ preserves the row sense. The
set of $x$ satisfying the scaled row is exactly the set satisfying the original
row.

### Example

The row

$$
1000x_1 + 2000x_2 \le 3000
$$

can be scaled by $s = 1000$:

$$
x_1 + 2x_2 \le 3.
$$

This row is mathematically equivalent, but it is easier for floating point
linear algebra.

## Dependent Equality Rows

For a pure equality system

$$
Ax = b,
$$

some rows may be linear combinations of other rows. If row $r_2$ is just
$2r_1$, it adds no new information.

Example:

$$
x_1 + x_2 = 3,
$$

$$
2x_1 + 2x_2 = 6.
$$

The second row is redundant. But

$$
2x_1 + 2x_2 = 7
$$

would be inconsistent with the first row.

Simplinho can optionally use rank-revealing QR or SVD-style logic to remove
dependent equality rows. The check is based on whether $b$ lies in the column
space implied by the independent rows. Conceptually, if $U_r$ spans the row
space, then the consistent part of $b$ is

$$
U_rU_r^Tb.
$$

If

$$
\|b - U_rU_r^Tb\|
$$

is too large, the equality system is infeasible.

### Linear Algebra Foundation

Rows of $A$ define equations. A dependent row adds no information if its right
hand side has the same dependency relation. If

$$
r_k = \sum_{i < k}\alpha_i r_i,
$$

then consistency requires

$$
b_k = \sum_{i < k}\alpha_i b_i.
$$

If this condition holds, the row is redundant. If it fails, the equality system
has no solution.

## Unboundedness Detection

A variable with an improving objective direction and no constraint blocking
that direction can prove the model unbounded.

For minimization:

- If $c_j < 0$, increasing $x_j$ improves the objective.
- If $c_j > 0$, decreasing $x_j$ improves the objective.

If $c_j < 0$, $u_j = +\infty$, and no row blocks increasing $x_j$, then

$$
x_j \rightarrow +\infty
$$

drives

$$
c_jx_j \rightarrow -\infty.
$$

Similarly, if $c_j > 0$, $l_j = -\infty$, and no row blocks decreasing $x_j$,
the model is unbounded.

### What Blocks a Direction?

For a `LE` row:

$$
a_i^Tx \le b_i,
$$

increasing $x_j$ is blocked if $a_{ij} > 0$, because the row activity rises.
Decreasing $x_j$ is blocked if $a_{ij} < 0$, because the row activity rises.

For a `GE` row, the signs are reversed. Equality rows block both directions
unless the coefficient is zero.

### Proof Sketch

Let $d$ be the direction that changes only coordinate $j$. If moving along $d$
does not violate any row and no finite bound stops movement in that direction,
then $x + td$ is feasible for all $t \ge 0$. If $c^T d < 0$, then

$$
c^T(x + td) + c_0 = c^T x + c_0 + t c^T d \rightarrow -\infty.
$$

Therefore the minimization problem is unbounded.

### Example

Minimize

$$
-x_1
$$

subject to

$$
x_1 \ge 0.
$$

There is no upper bound on $x_1$. Increasing $x_1$ keeps the row feasible and
improves the objective:

$$
-x_1 \rightarrow -\infty.
$$

The model is unbounded.

## Dual Fixing by Locks

Dual fixing uses objective signs and row locks. The idea is:

- If $c_j > 0$, the objective prefers smaller $x_j$.
- If moving $x_j$ down cannot violate any row, fix $x_j$ at $l_j$.
- If $c_j < 0$, the objective prefers larger $x_j$.
- If moving $x_j$ up cannot violate any row, fix $x_j$ at $u_j$.

A row creates a lock when movement in a direction can make that row infeasible.
Equality rows lock both directions. Inequality rows lock directions according
to coefficient sign.

### Proposition

If the objective prefers a finite bound and moving toward that bound cannot
violate any constraint, then there exists an optimal solution with the variable
fixed at that bound.

### Proof

Assume $c_j > 0$ and decreasing $x_j$ cannot violate any row. Given any
feasible solution with $x_j > l_j$, lower $x_j$ until it reaches $l_j$. The
result remains feasible by the no-lock assumption. The objective decreases by
$c_j(x_j - l_j)$, or stays the same only if the variable was already at the
bound. Thus an optimum can be chosen with $x_j = l_j$. The upper-bound case is
symmetric.

### Example

Minimize

$$
4x_1
$$

subject to

$$
x_1 + x_2 \le 10,
$$

and

$$
0 \le x_1 \le 5, \quad 0 \le x_2 \le 5.
$$

The objective wants $x_1$ smaller. Decreasing $x_1$ makes

$$
x_1 + x_2
$$

smaller, so it cannot violate the `LE` row. Therefore presolve can fix

$$
x_1 = 0.
$$

## Singleton Column Substitution

This is a structural pass and is not enabled in the conservative default LP
mode. Suppose variable $x_j$ appears in only one equality row:

$$
a_{ij}x_j + \sum_{k \ne j}a_{ik}x_k = b_i.
$$

If $a_{ij} \ne 0$, solve for $x_j$:

$$
x_j =
\frac{b_i}{a_{ij}}
-
\sum_{k \ne j}\frac{a_{ik}}{a_{ij}}x_k.
$$

Substitute this expression into the objective:

$$
c_jx_j + \sum_{k \ne j}c_kx_k.
$$

The objective shift becomes

$$
c_0 \leftarrow c_0 + c_j\frac{b_i}{a_{ij}},
$$

and for each $k \ne j$,

$$
c_k \leftarrow c_k - c_j\frac{a_{ik}}{a_{ij}}.
$$

### Postsolve Formula

After solving the reduced problem, postsolve reconstructs

$$
x_j =
\frac{b_i}{a_{ij}}
-
\sum_{k \ne j}\frac{a_{ik}}{a_{ij}}x_k.
$$

The row used for substitution is then automatically satisfied.

### Example

Let

$$
2x_1 + x_2 = 8.
$$

Solve for $x_2$:

$$
x_2 = 8 - 2x_1.
$$

If the objective is

$$
3x_1 + 5x_2,
$$

then substitution gives

$$
3x_1 + 5(8 - 2x_1) = 40 - 7x_1.
$$

So the reduced objective has

$$
c_0 = 40, \quad c_1 = -7.
$$

## Doubleton Equality Elimination

A doubleton equality has exactly two nonzero coefficients:

$$
a_px_p + a_qx_q = b.
$$

One variable can be expressed using the other:

$$
x_p = \frac{b - a_qx_q}{a_p}.
$$

This can reduce the number of variables, but it must be used carefully because
it can make rows denser. Simplinho keeps this pass optional.

### Bound Transfer

If $x_p$ has bounds

$$
l_p \le x_p \le u_p,
$$

then the substitution creates bounds on $x_q$:

$$
l_p \le \frac{b - a_qx_q}{a_p} \le u_p.
$$

Solving this two-sided inequality gives additional implied bounds. The signs of
$a_p$ and $a_q$ determine which inequalities reverse.

### Example

The equality

$$
x_1 + 2x_2 = 6
$$

gives

$$
x_1 = 6 - 2x_2.
$$

If

$$
0 \le x_1 \le 10,
$$

then

$$
0 \le 6 - 2x_2 \le 10.
$$

This implies

$$
-4 \le -2x_2 \le 6.
$$

Dividing by $-2$ reverses the inequalities:

$$
2 \ge x_2 \ge -3.
$$

So the useful bound is

$$
x_2 \le 2.
$$

## Huge Bound Relaxation

Some models use large finite numbers as fake infinity, for example

$$
-10^{20} \le x_j \le 10^{20}.
$$

Such bounds can damage numerical behavior. If the model itself implies a much
smaller finite bound, or if a huge bound is inactive, presolve can relax the
huge bound to true infinity:

$$
10^{20} \quad \leadsto \quad +\infty.
$$

This pass is optional because it changes how the solver interprets
user-provided large constants. When enabled, it uses row-implied bounds to
decide whether a huge explicit bound is inactive.

### Numerical Foundation

Interior-point methods, simplex ratio tests, and scaling routines all behave
better when bounds represent real modeling information instead of artificial
large constants. A coefficient near $1$ combined with a bound near $10^{20}$
can create row activity ranges that dwarf meaningful right hand sides.

### Example

Suppose a user gives

$$
0 \le x \le 10^{20},
$$

and the row

$$
x \le 7.
$$

The upper bound $10^{20}$ is not useful. The row already implies the real upper
bound $7$.

## Objective-Guided Probing

Objective-guided probing is disabled by default in LP presolve. It chooses
variables that have large objective impact and tries temporary bound changes.
If a temporary choice makes the model infeasible, the opposite bound can often
be inferred.

```text
function objective_guided_probe(P):
    choose variables with large |c[j]| times bound width

    for each candidate j:
        try fixing x[j] to l[j] in a copy of P
        run cheap bound propagation

        try fixing x[j] to u[j] in a copy of P
        run cheap bound propagation

        if both branches infeasible:
            return infeasible

        if one branch infeasible:
            apply the opposite fixing to the real model

        if both branches imply the same bound:
            apply that shared bound to the real model
```

This can be powerful, but it costs more than simple row activity checks.

### Logical Foundation

Probing is proof by contradiction. If assuming $x_j = l_j$ makes the model
infeasible, then every feasible solution must satisfy $x_j \ne l_j$. In integer
or bounded settings this can imply the opposite value or a stronger bound.

## MIP Presolve

The MIP presolver builds on the same ideas, but it also uses integrality. It
appears mainly in `src/bnb/mip_presolve.cpp` and `include/bnb/mip_presolve.h`.

For integer variables, bounds are rounded:

$$
l_j \leftarrow \lceil l_j \rceil,
$$

and

$$
u_j \leftarrow \lfloor u_j \rfloor.
$$

For binary variables, the domain is also restricted to

$$
x_j \in \{0,1\}.
$$

If rounding creates

$$
l_j > u_j,
$$

the MIP is infeasible.

### Proposition

Rounding integer bounds inward removes no integer feasible point.

### Proof

If $x_j$ must be integer and $x_j \ge l_j$, then $x_j$ must also satisfy
$x_j \ge \lceil l_j \rceil$. Similarly, if $x_j \le u_j$, then
$x_j \le \lfloor u_j \rfloor$. Therefore inward rounding preserves exactly the
integer points in the interval.

### Example

Let $x$ be integer and suppose branching or row propagation gives

$$
1.2 \le x \le 3.8.
$$

Then MIP presolve tightens this to

$$
2 \le x \le 3.
$$

## MIP Node Bound Presolve

At a branch-and-bound node, Simplinho has local bounds from branching
decisions. Node presolve propagates those bounds through the base constraints
and any extra cuts.

```text
function presolve_mip_node_bounds(problem, lower, upper, cuts):
    round integer and binary bounds

    build sparse rows from base constraints and cuts
    build column-to-row adjacency
    mark all rows dirty

    for a small number of propagation rounds:
        for each dirty row:
            tighten bounds implied by that row
            mark neighboring rows dirty when a bound changes

        stop if nothing changed

    return tightened bounds or infeasible
```

The implementation deliberately uses a small number of propagation rounds for
speed. Node presolve runs many times in branch-and-bound, so cheap propagation
can be better overall than an expensive full presolve at every node.

### Example: Binary Propagation

Suppose

$$
x_1 + x_2 + x_3 \le 1,
$$

and all variables are binary. If branching fixes

$$
x_1 = 1,
$$

then the row becomes

$$
1 + x_2 + x_3 \le 1.
$$

Therefore

$$
x_2 = 0, \quad x_3 = 0.
$$

This can prune a large part of the search tree.

## Cut Simplification

Cuts are extra linear constraints generated during MIP search. Given a cut

$$
\sum_j a_jx_j \; \{\le,=,\ge\} \; b,
$$

Simplinho computes the minimum and maximum possible activity from the current
bounds. A cut can be:

- redundant,
- infeasible,
- simplified by substituting fixed variables.

For a `LE` cut:

$$
U \le b
$$

means the cut is redundant, while

$$
L > b
$$

means the node is infeasible.

For a fixed variable $x_k = \bar{x}_k$, the cut is updated as

$$
\sum_{j \ne k}a_jx_j \; \{\le,=,\ge\} \; b - a_k\bar{x}_k.
$$

### Example

Consider the cut

$$
2x_1 + 3x_2 + x_3 \le 8.
$$

If node bounds fix

$$
x_2 = 1,
$$

then the simplified cut is

$$
2x_1 + x_3 \le 5.
$$

## MIP Root Presolve

Root presolve runs once near the start of branch-and-bound. Since it runs much
less often than node presolve, it can afford stronger reductions.

The modern root pipeline is organized into fast, medium, and exhaustive passes:

```text
function presolve_mip_root_problem(problem):
    round integer and binary bounds

    for pass = 1 to max_passes:
        changed = false

        changed |= fast passes:
            relax huge bounds
            canonicalize and merge rows
            detect implied integer variables
            run cheap node-style bound propagation

        detect connected components

        changed |= medium passes:
            singleton column substitution
            simple substitution
            simplify integral inequalities
            coefficient strengthening
            strong probing
            clique and parallel-column reductions

        changed |= exhaustive passes:
            dual fixing
            dual-inference bound tightening
            free-variable substitution
            sparsification with equalities
            aggregation of implied-free continuous variables

        if not changed:
            break

    return simplified MIP or infeasible
```

Small models use a legacy root presolve path. For tiny instances, the overhead
of a large presolve pipeline can cost more than it saves.

## Coefficient Strengthening

For integer variables, coefficients can sometimes be reduced without changing
the set of integer feasible solutions.

Consider a knapsack-like inequality

$$
5x_1 + 5x_2 + 2x_3 \le 7,
$$

with binary variables. If $x_1 = 1$, then at most one of the remaining positive
terms can fit. In some cases, large coefficients can be strengthened using the
slack structure of the row.

The exact details are implementation-specific, but the teaching idea is:
integrality turns continuous intervals into discrete sets, and this can make
some coefficients stronger than they look in the LP relaxation.

## Implied Integer Detection

Sometimes a continuous variable is forced to take integer values because of
rows and other integer variables. For example,

$$
x - 2y = 0,
$$

where $y$ is integer. Then

$$
x = 2y,
$$

so $x$ is also integer-valued for every feasible solution. MIP presolve can
mark such variables as implied integer when it can prove the property safely.

This matters because integer variables allow stronger rounding and propagation.

## Aggregation

Aggregation replaces a variable by an expression involving other variables.
For example,

$$
x_1 - x_2 = 0
$$

allows

$$
x_1 = x_2.
$$

If the objective is

$$
4x_1 + 7x_2,
$$

then after substituting $x_1 = x_2$, the objective contribution becomes

$$
11x_2.
$$

Aggregation can reduce variables and coefficients, but it can also make rows
denser. Good presolvers balance reduction strength against fill-in.

## Connected Components

A MIP can sometimes be split into independent pieces. Build a bipartite graph:

- variable nodes,
- constraint nodes,
- an edge when variable $x_j$ appears in row $i$.

If the graph has multiple connected components, those components interact only
through the objective. This information can guide root presolve and search.

### Example

If the model has constraints

$$
x_1 + x_2 \le 1,
$$

and

$$
x_3 + x_4 \le 1,
$$

with no row containing variables from both groups, then

$$
\{x_1,x_2\}
$$

and

$$
\{x_3,x_4\}
$$

are separate components.

## Complete LP Example

Consider the LP

$$
\min \; x_1 + 2x_2 + 0x_3
$$

subject to

$$
x_1 + x_2 \le 5,
$$

$$
2x_1 + 2x_2 \le 10,
$$

$$
x_3 = 4,
$$

and

$$
0 \le x_1 \le 100, \quad 0 \le x_2 \le 100, \quad 4 \le x_3 \le 4.
$$

Presolve can do the following:

1. Detect that $x_3$ is fixed:

$$
x_3 = 4.
$$

2. Remove or zero the column for $x_3$.

3. Detect that the second row is a duplicate of the first:

$$
2x_1 + 2x_2 \le 10
\quad \Longleftrightarrow \quad
x_1 + x_2 \le 5.
$$

4. Tighten bounds from the remaining row:

$$
x_1 \le 5,
$$

and

$$
x_2 \le 5.
$$

The reduced model is essentially

$$
\min \; x_1 + 2x_2
$$

subject to

$$
x_1 + x_2 \le 5,
$$

and

$$
0 \le x_1 \le 5, \quad 0 \le x_2 \le 5.
$$

Postsolve restores

$$
x_3 = 4.
$$

### Classroom Discussion

Ask students which reductions require a postsolve action. Removing the duplicate
row does not need one, because no variable value is lost. Fixing $x_3$ does need
one, because the final solution must contain the original coordinate.

## Complete MIP Example

Consider the binary MIP

$$
\min \; -3x_1 - x_2 + 2x_3
$$

subject to

$$
x_1 + x_2 + x_3 \le 1,
$$

$$
x_1 + y = 1,
$$

with

$$
x_1,x_2,x_3 \in \{0,1\}, \quad 0 \le y \le 1.
$$

Root presolve can reason as follows:

1. The equation $x_1 + y = 1$ implies

$$
y = 1 - x_1.
$$

2. Since $x_1$ is binary, $y$ is also binary-valued, so $y$ may be detected as
implied integer.

3. If branching later fixes $x_1 = 1$, node presolve gets

$$
y = 0.
$$

4. The row

$$
x_1 + x_2 + x_3 \le 1
$$

then implies

$$
x_2 = 0, \quad x_3 = 0.
$$

This one branch decision fixes all variables.

## Practical Design Choices in Simplinho

Simplinho's presolve is intentionally split by cost:

- LP presolve favors safe, reversible, numerically conservative reductions.
- MIP node presolve favors cheap propagation because it runs many times.
- MIP root presolve allows stronger passes because it runs once.
- Structural LP substitutions are optional because they can complicate basis
  management and postsolve.
- Huge-bound relaxation is optional because some users intentionally model with
  large finite bounds.

This design is common in production solvers: presolve is not just about
removing as much as possible. It is about removing enough, cheaply enough, while
preserving numerical reliability.

## Suggested Exercises

1. For the row

$$
4x_1 - x_2 + 3x_3 \le 12,
$$

with

$$
0 \le x_1 \le 10, \quad 2 \le x_2 \le 6, \quad 0 \le x_3 \le 5,
$$

compute $L$ and $U$. Is the row redundant?

2. For

$$
2x_1 - 5x_2 \ge 7,
$$

with

$$
0 \le x_1 \le 10, \quad 0 \le x_2 \le 10,
$$

derive the implied lower or upper bound for each variable.

3. Suppose $x$ is integer and

$$
1.001 \le x \le 4.999.
$$

What are the tightened MIP bounds?

4. Find the fixed variables and redundant rows in

$$
x_1 + x_2 = 2,
$$

$$
2x_1 + 2x_2 = 4,
$$

$$
x_3 = 0,
$$

with

$$
0 \le x_1,x_2 \le 2, \quad x_3 = 0.
$$

5. Give an example of a zero column that proves unboundedness. Then modify one
bound so that presolve can fix the variable instead.

6. For the equality

$$
3x_1 - x_2 = 6,
$$

solve for $x_2$, substitute it into the objective $2x_1 + 4x_2$, and compute
the new objective shift and coefficient.

7. Explain why a duplicate equality row needs no postsolve action, but a
singleton column substitution does.

## Where to Read the Code

The most relevant files are:

- `include/simplex/presolver.h`: dense LP presolve, action stack, and postsolve
  transformations.
- `include/simplex/postsolve.h`: helpers that attach postsolved information to
  LP solutions.
- `include/bnb/mip_presolve.h`: public MIP presolve result types and entry
  points.
- `src/bnb/mip_presolve.cpp`: MIP root presolve, node bound propagation, cut
  simplification, probing, aggregation, and strengthening.

When reading the implementation, keep the main invariant in mind:

$$
\text{solution of reduced model}
\quad \xrightarrow{\text{postsolve}} \quad
\text{solution of original model}.
$$
