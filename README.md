# auto-optimal-control

Express an optimal control problem symbolically and get every function the
indirect (costate) method needs to solve it as a two-point boundary value
problem: the Hamiltonian, the optimal control law, switching functions for
bang-bang controls, the state–costate dynamics and the boundary-condition
residuals.

Integration runs on
[fast-numerical-integration](https://github.com/idm23/fast-numerical-integration)
(FNI), which this package depends on from PyPI.

> **Status: pre-alpha.** Symbolic problem definition, Pontryagin derivation,
> generated and compiled C kernels, and arc-by-arc propagation of extremals
> with the state transition matrix are in place.

The core operation is the **forward map** from an initial costate to an
extremal: pick `lambda(t_0)` and get the optimal trajectory, its controls,
its cost, its switch times, its boundary residuals and, if you want them,
exact sensitivities. The package doesn't decide how `lambda(t_0)` is chosen.
Shooting, multiple shooting, continuation, or a learned sampler that proposes
costates can all be built on top of the same calls. One such layer ships with
the package, optionally: `auto_optimal_control.shooting`, a single-shooting
system with exact Jacobians and a solver.

## Installation

```bash
uv add auto-optimal-control        # once released; for now, from a clone:
git clone https://github.com/idm23/auto-optimal-control
cd auto-optimal-control
uv sync
uv run pytest
```

## Example: minimum-time double integrator

```python
import sympy as sp
from auto_optimal_control.problem import Box, OptimalControlProblem
from auto_optimal_control.pontryagin import derive

t, x, v, u = sp.symbols("t x v u", real=True)
problem = OptimalControlProblem(
    time=t, states=[x, v], controls=[u],
    dynamics=[v, u], running_cost=1,                # minimum time
    initial_state=[1, 0], final_state={x: 0, v: 0},
    control_sets=[Box(u, -1, 1)])                   # |u| <= 1

pmp = derive(problem)
pmp.hamiltonian                  # Piecewise: 1 + lam_x v - |lam_v|
pmp.control_law[u]               # Piecewise((1, lam_v < 0), (-1, True))
pmp.switching_functions[0].expr  # lam_v
pmp.dynamics                     # (v, u*, 0, -lam_x) for z = [x, v, lam_x, lam_v]
pmp.boundary.residuals           # x(t_0) - 1, v(t_0), x(t_f), v(t_f), transversality, H(t_f)
pmp.boundary.unknowns            # x(t_0), v(t_0), lam_x(t_0), lam_v(t_0), nu_0, nu_1, t_f
```

Compile it and evaluate extremals from any initial costate:

```python
from auto_optimal_control.build import build_kernels

k = build_kernels(pmp)           # generate C, compile (cached), load
r = k.propagate(0.0, k.z0([1, 1]), 2.0, cost=True, stm=True)
r["switch_times"]                # [1.0]: u switches from -1 to +1
r["z"]                           # [x, v, lam_x, lam_v] at t = 2: x = v = 0
r["cost"]                        # 2.0 (minimum time)
r["stm"]                         # dz(2)/dz(0), including the jump at the switch
k.residual(k.z0([1, 1]), r["z"], r["z"][2:], 2.0)   # ~0 (nu = lam(t_f))
```

## Using the compiled kernels

`build_kernels(pmp)` prints the necessary conditions as plain C (see
`codegen.py` for the function list), wraps them for
[fast-numerical-integration](https://github.com/idm23/fast-numerical-integration)'s
GIL-free C API, compiles both and returns a `Kernels` object. Builds are
cached by a hash of the generated code and the toolchain. The first build
takes about 10 s and later ones load instantly, across processes too. The
cache is in `~/.cache/auto-optimal-control`, or `$AUTO_OPTIMAL_CONTROL_CACHE`
if set. Building needs a C compiler.

**The forward map (from an initial costate to an extremal):**

| Call | Returns |
|---|---|
| `k.propagate(t0, z0, t1, params, cost=, stm=, param_sensitivity=)` | final `z`, switch times, cost `J = int L`, and optionally `Phi = dz(t1)/dz(t0)`, `dz(t1)/dp` and `dz(t1)/dt1` |
| `k.propagate_batch(t0, Z0, t1, params, ..., n_threads=)` | the same for many rows at once, in parallel threads without the GIL; each row is bit-identical to `propagate` |
| `k.trajectory(t0, z0, t1, params, n_points=)` | sampled `t`, `z`, controls `u` and `H`, for plots and diagnostics |
| `k.z0(lam0, params)` | `[x0, lam0]` from the problem's fixed initial state |

Propagation runs **arc by arc**: it stops at each bang-bang switch and
restarts with the control fixed, so no integration step straddles the jump
in the control. The STM gets the saltation-matrix jump at every switch.

**Pointwise kernels:** `rhs`, `rhs_arc`, `jacobian` (`dF/dz`), `dfdp`,
`control`, `hamiltonian`, `cost_rate`, `switching`, `switching_gradient`,
`residual`, `residual_jacobian` (with respect to `z0`, `zf`, `nu`, `tf`, `p`)
and `stm_jump`.

**FNI objects:** `k.Dynamics(params)` and `k.ArcDynamics(params, arc, stm=)`
are native `OdeSystem`s, and `k.switching_events(params)` gives native
`EventFunction`s. They work with FNI's `solve_ivp`, `integrate` and
`integrate_batch`, and their `params` can be changed between solves without
recompiling.

**Plain C:** `k.write_c(directory)` writes the header and source on their
own; they depend only on `<math.h>`.

**Speed** (planar low-thrust example, 10 states, 3 switches, DOP853 at
`rtol=1e-10`):

| | time per extremal |
|---|---|
| NumPy reference (`NumericSystem` + FNI `solve_ivp`) | 67 ms |
| `k.propagate` | 0.05 ms |
| `k.propagate(..., stm=True)` (110 states) | 0.28 ms |
| `k.propagate_batch`, 1000 rows, 1 thread / 16 threads | 36 µs / 11 µs |

`auto_optimal_control.numeric.NumericSystem` is still there as a
pure-Python reference (lambdified with SymPy); the kernel tests compare
against it.

## Single shooting (optional)

`auto_optimal_control.shooting` turns the forward map into a square system
`R(v) = 0`. Nothing else in the package imports it. The unknowns `v` are
the free initial states, `lam0`, the terminal multipliers `nu`, and `t_f`
when it is free. The Jacobian is exact: the generated boundary Jacobians
chained with the STM, with the jumps at switches included.

```python
from auto_optimal_control.shooting import SingleShooting, continuation

shoot = SingleShooting(k, params)              # x0 defaults to the problem's
v = shoot.guess_multipliers(lam0, tf=2.0)      # least-squares nu for this lam0
R, dR = shoot.residual_jacobian(v)             # NumPy only: drive it with anything
sol = shoot.solve(v)                           # MINPACK hybr / lm, needs SciPy
sol.success, sol.lam0, sol.tf, sol.z0
```

- **`multistart(sample, n_tries)`** solves from random guesses until one
  converges.
- **`continuation(solve_at, start, values, v0)`** follows a homotopy
  parameter. It halves the step whenever a solve fails.
- **`trajectory(v)`** samples the extremal at `v`.

`solve` needs SciPy: `pip install auto-optimal-control[shooting]`. The
residual and Jacobian need only NumPy, so a sampler or any other solver can
use them directly.

## Defining a problem

`OptimalControlProblem` is in Bolza form:

```
minimize    phi(t_f, x(t_f)) + integral L(t, x, u) dt
subject to  x' = f(t, x, u),   x_i(t_0) fixed (or free),   psi(t_f, x(t_f)) = 0,   u in U
```

- **Parameters** (`params=[...]`) stay symbolic, so one derivation (and later
  one compiled kernel) serves a whole continuation sweep. Their order is the
  order of FNI's runtime parameter vector.
- **Final time** is free by default; pass `final_time=` to fix it.
- **Control sets:** controls are unconstrained unless placed in a set.
  - `Box(u, lower, upper)`: a Hamiltonian linear in `u` gives **bang-bang**
    control with a switching function; a convex quadratic one gives a
    **saturated** (clipped) control.
  - `UnitVector([u1, u2, ...])`: a direction such as a thrust direction or a
    heading. The optimum is `-b/||b||` (the primer vector for thrust).
- Declare positive quantities (masses, thrust, speeds) with `positive=True`.
  The derivation relies on it to simplify norms and to prove convexity.

## What `derive` produces

`derive(problem)` returns a `PontryaginSystem`:

| Attribute | Contents |
|---|---|
| `costates` | `lam_<state>`, one per state |
| `hamiltonian_full` | `H = L + lam . f` with the controls symbolic |
| `control_law` | `u*(t, x, lam, p)` minimizing `H` |
| `switching_functions` | `SwitchingFunction(control, expr, lower, upper)` per bang-bang control: `u = upper` where `expr < 0` |
| `hamiltonian` | `H` at `u*`; constant along extremals when `is_autonomous` |
| `dynamics` | `z' = F(t, z, p)` for `z = [x, lam]`: `x' = dH/dlam`, `lam' = -dH/dx` |
| `running_cost` | `L` at `u*`, to accumulate the cost |
| `boundary` | the boundary conditions: `residuals`, `unknowns` and the boundary symbols |

The boundary residuals are the fixed initial states (or `lam_i(t_0) = 0` when
free), the terminal constraints `psi`, transversality
`lam(t_f) = d(phi + nu . psi)/dx`, and `H(t_f) + d(phi + nu . psi)/dt = 0`
when the final time is free. There are as many residuals as unknowns.

`auto_optimal_control.examples` has the problems the tests use: Zermelo's
navigation problem, minimum-time and minimum-energy double integrators
(unbounded and saturated), a terminal-cost LQ problem and a fuel-optimal
planar low-thrust transfer.

## Tests

`uv run pytest` (about 30 s; it compiles every example from scratch) checks the following; `uv run pytest --runslow` (about 3 min) adds the benchmarks:

- **`test_problem.py`:** problem validation.
- **`test_derivation.py`:** each example's Hamiltonian, control law,
  switching function, costate equations and boundary conditions against the
  textbook forms. On every example at random points, `u*` minimizes `H` over
  the admissible controls and the dynamics are Hamilton's equations of `H*`.
- **`test_extremals.py`:** extremals of the NumPy reference, integrated with
  FNI from known optimal costates. They reproduce the analytic solutions and
  satisfy every derived boundary condition.
- **`test_kernels.py`:**
  - Every C kernel agrees with the NumPy reference at random points.
  - Every derivative kernel agrees with finite differences.
  - The plain C compiles under `-Wall -Wextra -Werror -pedantic`.
  - The cache is reused.
  - Native systems release the GIL.
- **`test_propagation.py`:**
  - Switch times, costs and endpoints match the analytic answers.
  - The STM and `dz/dp` match finite differences of the forward map across
    switches.
  - Leaving out the jump gives the wrong STM.
  - Batches equal single propagations, and threads agree with serial runs.
- **`test_benchmarks.py`** (slow; `--runslow`): every literature benchmark
  above.
- **`test_jacobian_chain.py`:** a bare Newton iteration built from the
  kernels converges quadratically, which shows the derivatives compose into
  an exact Jacobian, using the raw kernels rather than the shooting module.
- **`test_shooting.py`:** `SingleShooting`'s unknown layout, its Jacobian
  against finite differences (including a free initial state), solves
  against closed forms, multistart and continuation.

## Benchmarks against the literature

`benchmarks/` solves five canonical problems from scratch and compares each
indirect solution with a published value. It starts from random costate
guesses, uses continuation where needed, and for the cart-pole lifts
costates from a direct solution. Run them with
`uv run python benchmarks/run.py`; `uv run pytest --runslow`, which CI runs,
checks them too. They are solved with `auto_optimal_control.shooting`.

| Benchmark | Quantity | This package | Reference | Source |
|---|---|---|---|---|
| Brachistochrone (3 cases) | `t_f` | matches to 1e-13 | closed-form cycloid | — |
| Maximum-radius orbit transfer | `r(t_f)` | 1.5252777 | 1.5252776 | Bryson & Ho (1975) pp. 66–69; value from YAPSS/IPOPT |
| Earth–Mars fuel-optimal rendezvous | `m(t_f)` | 603.9402 kg | 603.935 kg | arXiv:2409.01490, Tables 1–2 |
| COPS robot arm, minimum time | `t_f` | 9.1409117 | 9.1409 | COPS #8 (Dolan, Moré & Munson); GAMS NOA, PROPT |
| Cart-pole swing-up (published extremal) | `∫u²` | 58.807667 | 58.80819 | Kelly (2017) SIAM Review; value from the ASSET reproduction |
| Cart-pole swing-up (random search) | `∫u²` | **57.944780** | — | this work; checked by an independent simulation |

Notes:
- **The reference values come from direct (collocation) methods or smoothed
  controls,** so agreement to 1e-6–1e-5 is what to expect. The indirect
  solutions here locate switches exactly. On Earth–Mars, the exact bang-bang
  solution keeps 0.005 kg more mass than the smoothed one (ρ = 1e-5), which is
  the expected sign. It has four switches, at 46.6, 68.0, 142.7 and 290.3 days.
- **Robot arm:** pure bang-bang with five switches, found by continuation
  from `L = 1 + ε Σu²` (saturated controls) to ε = 1e-4, then solved exactly.
  The `u_θ` switch falls at `t_f/2`, by the problem's symmetry.
- **Cart-pole:** Kelly's direct method from his straight-line guess converges
  to an extremal where the cart overshoots to `x = 1.18`, with `J = 58.808`.
  Polishing it indirectly reproduces the published value. Random costate
  search finds a different extremal that never passes `x = 1`, with
  `J = 57.945`, 1.5% cheaper. Driving SciPy's integrator on Kelly's equations
  with that control history confirms it reaches the target with that cost.
  Indirect search over costate space found a better local optimum than the
  published one.

## Limitations

- **Singular arcs** are not handled: where a switching function stays at
  zero over an interval, the control takes its lower bound.
- **Saturation corners are not located.** The corners of a clipped
  (quadratic, bounded) control are kinks in the dynamics, not jumps, and the
  integrator steps across them with errors around 1e-9 at tight tolerances.
  Bang-bang switches are located exactly.
- **Tangential switches** (`dS/dt = 0` at the switch) have no jump and stop
  propagation with status `-3`.

## Developing against a local fast-numerical-integration

CI and normal installs use FNI from PyPI, pinned in `uv.lock`. To try changes
to a local FNI checkout (here a sibling directory):

- **One-off:** run anything against the local checkout without touching the
  lockfile. uv builds the local copy in an overlay environment.

  ```bash
  uv run --with-editable ../fast-numerical-integration pytest
  ```

- **Persistent:** add this to `pyproject.toml`, then `uv sync`:

  ```toml
  [tool.uv.sources]
  fast-numerical-integration = { path = "../fast-numerical-integration", editable = true }
  ```

  This rewrites `uv.lock` to point at the path, so don't commit either
  change. Release the FNI change to PyPI, bump the requirement here and run
  `uv lock --upgrade-package fast-numerical-integration` instead.

## Roadmap

1. ~~Symbolic problem definition and Pontryagin derivation~~
2. ~~Generated C kernels, compiled and wrapped for FNI~~
3. ~~State transition matrix, with jumps at switches; parameter sensitivities~~
4. ~~Literature benchmarks: orbit raising, Earth–Mars, COPS robot arm,
   cart-pole, brachistochrone~~. Next: modified equinoctial elements and
   many-revolution transfers (Earth–Dionysus, GTO–GEO), and minimum-time
   attitude slews.
5. Ways to choose initial costates, as separate optional layers on top of the
   forward map: ~~single shooting and continuation~~; next, gradient-free
   search (PSO, differential evolution) and learned samplers of the costate
   space.

## License

MIT; see [LICENSE](LICENSE).
