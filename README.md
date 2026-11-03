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
costates can all be built on top of the same calls.

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

`uv run pytest` (about 30 s; it compiles every example from scratch) checks:

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
- **`test_jacobian_chain.py`:** a bare Newton iteration built from the
  kernels converges quadratically, which shows the derivatives compose into
  an exact Jacobian. It lives only in the tests; the package has no solver.

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
4. Problems that matter: low-thrust transfers in modified equinoctial
   elements, the energy-to-fuel homotopy.
5. Ways to choose initial costates, as separate optional layers on top of the
   forward map: for example learned samplers of the costate space, shooting
   and continuation.

## License

MIT; see [LICENSE](LICENSE).
