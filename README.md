# auto-optimal-control

Express an optimal control problem symbolically and get every function the
indirect (costate) method needs to solve it as a two-point boundary value
problem: the Hamiltonian, the optimal control law, switching functions for
bang-bang controls, the state–costate dynamics and the boundary-condition
residuals.

Integration runs on
[fast-numerical-integration](https://github.com/idm23/fast-numerical-integration)
(FNI), which this package depends on from PyPI.

> **Status: pre-alpha.** The symbolic problem definition and the Pontryagin
> derivation are in place. Printing and compiling C kernels, then shooting,
> multiple shooting and continuation, come next (see [Roadmap](#roadmap)).

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

To integrate an extremal now, before the C kernels exist, use the NumPy
reference implementation with FNI:

```python
from auto_optimal_control.numeric import NumericSystem
from fast_numerical_integration.integrators import solve_ivp

num = NumericSystem(pmp)
sol = solve_ivp(num.ode(), (0, 2), [1, 0, 1, 1], method="DOP853",
                events=num.switching_events(), rtol=1e-12, atol=1e-12)
sol.t_events[0]                  # [1.0]: the switch from u = -1 to u = +1
num.residual(sol.y[:, 0], sol.y[:, -1], sol.y[2:, -1], 2.0)   # ~0
```

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
| `boundary` | the shooting function: `residuals`, `unknowns` and the boundary symbols |

The boundary residuals are the fixed initial states (or `lam_i(t_0) = 0` when
free), the terminal constraints `psi`, transversality
`lam(t_f) = d(phi + nu . psi)/dx`, and `H(t_f) + d(phi + nu . psi)/dt = 0`
when the final time is free. There are as many residuals as unknowns.

`auto_optimal_control.examples` has the problems the tests use: Zermelo's
navigation problem, minimum-time and minimum-energy double integrators
(unbounded and saturated), a terminal-cost LQ problem and a fuel-optimal
planar low-thrust transfer.

## Tests

`uv run pytest` checks:

- **`tests/test_problem.py`:** problem validation.
- **`tests/test_derivation.py`:** each example's Hamiltonian, control law,
  switching function, costate equations and boundary conditions against the
  textbook forms. On every example at random points, it also checks that `u*`
  minimizes `H` over the admissible controls and that the dynamics are
  Hamilton's equations of `H*`.
- **`tests/test_extremals.py`:** extremals integrated with FNI from known
  optimal costates. They must reproduce the analytic solutions (Zermelo's
  heading law, switch and arrival times, minimum-energy cost), satisfy every
  derived boundary condition, and conserve `H` across bang-bang switches.

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

1. ~~Symbolic problem definition and Pontryagin derivation~~ (this release)
2. Print C kernels (dynamics, control law, `H`, Jacobians, boundary residuals,
   switching functions) with common-subexpression elimination, compile them,
   and wrap them as FNI `OdeSystem` / `EventFunction` objects using its
   GIL-free C API.
3. State transition matrix: variational equations, with jump corrections at
   switches.
4. Shooting, multiple shooting and continuation (for example the
   energy-to-fuel homotopy), driven by low-thrust orbit transfers.

## License

MIT; see [LICENSE](LICENSE).
