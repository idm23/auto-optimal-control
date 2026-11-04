# auto-optimal-control

Write down an optimal control problem in SymPy and get every function the
indirect (costate) method needs to solve it: the Hamiltonian, the optimal
control law, switching functions for bang-bang controls, the state–costate
dynamics, the boundary-condition residuals, and the state transition matrix
with its jumps at switches. All of it is generated as C, compiled, and
integrated by
[fast-numerical-integration](https://github.com/idm23/fast-numerical-integration).

```bash
pip install "auto-optimal-control[shooting]"
```

The kernels are compiled when you first build them, so you need a C compiler
(`gcc` or `clang`).

## Where to start

- **{doc}`notebooks/index`**: a short lecture course on indirect optimal
  control. It follows one cart-pole through increasingly constrained
  problems: the necessary conditions, shooting, control limits, minimum time
  and bang-bang control, and state constraints.
- **{doc}`api`**: the package's modules, classes and functions.
- **The README on [GitHub](https://github.com/idm23/auto-optimal-control)**:
  the benchmarks against the literature, the animations, and the current
  limitations.

```{image} https://raw.githubusercontent.com/idm23/auto-optimal-control/main/docs/visuals/cart_pole_polish.png
:alt: Cart-pole swing-up, shooting iterations converging to the extremal
:width: 100%
```

```{toctree}
:hidden:
:maxdepth: 2

notebooks/index
api
```
