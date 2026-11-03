"""NumPy callables for a :class:`~auto_optimal_control.pontryagin.PontryaginSystem`.

This is a reference implementation built with :func:`sympy.lambdify`: a quick
way to evaluate and integrate the necessary conditions with
fast-numerical-integration's ``solve_ivp``, and the oracle the generated C
kernels will be tested against. It runs in Python, so the integrator holds the
GIL.

Every function takes the augmented state ``z = [x, lambda]`` and the parameter
vector ``p``, in the orders of ``system.z`` and ``system.params``.
"""

import numpy as np
import sympy as sp


def _vector(fn):
    def wrapped(*args):
        return np.array(fn(*args), dtype=np.float64).reshape(-1)
    return wrapped


class NumericSystem:
    """Lambdified necessary conditions."""

    def __init__(self, system):
        self.system = system
        t, z, p = system.time, list(system.z), list(system.params)
        bc = system.boundary
        args = (t, z, p)

        def lam(exprs):
            return _vector(sp.lambdify(args, list(exprs), modules="numpy"))

        self._rhs = lam(system.dynamics)
        self._control = lam(system.control_law[u] for u in system.controls)
        self._switching = lam(sw.expr for sw in system.switching_functions)
        self._hamiltonian = sp.lambdify(args, system.hamiltonian, modules="numpy")
        self._cost_rate = sp.lambdify(args, system.running_cost, modules="numpy")

        tf_free = system.problem.free_final_time
        tf = bc.tf if tf_free else sp.Dummy("t_f")
        bc_args = (list(bc.x0 + bc.lam0), list(bc.xf + bc.lamf), list(bc.nu), tf, p)
        self._residual = _vector(sp.lambdify(bc_args, list(bc.residuals), modules="numpy"))
        self._t0 = sp.lambdify([p], bc.t0, modules="numpy")
        self._tf = None if tf_free else sp.lambdify([p], bc.tf, modules="numpy")

    @property
    def n(self):
        """Number of states (``z`` has ``2 n`` entries)."""
        return len(self.system.states)

    def rhs(self, t, z, p=()):
        """``z' = F(t, z, p)``."""
        return self._rhs(t, z, p)

    def control(self, t, z, p=()):
        """Optimal control ``u*(t, z, p)``, in the order of ``system.controls``."""
        return self._control(t, z, p)

    def switching(self, t, z, p=()):
        """Switching-function values, in the order of ``system.switching_functions``."""
        return self._switching(t, z, p)

    def hamiltonian(self, t, z, p=()):
        return float(self._hamiltonian(t, z, p))

    def cost_rate(self, t, z, p=()):
        """Running cost ``L`` at ``u*``."""
        return float(self._cost_rate(t, z, p))

    def residual(self, z0, zf, nu=(), tf=np.nan, p=()):
        """Boundary-condition residuals; ``tf`` is used only when the final time is free."""
        return self._residual(z0, zf, nu, tf, p)

    def initial_time(self, p=()):
        return float(self._t0(p))

    def final_time(self, p=()):
        """The fixed final time, or ``None`` when it is free."""
        return None if self._tf is None else float(self._tf(p))

    def ode(self, p=()):
        """``fun(t, z)`` for ``solve_ivp``, with the parameters bound."""
        p = np.asarray(p, dtype=np.float64)
        rhs = self._rhs
        return lambda t, z: rhs(t, z, p)

    def switching_events(self, p=(), terminal=False, direction=0.0):
        """One scipy-style event callable per switching function."""
        p = np.asarray(p, dtype=np.float64)
        events = []
        for k in range(len(self.system.switching_functions)):
            def event(t, z, k=k):
                return float(self._switching(t, z, p)[k])
            event.terminal = terminal
            event.direction = direction
            events.append(event)
        return events
