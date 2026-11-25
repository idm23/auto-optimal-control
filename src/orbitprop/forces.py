"""Accelerations as SymPy vectors.

A force model is a function of the position (a SymPy 3-vector, whose entries may be
symbols or expressions in other states, such as the equinoctial elements) and its
parameters (numbers or SymPy symbols, so a problem can vary them). Conservative forces
are written as potentials and differentiated, so the acceleration is the exact
gradient of the potential, and its Jacobian is exact too.
"""

import numpy as np
import sympy as sp


def norm(r):
    return sp.sqrt(sum(x ** 2 for x in r))


def _gradient_at(potential, r):
    """The gradient of ``potential(X)`` at the position ``r``: differentiated in plain
    symbols, then evaluated at ``r`` (which may be any expressions)."""
    X = sp.symbols("_X _Y _Z", real=True)
    U = potential(sp.Matrix(X))
    grad = sp.Matrix([sp.diff(U, x) for x in X])
    return grad.xreplace(dict(zip(X, r)))


def point_mass(r, mu):
    """``-mu r / |r|^3``."""
    r = sp.Matrix(r)
    return -mu * r / norm(r) ** 3


def zonal_potential(r, mu, radius, J):
    """The zonal harmonics' part of the gravity potential (the acceleration is its
    gradient): ``-(mu/r) sum_n J_n (R/r)^n P_n(z/r)``, ``J`` as ``{n: J_n}``."""
    rn = norm(r)
    s = r[2] / rn
    return -mu / rn * sum(Jn * (radius / rn) ** n * sp.legendre(n, s) for n, Jn in J.items())


def zonal(r, mu, radius, J):
    """The acceleration of the zonal harmonics ``J`` (``{n: J_n}``) at ``r``."""
    return _gradient_at(lambda X: zonal_potential(X, mu, radius, J), r)


class Gravity:
    """A body's gravity: a point mass ``mu`` and, optionally, zonal harmonics.

    ``mu``, ``radius`` and the ``J`` values may be numbers or SymPy symbols. Use
    :meth:`of` to take them from a :class:`~orbitprop.constants.Body`."""

    def __init__(self, mu, radius=None, J=None):
        self.mu, self.radius, self.J = mu, radius, dict(J or {})
        if self.J and radius is None:
            raise ValueError("zonal harmonics need the body's reference radius")

    @classmethod
    def of(cls, body, degree=2, units=None):
        """``body``'s gravity to zonal ``degree`` (1 for a point mass), in canonical
        ``units`` if given (``mu = 1``)."""
        mu, radius = body.mu, body.radius
        if units is not None:
            mu, radius = 1.0, body.radius / units.length
        return cls(mu, radius, body.zonals(degree) if degree >= 2 else {})

    def potential(self, r):
        """The potential ``V`` (the acceleration is ``grad V``)."""
        V = self.mu / norm(r)
        if self.J:
            V = V + zonal_potential(r, self.mu, self.radius, self.J)
        return V

    def perturbation(self, r):
        """The acceleration beyond the point mass (zero vector without harmonics)."""
        if not self.J:
            return sp.zeros(3, 1)
        return zonal(r, self.mu, self.radius, self.J)

    def acceleration(self, r):
        return point_mass(r, self.mu) + self.perturbation(r)


def rtn_basis(r, v):
    """The radial / transverse / normal unit vectors ``(i_r, i_t, i_n)`` of the orbit
    through ``(r, v)``: ``i_r`` along ``r``, ``i_n`` along ``r x v``, ``i_t = i_n x i_r``."""
    r, v = sp.Matrix(r), sp.Matrix(v)
    ir = r / norm(r)
    hv = r.cross(v)
    i_n = hv / norm(hv)
    return ir, i_n.cross(ir), i_n


def numeric(exprs, symbols):
    """A NumPy function of ``symbols`` (a flat sequence) giving ``exprs`` (a sequence or
    a Matrix) as a float array, compiled with common subexpressions shared."""
    fn = sp.lambdify(list(symbols), list(exprs), modules="numpy", cse=True)
    return lambda *args: np.asarray(fn(*args), dtype=float)
