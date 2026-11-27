"""Equations of motion, Cartesian and modified equinoctial.

Both take a :class:`~orbitprop.forces.Gravity` and an optional extra acceleration (a
thrust ``T delta / m`` times its direction, say): in the inertial frame for
:func:`cartesian`, and in the radial / transverse / normal frame for
:func:`equinoctial`, which is the frame a thrust direction is usually given in.
The mass equation, if any, belongs to the caller.
"""

import sympy as sp

from orbitprop.elements import equinoctial_frame, symbolic_equinoctial_to_cartesian


def cartesian(gravity, r, v, thrust=None):
    """``d(r, v)/dt``: the six rates, as a list of SymPy expressions. ``thrust`` is an
    inertial acceleration (3-vector) added to gravity's."""
    a = gravity.acceleration(sp.Matrix(r))
    if thrust is not None:
        a = a + sp.Matrix(thrust)
    return [*v, *a]


def equinoctial(gravity, mee, thrust=None):
    """``d(p, f, g, h, k, L)/dt``: Gauss's variational equations in modified equinoctial
    elements (Walker, Ireland & Owens 1985; Betts 2010 eq. 6.35), as a list of SymPy
    expressions.

    The perturbing acceleration ``Delta = (D_r, D_t, D_n)`` in the radial / transverse /
    normal frame is gravity's beyond the point mass, projected on that frame, plus
    ``thrust`` (a 3-vector in the same frame)::

        q = 1 + f cos L + g sin L,   s^2 = 1 + h^2 + k^2,   z = h sin L - k cos L
        p' = 2 p / q sqrt(p/mu) D_t
        f' = sqrt(p/mu) (  D_r sin L + ((q + 1) cos L + f) D_t / q - z g D_n / q)
        g' = sqrt(p/mu) (- D_r cos L + ((q + 1) sin L + g) D_t / q + z f D_n / q)
        h' = sqrt(p/mu) s^2 cos L D_n / (2 q)
        k' = sqrt(p/mu) s^2 sin L D_n / (2 q)
        L' = sqrt(mu p) (q / p)^2 + sqrt(p/mu) z D_n / q
    """
    p, f, g, h, k, L = mee
    mu = gravity.mu
    D = sp.zeros(3, 1)
    if gravity.J:
        pos, _ = symbolic_equinoctial_to_cartesian(p, f, g, h, k, L, mu)
        fh, gh, wh = equinoctial_frame(h, k)
        ir = sp.cos(L) * fh + sp.sin(L) * gh
        it = -sp.sin(L) * fh + sp.cos(L) * gh
        a = gravity.perturbation(pos)
        D = D + sp.Matrix([ir.dot(a), it.dot(a), wh.dot(a)])
    if thrust is not None:
        D = D + sp.Matrix(thrust)
    Dr, Dt, Dn = D
    q = 1 + f * sp.cos(L) + g * sp.sin(L)
    s2 = 1 + h ** 2 + k ** 2
    z = h * sp.sin(L) - k * sp.cos(L)
    sq = sp.sqrt(p / mu)
    return [
        2 * p / q * sq * Dt,
        sq * (Dr * sp.sin(L) + ((q + 1) * sp.cos(L) + f) * Dt / q - z * g * Dn / q),
        sq * (-Dr * sp.cos(L) + ((q + 1) * sp.sin(L) + g) * Dt / q + z * f * Dn / q),
        sq * s2 * sp.cos(L) * Dn / (2 * q),
        sq * s2 * sp.sin(L) * Dn / (2 * q),
        sp.sqrt(mu * p) * (q / p) ** 2 + sq * z * Dn / q,
    ]
