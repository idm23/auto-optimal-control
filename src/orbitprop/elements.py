"""Orbital elements: Keplerian, Cartesian and modified equinoctial.

Modified equinoctial elements (Walker, Ireland & Owens 1985) are

    p = a (1 - e^2)                    semi-latus rectum
    f = e cos(omega + Omega)           g = e sin(omega + Omega)
    h = tan(i/2) cos Omega             k = tan(i/2) sin Omega
    L = Omega + omega + nu             true longitude

and are regular for circular and equatorial orbits (only i = 180 deg is singular),
which is why low-thrust transfers are usually solved in them: five of the six move
slowly, and only ``L`` turns once per revolution.

The equinoctial frame: ``f_hat`` and ``g_hat`` span the orbit plane (``f_hat`` points
where ``L = 0``), and ``w_hat = f_hat x g_hat`` is the orbit normal,

    f_hat = (1 + h^2 - k^2, 2hk, -2k) / s^2
    g_hat = (2hk, 1 - h^2 + k^2, 2h) / s^2
    w_hat = (2k, -2h, 1 - h^2 - k^2) / s^2,      s^2 = 1 + h^2 + k^2.

The NumPy functions take and return arrays; :func:`symbolic_equinoctial_to_cartesian`
gives the same map as SymPy expressions.
"""

import numpy as np
import sympy as sp


def _frame(h, k):
    s2 = 1 + h ** 2 + k ** 2
    fh = [(1 + h ** 2 - k ** 2) / s2, 2 * h * k / s2, -2 * k / s2]
    gh = [2 * h * k / s2, (1 - h ** 2 + k ** 2) / s2, 2 * h / s2]
    wh = [2 * k / s2, -2 * h / s2, (1 - h ** 2 - k ** 2) / s2]
    return fh, gh, wh


def equinoctial_frame(h, k):
    """``(f_hat, g_hat, w_hat)`` as SymPy 3-vectors (or numbers, for numbers)."""
    fh, gh, wh = _frame(h, k)
    return sp.Matrix(fh), sp.Matrix(gh), sp.Matrix(wh)


def symbolic_equinoctial_to_cartesian(p, f, g, h, k, L, mu):
    """Position and velocity (SymPy 3-vectors) from modified equinoctial elements."""
    fh, gh, _ = equinoctial_frame(h, k)
    w = 1 + f * sp.cos(L) + g * sp.sin(L)
    r = p / w
    pos = r * (sp.cos(L) * fh + sp.sin(L) * gh)
    vel = sp.sqrt(mu / p) * (-(g + sp.sin(L)) * fh + (f + sp.cos(L)) * gh)
    return pos, vel


def equinoctial_to_cartesian(mee, mu):
    """``(r, v)`` from ``mee = (p, f, g, h, k, L)`` (NumPy)."""
    p, f, g, h, k, L = np.asarray(mee, dtype=float)
    fh, gh, _ = (np.array(v) for v in _frame(h, k))
    w = 1 + f * np.cos(L) + g * np.sin(L)
    r = p / w * (np.cos(L) * fh + np.sin(L) * gh)
    v = np.sqrt(mu / p) * (-(g + np.sin(L)) * fh + (f + np.cos(L)) * gh)
    return r, v


def cartesian_to_equinoctial(r, v, mu):
    """``(p, f, g, h, k, L)`` from position and velocity (NumPy); ``L`` in (-pi, pi]."""
    r, v = np.asarray(r, dtype=float), np.asarray(v, dtype=float)
    hv = np.cross(r, v)
    hn = np.linalg.norm(hv)
    p = hn ** 2 / mu
    w = hv / hn                                    # (sin i sin O, -sin i cos O, cos i)
    if w[2] <= -1 + 1e-14:
        raise ValueError("retrograde equatorial orbit: modified equinoctial elements are singular")
    h, k = -w[1] / (1 + w[2]), w[0] / (1 + w[2])
    fh, gh, _ = (np.array(x) for x in _frame(h, k))
    e = np.cross(v, hv) / mu - r / np.linalg.norm(r)
    return np.array([p, e @ fh, e @ gh, h, k, np.arctan2(r @ gh, r @ fh)])


def keplerian_to_equinoctial(a, e, i, raan, argp, nu):
    """``(p, f, g, h, k, L)`` from Keplerian elements (angles in radians)."""
    return np.array([a * (1 - e ** 2), e * np.cos(argp + raan), e * np.sin(argp + raan),
                     np.tan(i / 2) * np.cos(raan), np.tan(i / 2) * np.sin(raan),
                     raan + argp + nu])


def equinoctial_to_keplerian(mee):
    """``(a, e, i, raan, argp, nu)`` from ``(p, f, g, h, k, L)``; for a circular or
    equatorial orbit the undefined angles are taken as zero."""
    p, f, g, h, k, L = np.asarray(mee, dtype=float)
    e = np.hypot(f, g)
    i = 2 * np.arctan(np.hypot(h, k))
    raan = np.arctan2(k, h) if np.hypot(h, k) > 0 else 0.0
    lon_peri = np.arctan2(g, f) if e > 0 else 0.0
    return np.array([p / (1 - e ** 2), e, i, raan, lon_peri - raan, L - lon_peri])
