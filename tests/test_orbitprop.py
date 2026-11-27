"""orbitprop: element conversions, force models and equations of motion.

The equinoctial equations are checked against the Cartesian ones by integrating the
same orbit both ways under J2; the Cartesian ones against energy conservation and
J2's secular drift of the node.
"""

import numpy as np
import pytest
import sympy as sp

from orbitprop import (EARTH, CanonicalUnits, Gravity, cartesian, cartesian_to_equinoctial,
                       equinoctial, equinoctial_to_cartesian, equinoctial_to_keplerian,
                       keplerian_to_equinoctial, numeric, symbolic_equinoctial_to_cartesian)
from orbitprop.forces import point_mass

scipy_integrate = pytest.importorskip("scipy.integrate")

U = CanonicalUnits.of(EARTH)
ORBITS = [   # a (Earth radii), e, i, raan, argp, nu
    (1.1, 0.0, 0.0, 0.0, 0.0, 0.3),
    (1.2, 0.01, np.radians(28.5), 0.4, 1.0, 2.0),
    (4.0, 0.72, np.radians(7.0), -1.2, 3.0, -0.5),          # a GTO
    (6.6, 1e-4, np.radians(0.05), 2.0, -1.0, 1.0),          # near GEO
    (1.15, 0.1, np.radians(98.0), 0.1, 0.2, 0.3),           # retrograde, sun-synchronous-like
]


@pytest.mark.parametrize("orbit", ORBITS)
def test_elements_round_trip(orbit):
    mee = keplerian_to_equinoctial(*orbit)
    r, v = equinoctial_to_cartesian(mee, 1.0)
    back = cartesian_to_equinoctial(r, v, 1.0)
    assert back[:5] == pytest.approx(mee[:5], abs=1e-12)
    assert np.cos(back[5] - mee[5]) == pytest.approx(1, abs=1e-12)
    a, e, i, *_ = equinoctial_to_keplerian(mee)
    assert (a, e, i) == pytest.approx(orbit[:3], abs=1e-12)
    # the vis-viva energy and the angular momentum
    assert v @ v / 2 - 1 / np.linalg.norm(r) == pytest.approx(-1 / (2 * orbit[0]), rel=1e-12)
    assert np.linalg.norm(np.cross(r, v)) ** 2 == pytest.approx(mee[0], rel=1e-12)


def test_symbolic_and_numeric_conversions_agree():
    els = sp.symbols("p f g h k L", real=True)
    pos, vel = symbolic_equinoctial_to_cartesian(*els, 1)
    fn = numeric([*pos, *vel], els)
    for orbit in ORBITS:
        mee = keplerian_to_equinoctial(*orbit)
        r, v = equinoctial_to_cartesian(mee, 1.0)
        assert fn(*mee) == pytest.approx(np.concatenate([r, v]), abs=1e-13)


def test_earth_constants():
    assert EARTH.J[2] == pytest.approx(1.0826262e-3, rel=1e-7)        # EGM2008 (EGM96: 1.0826267e-3)
    assert EARTH.J[3] == pytest.approx(-2.5324e-6, rel=1e-4)
    assert EARTH.J[4] == pytest.approx(-1.6199e-6, rel=1e-4)
    assert U.time == pytest.approx(806.81, rel=1e-4)            # s
    assert U.speed == pytest.approx(7.9054, rel=1e-4)           # km/s


def test_j2_acceleration_is_the_gradient_of_its_potential_in_closed_form():
    x, y, z = r = sp.symbols("x y z", real=True)
    mu, R, J2 = sp.symbols("mu R J_2", positive=True)
    a = Gravity(mu, R, {2: J2}).perturbation(sp.Matrix(r))
    rn = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
    c = -sp.Rational(3, 2) * J2 * mu * R ** 2 / rn ** 5
    closed = sp.Matrix([c * x * (1 - 5 * z ** 2 / rn ** 2), c * y * (1 - 5 * z ** 2 / rn ** 2),
                        c * z * (3 - 5 * z ** 2 / rn ** 2)])
    pt = {x: 0.7, y: -0.4, z: 0.9, mu: 1.3, R: 0.8, J2: 1e-3}
    assert [float(e.subs(pt)) for e in a] == pytest.approx([float(e.subs(pt)) for e in closed], rel=1e-12)


def _cartesian_rhs(gravity):
    r = sp.symbols("x y z", real=True)
    v = sp.symbols("vx vy vz", real=True)
    fn = numeric(cartesian(gravity, r, v), [*r, *v])
    return lambda t, s: fn(*s)


def _equinoctial_rhs(gravity, thrust=None, extra=()):
    els = sp.symbols("p f g h k L", real=True)
    fn = numeric(equinoctial(gravity, els, thrust), [*els, *extra])
    return lambda t, s: fn(*s)


def _integrate(rhs, s0, tf, n=400):
    sol = scipy_integrate.solve_ivp(rhs, (0, tf), s0, method="DOP853", rtol=1e-12, atol=1e-13,
                                    t_eval=np.linspace(0, tf, n))
    assert sol.success
    return sol.y.T


@pytest.mark.parametrize("orbit", ORBITS[1:4])
def test_equinoctial_and_cartesian_motion_agree_under_zonal_gravity(orbit):
    """Gauss's equations with the zonal perturbation projected on the RTN frame give the
    same motion as Newton's in Cartesian coordinates (J2 to J4, five orbits)."""
    gravity = Gravity.of(EARTH, degree=4, units=U)
    mee0 = keplerian_to_equinoctial(*orbit)
    r0, v0 = equinoctial_to_cartesian(mee0, 1.0)
    tf = 5 * 2 * np.pi * orbit[0] ** 1.5
    X = _integrate(_cartesian_rhs(gravity), np.concatenate([r0, v0]), tf)
    M = _integrate(_equinoctial_rhs(gravity), mee0, tf)
    for x, m in zip(X, M):
        r, v = equinoctial_to_cartesian(m, 1.0)
        assert np.concatenate([r, v]) == pytest.approx(x, abs=2e-9 * orbit[0])


def test_two_body_equinoctial_elements_are_constant_but_the_longitude():
    mee0 = keplerian_to_equinoctial(*ORBITS[2])
    M = _integrate(_equinoctial_rhs(Gravity(1.0)), mee0, 2 * np.pi * ORBITS[2][0] ** 1.5)
    assert np.abs(M[:, :5] - mee0[:5]).max() < 1e-13
    assert M[-1, 5] - mee0[5] == pytest.approx(2 * np.pi, rel=1e-9)


def test_zonal_gravity_conserves_energy():
    gravity = Gravity.of(EARTH, degree=6, units=U)
    r = sp.symbols("x y z", real=True)
    V = numeric([gravity.potential(sp.Matrix(r))], r)
    mee0 = keplerian_to_equinoctial(*ORBITS[1])
    r0, v0 = equinoctial_to_cartesian(mee0, 1.0)
    X = _integrate(_cartesian_rhs(gravity), np.concatenate([r0, v0]), 30.0)
    E = [s[3:] @ s[3:] / 2 - V(*s[:3])[0] for s in X]
    assert np.ptp(E) < 1e-11


def test_j2_drifts_the_node_at_its_secular_rate():
    """Omega' = -3/2 n J2 (R/p)^2 cos i on average, for the ISS's orbit (420 km, 51.6 deg)
    over a day: about -5 deg/day."""
    gravity = Gravity.of(EARTH, degree=2, units=U)
    a, e, i = 1 + 420 / EARTH.radius, 0.001, np.radians(51.6)
    mee0 = keplerian_to_equinoctial(a, e, i, 0.3, 0.0, 0.0)
    tf = 86400.0 / U.time
    M = _integrate(_equinoctial_rhs(gravity), mee0, tf, n=2000)
    raan = np.unwrap(np.arctan2(M[:, 4], M[:, 3]))
    rate = np.polyfit(np.linspace(0, tf, len(M)), raan, 1)[0]
    p = a * (1 - e ** 2)
    expected = -1.5 * a ** -1.5 * EARTH.J[2] * (1 / p) ** 2 * np.cos(i)
    assert rate == pytest.approx(expected, rel=5e-3)
    assert np.degrees(rate) * 86400 / U.time == pytest.approx(-5.0, abs=0.3)   # deg/day


def test_thrust_in_rtn_matches_thrust_in_the_inertial_frame():
    """A constant tangential thrust in Gauss's equations is the same push as its inertial
    vector added to Newton's (J2 included), and it raises the orbit."""
    gravity = Gravity.of(EARTH, degree=2, units=U)
    T = 1e-3
    rhs_m = _equinoctial_rhs(gravity, thrust=[0, T, 0])
    r, v = sp.symbols("x y z", real=True), sp.symbols("vx vy vz", real=True)
    hv = sp.Matrix(r).cross(sp.Matrix(v))
    it = hv.cross(sp.Matrix(r)) / (sp.sqrt(hv.dot(hv)) * sp.sqrt(sum(x ** 2 for x in r)))
    fn = numeric(cartesian(gravity, r, v, thrust=T * it), [*r, *v])
    mee0 = keplerian_to_equinoctial(1.2, 0.05, np.radians(30), 0.5, 0.5, 0.0)
    r0, v0 = equinoctial_to_cartesian(mee0, 1.0)
    tf = 3 * 2 * np.pi * 1.2 ** 1.5
    X = _integrate(lambda t, s: fn(*s), np.concatenate([r0, v0]), tf)
    M = _integrate(rhs_m, mee0, tf)
    rr, vv = equinoctial_to_cartesian(M[-1], 1.0)
    assert np.concatenate([rr, vv]) == pytest.approx(X[-1], abs=1e-9)
    assert M[-1, 0] > mee0[0] + 0.01
