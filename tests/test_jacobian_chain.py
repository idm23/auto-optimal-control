"""End-to-end check that the generated derivatives compose into an exact shooting Jacobian.

This is a test of the kernels, not a solver: a bare Newton iteration on the
boundary residuals, with the Jacobian assembled from ``residual_jacobian``,
the propagated STM and ``dz_dt1``. Quadratic convergence from a perturbed
guess shows every piece is right. The package itself stays independent of
any particular way of choosing costates.
"""

import numpy as np


def residual_and_jacobian(k, v, p, t0=0.0):
    """Unknowns v = [z0 (NZ), nu (NNU), tf]: R(v) and dR/dv."""
    nz, nnu = k.N_Z, k.N_MULTIPLIERS
    z0, nu, tf = v[:nz], v[nz:nz + nnu], v[-1]
    r = k.propagate(t0, z0, tf, p, stm=True, rtol=1e-13, atol=1e-14)
    R = k.residual(z0, r["z"], nu, tf, p)
    J = k.residual_jacobian(z0, r["z"], nu, tf, p)
    dR = np.hstack([J["z0"] + J["zf"] @ r["stm"], J["nu"],
                    (J["tf"] + J["zf"] @ r["dz_dt1"])[:, None]])
    return R, dR


def newton(k, v, p, iters=12):
    norms = []
    for _ in range(iters):
        R, dR = residual_and_jacobian(k, v, p)
        norms.append(np.linalg.norm(R))
        if norms[-1] < 1e-12:
            break
        v = v - np.linalg.solve(dR, R)
    return v, norms


def test_min_time_double_integrator_converges_quadratically(compiled):
    _, _, k = compiled["min_time"]
    a, umax = 1.0, 2.0
    p = [umax, a, 0.0]
    ts = np.sqrt(a / umax)
    exact_lam = [1.0 / (umax * ts), 1.0 / umax]
    guess = np.array([a, 0.0, 1.15 * exact_lam[0], 0.9 * exact_lam[1], 0.0, 0.0, 1.1 * 2 * ts])
    v, norms = newton(k, guess, p)
    assert norms[-1] < 1e-12
    np.testing.assert_allclose(v[2:4], exact_lam, rtol=1e-10)
    assert abs(v[-1] - 2 * ts) < 1e-10
    # Quadratic convergence: each error is at most ~ the square of the previous.
    tail = [n for n in norms if n > 1e-11]
    for prev, nxt in zip(tail[1:], tail[2:]):
        assert nxt < 10 * prev ** 2 + 1e-12


def test_zermelo_converges(compiled):
    _, _, k = compiled["zermelo"]
    p = [1.0, 1.0, 1.0, 0.5]           # V, k, target (1, 0.5)
    guess = np.array([0.0, 0.0, -0.7, -0.7, -0.7, -0.7, 1.0])
    v, norms = newton(k, guess, p)
    assert norms[-1] < 1e-12
    # Optimality: H = 0 along the solution, and the trajectory hits the target.
    r = k.propagate(0.0, v[:4], v[-1], p, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(r["z"][:2], [1.0, 0.5], atol=1e-12)
    assert abs(k.hamiltonian(0.0, v[:4], p)) < 1e-12
