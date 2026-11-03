"""Generated C kernels against the NumPy reference and against finite differences."""

import shutil
import subprocess
import threading
import time

import numpy as np
import pytest

from auto_optimal_control.build import build_kernels
from auto_optimal_control.pontryagin import derive
from auto_optimal_control.examples import all_examples, double_integrator_min_energy
from fast_numerical_integration.integrators import integrate, solve_ivp

NAMES = list(all_examples())


def random_point(pmp, rng):
    """t, z, p with positive symbols kept positive."""
    z = np.array([rng.uniform(0.3, 2.0) if s.is_positive else rng.uniform(-2.0, 2.0)
                  for s in pmp.z])
    p = np.array([rng.uniform(0.3, 2.0) if s.is_positive else rng.uniform(-2.0, 2.0)
                  for s in pmp.params])
    return rng.uniform(0.1, 2.0), z, p


def fd(f, x, h=1e-6):
    """Central differences of f at x: columns are df/dx_j."""
    x = np.asarray(x, dtype=float)
    cols = []
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = h * max(1.0, abs(x[j]))
        cols.append((np.asarray(f(x + e)) - np.asarray(f(x - e))) / (2 * e[j]))
    return np.array(cols).T


@pytest.mark.parametrize("name", NAMES)
def test_pointwise_functions_match_numpy_reference(compiled, name):
    pmp, num, k = compiled[name]
    rng = np.random.default_rng(3)
    bc = pmp.boundary
    for _ in range(25):
        t, z, p = random_point(pmp, rng)
        np.testing.assert_allclose(k.rhs(t, z, p), num.rhs(t, z, p), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(k.control(t, z, p), num.control(t, z, p), rtol=1e-12, atol=1e-13)
        assert k.hamiltonian(t, z, p) == pytest.approx(num.hamiltonian(t, z, p), rel=1e-12, abs=1e-13)
        assert k.cost_rate(t, z, p) == pytest.approx(num.cost_rate(t, z, p), rel=1e-12, abs=1e-13)
        np.testing.assert_allclose(k.switching(t, z, p), num.switching(t, z, p), rtol=1e-12, atol=1e-13)
        # On the arc the point lies on, the arc dynamics are the dynamics.
        arc = k.initial_arc(t, z, p) if k.N_SWITCHING else None
        np.testing.assert_allclose(k.rhs_arc(t, z, p, arc), k.rhs(t, z, p), rtol=1e-12, atol=1e-13)

        z0, zf = rng.normal(size=len(z)), rng.normal(size=len(z))
        nu = rng.normal(size=len(bc.nu))
        tf = rng.uniform(0.5, 2.0)
        np.testing.assert_allclose(k.residual(z0, zf, nu, tf, p), num.residual(z0, zf, nu, tf, p),
                                   rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("name", NAMES)
def test_derivative_kernels_match_finite_differences(compiled, name):
    pmp, _, k = compiled[name]
    rng = np.random.default_rng(4)
    nsw = k.N_SWITCHING
    for _ in range(5):
        t, z, p = random_point(pmp, rng)
        for arc in ([None] if nsw == 0 else [np.zeros(nsw, int), np.ones(nsw, int)]):
            A = k.jacobian(t, z, p, arc)
            np.testing.assert_allclose(A, fd(lambda zz: k.rhs_arc(t, zz, p, arc), z),
                                       rtol=1e-6, atol=1e-7)
            B = k.dfdp(t, z, p, arc)
            np.testing.assert_allclose(B, fd(lambda pp: k.rhs_arc(t, z, pp, arc), p),
                                       rtol=1e-6, atol=1e-7)
        if nsw:
            St, Sz, Sp = k.switching_gradient(t, z, p)
            np.testing.assert_allclose(Sz, fd(lambda zz: k.switching(t, zz, p), z), rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(Sp, fd(lambda pp: k.switching(t, z, pp), p), rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(St, fd(lambda tt: k.switching(tt[0], z, p), [t])[:, 0],
                                       rtol=1e-6, atol=1e-7)

        bc = pmp.boundary
        z0, zf = rng.normal(size=len(z)), rng.normal(size=len(z))
        # Keep the low-thrust ||lam_v(t_f)|| away from 0 and the bang-bang
        # Hamiltonian away from its kink.
        nu, tf = rng.normal(size=len(bc.nu)), rng.uniform(0.5, 2.0)
        J = k.residual_jacobian(z0, zf, nu, tf, p)
        np.testing.assert_allclose(J["z0"], fd(lambda a: k.residual(a, zf, nu, tf, p), z0), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(J["zf"], fd(lambda a: k.residual(z0, a, nu, tf, p), zf), rtol=1e-6, atol=1e-7)
        if len(nu):
            np.testing.assert_allclose(J["nu"], fd(lambda a: k.residual(z0, zf, a, tf, p), nu), rtol=1e-6, atol=1e-7)
        if pmp.problem.free_final_time:
            np.testing.assert_allclose(J["tf"], fd(lambda a: k.residual(z0, zf, nu, a[0], p), [tf])[:, 0],
                                       rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(J["p"], fd(lambda a: k.residual(z0, zf, nu, tf, a), p), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("name", NAMES)
def test_standalone_c_compiles_cleanly(compiled, name, tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler")
    _, _, k = compiled[name]
    h, c = k.write_c(tmp_path)
    assert h.read_text().count("#define") >= 7
    subprocess.run([cc, "-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic", "-c", str(c),
                    "-o", str(tmp_path / "k.o")], check=True)


def test_cache_reuses_the_compiled_module(compiled, kernel_cache):
    pmp, _, k = compiled["min_time"]
    mtime = k.path.stat().st_mtime_ns
    start = time.perf_counter()
    again = build_kernels(pmp, "min_time", cache_dir=kernel_cache)
    assert time.perf_counter() - start < 2.0
    assert again.path == k.path and again.path.stat().st_mtime_ns == mtime
    assert again.module is k.module
    # A different problem gets a different module.
    other = build_kernels(derive(double_integrator_min_energy()), "min_time", cache_dir=kernel_cache)
    assert other.path != k.path


def test_native_dynamics_integrate_like_the_reference_and_without_the_gil(compiled):
    pmp, num, k = compiled["zermelo"]
    p = [1.3, 0.8, 0.0, 0.0]
    z0 = [0.0, 0.0, -0.35, -0.68]
    kw = dict(method="DOP853", rtol=1e-12, atol=1e-12)
    native = solve_ivp(k.Dynamics(p), (0.0, 1.5), z0, **kw)
    ref = solve_ivp(num.ode(p), (0.0, 1.5), z0, **kw)
    np.testing.assert_allclose(native.y[:, -1], ref.y[:, -1], rtol=1e-11, atol=1e-12)
    res = integrate(k.Dynamics(p), (0.0, 1.5), z0, **kw)
    assert res.gil_released
    np.testing.assert_allclose(res.y, native.y[:, -1], rtol=1e-14)
    # Runtime parameters: changing them equals constructing with them.
    sys = k.Dynamics(p)
    sys.params = [1.0, 0.5, 0.0, 0.0]
    a = solve_ivp(sys, (0.0, 1.0), z0, **kw)
    b = solve_ivp(k.Dynamics([1.0, 0.5, 0.0, 0.0]), (0.0, 1.0), z0, **kw)
    np.testing.assert_array_equal(a.y, b.y)
    assert sys.param_names == ("V", "k", "x_T", "y_T")


def test_argument_validation(compiled):
    _, _, k = compiled["min_time"]
    with pytest.raises(TypeError, match="params"):
        k.rhs(0.0, np.zeros(4))
    with pytest.raises(ValueError, match="4 values"):
        k.rhs(0.0, np.zeros(3), [1, 1, 0])
    with pytest.raises(ValueError, match="arc"):
        k.rhs_arc(0.0, np.zeros(4), [1, 1, 0], [2])
    with pytest.raises(ValueError, match="param_sensitivity"):
        k.propagate(0.0, np.zeros(4), 1.0, [1, 1, 0], param_sensitivity=True)
    with pytest.raises(ValueError, match="out of range"):
        k.Switching(1, [1, 1, 0])
