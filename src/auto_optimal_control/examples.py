"""Textbook optimal control problems with known solutions.

Each function returns an :class:`~auto_optimal_control.problem.OptimalControlProblem`.
Constants are runtime parameters, so a derived system serves every value.
"""

import sympy as sp

from auto_optimal_control.problem import Box, OptimalControlProblem, UnitVector


def zermelo():
    """Zermelo's navigation problem with a linearly sheared current.

    A boat with speed ``V`` steers the heading ``(ux, uy)`` through a current
    ``k * y`` along x, from the origin to ``(x_T, y_T)`` in minimum time.
    Known extremal (Bryson & Ho, section 2.7): ``tan(theta) = tan(theta_0) - k t``.

    params: ``V, k, x_T, y_T``.
    """
    t, x, y, ux, uy = sp.symbols("t x y u_x u_y", real=True)
    V, k = sp.symbols("V k", positive=True)
    xT, yT = sp.symbols("x_T y_T", real=True)
    return OptimalControlProblem(
        name="zermelo", time=t, states=[x, y], controls=[ux, uy], params=[V, k, xT, yT],
        dynamics=[V * ux + k * y, V * uy], running_cost=1,
        initial_state=[0, 0], final_state={x: xT, y: yT},
        control_sets=[UnitVector([ux, uy])])


def double_integrator_min_time():
    """Minimum-time double integrator ``x'' = u``, ``|u| <= u_max``, to the origin.

    Bang-bang with at most one switch. From rest at ``x = a > 0``,
    ``u = -u_max`` then ``+u_max``, switching at ``sqrt(a/u_max)`` and
    arriving at ``t_f = 2 sqrt(a/u_max)``.

    params: ``u_max, a, b`` (initial position and velocity).
    """
    t, x, v, u = sp.symbols("t x v u", real=True)
    umax = sp.Symbol("u_max", positive=True)
    a, b = sp.symbols("a b", real=True)
    return OptimalControlProblem(
        name="double_integrator_min_time", time=t, states=[x, v], controls=[u],
        params=[umax, a, b], dynamics=[v, u], running_cost=1,
        initial_state=[a, b], final_state={x: 0, v: 0},
        control_sets=[Box(u, -umax, umax)])


def double_integrator_min_energy(bounded=False):
    """Minimum-energy double integrator: ``min int u^2/2``, ``x'' = u``, fixed ``t_f = T``.

    From ``(a, b)`` to the origin. Unbounded, ``u*`` is linear in time. With
    ``bounded=True`` the control is limited to ``|u| <= u_max`` and the law
    becomes the clipped one.

    params: ``T, a, b`` (plus ``u_max`` if bounded).
    """
    t, x, v, u = sp.symbols("t x v u", real=True)
    T = sp.Symbol("T", positive=True)
    a, b = sp.symbols("a b", real=True)
    params, sets = [T, a, b], []
    if bounded:
        umax = sp.Symbol("u_max", positive=True)
        params.append(umax)
        sets.append(Box(u, -umax, umax))
    return OptimalControlProblem(
        name="double_integrator_min_energy", time=t, states=[x, v], controls=[u],
        params=params, dynamics=[v, u], running_cost=u ** 2 / 2,
        initial_state=[a, b], final_state={x: 0, v: 0}, final_time=T, control_sets=sets)


def soft_terminal_lq():
    """``min w x(T)^2 / 2 + int u^2/2``, ``x' = u``, ``x(0) = a``, free final state.

    Exercises transversality with a terminal cost:
    ``lambda(T) = w x(T)``, giving ``lambda = w a / (1 + w T)``.

    params: ``T, w, a``.
    """
    t, x, u = sp.symbols("t x u", real=True)
    T, w = sp.symbols("T w", positive=True)
    a = sp.Symbol("a", real=True)
    return OptimalControlProblem(
        name="soft_terminal_lq", time=t, states=[x], controls=[u], params=[T, w, a],
        dynamics=[u], running_cost=u ** 2 / 2, terminal_cost=w * x ** 2 / 2,
        initial_state=[a], final_time=T)


def planar_low_thrust_fuel():
    """Fuel-optimal planar low-thrust transfer between circular orbits (Cartesian).

    States ``r = (r_x, r_y)``, ``v = (v_x, v_y)`` and mass ``m``; controls the
    thrust direction ``(a_x, a_y)`` and throttle ``delta in [0, 1]``; thrust
    ``T_max``, exhaust speed ``c``. Minimizes propellant ``int T_max delta / c``
    over a fixed transfer time ``t_f`` from the circular orbit of radius
    ``r_0`` to the one of radius ``r_f`` (any phase).

    The thrust direction follows the primer vector ``-lambda_v / ||lambda_v||``
    and the throttle is bang-bang on
    ``S = (T_max / c) (1 - c ||lambda_v|| / m - lambda_m)``.

    params: ``mu, T_max, c, m_0, r_0, r_f, t_f``.
    """
    t = sp.Symbol("t", real=True)
    rx, ry, vx, vy, ax, ay = sp.symbols("r_x r_y v_x v_y a_x a_y", real=True)
    m = sp.Symbol("m", positive=True)
    delta = sp.Symbol("delta", real=True)
    mu, Tmax, c, m0, r0, rf, tf = sp.symbols("mu T_max c m_0 r_0 r_f t_f", positive=True)
    r3 = (rx ** 2 + ry ** 2) ** sp.Rational(3, 2)
    thrust = Tmax * delta / m
    return OptimalControlProblem(
        name="planar_low_thrust_fuel", time=t, states=[rx, ry, vx, vy, m],
        controls=[ax, ay, delta], params=[mu, Tmax, c, m0, r0, rf, tf],
        dynamics=[vx, vy, -mu * rx / r3 + thrust * ax, -mu * ry / r3 + thrust * ay,
                  -Tmax * delta / c],
        running_cost=Tmax * delta / c,
        initial_state=[r0, 0, 0, sp.sqrt(mu / r0), m0],
        terminal_constraints=[rx ** 2 + ry ** 2 - rf ** 2,
                              rx * vx + ry * vy,
                              vx ** 2 + vy ** 2 - mu / rf],
        final_time=tf,
        control_sets=[UnitVector([ax, ay]), Box(delta, 0, 1)])
