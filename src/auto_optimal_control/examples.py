"""Textbook optimal control problems with known solutions.

Each function returns an :class:`~auto_optimal_control.problem.OptimalControlProblem`.
Constants are runtime parameters, so a derived system serves every value.
"""

import sympy as sp

from auto_optimal_control.problem import Box, OptimalControlProblem, PathConstraint, UnitVector


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


def bryson_denham():
    """The Bryson-Denham problem: ``min int u^2/2``, ``x'' = u``, with ``x <= l``.

    From ``x = 0, v = 1`` back to ``x = 0, v = -1`` at ``t = 1``. The limit
    ``x - l <= 0`` has order 2 and carries the penalty weight ``rho``. In closed
    form (Bryson, Denham and Dreyfus 1963) the limit is inactive for
    ``l >= 1/4``, touched at ``t = 1/2`` for ``1/6 <= l < 1/4``, and ridden on
    ``[3 l, 1 - 3 l]`` with cost ``4 / (9 l)`` for ``l < 1/6``.

    params: ``l, rho``.
    """
    t, x, v, u = sp.symbols("t x v u", real=True)
    ell, rho = sp.symbols("l rho", positive=True)
    return OptimalControlProblem(
        name="bryson_denham", time=t, states=[x, v], controls=[u], params=[ell, rho],
        dynamics=[v, u], running_cost=u ** 2 / 2, initial_state=[0, 1],
        final_state={x: 0, v: -1}, final_time=1,
        path_constraints=[PathConstraint(x - ell, "x_max", penalty=rho)])


def velocity_limited_move(bounded=False):
    """Minimum-energy rest-to-rest move ``x: 0 -> 1`` in ``t = 1`` with ``v <= V``.

    ``x'' = u``, ``min int u^2/2``. Unconstrained, ``v`` peaks at 1.5, so for
    ``1 < V < 3/2`` the order-1 limit ``v - V <= 0`` is ridden on
    ``[t1, 1 - t1]`` with ``t1 = 3 (V - 1) / (2 V)``. Before it,
    ``u = 2 V (t1 - t) / t1^2`` (decelerating smoothly onto the limit), and the
    cost is ``4 V^2 / (3 t1)``.

    With ``bounded=True`` the force is also limited, ``|u| <= u_max``. Below
    ``u_max = 2 V / t1`` the move starts at full force, then eases onto the
    limit: see :func:`velocity_limited_move_bounded_exact`.

    params: ``V, rho`` (plus ``u_max`` if bounded).
    """
    t, x, v, u = sp.symbols("t x v u", real=True)
    V, rho = sp.symbols("V rho", positive=True)
    params, sets = [V, rho], []
    if bounded:
        umax = sp.Symbol("u_max", positive=True)
        params.append(umax)
        sets.append(Box(u, -umax, umax))
    return OptimalControlProblem(
        name="velocity_limited_move" + ("_bounded" if bounded else ""), time=t, states=[x, v],
        controls=[u], params=params, dynamics=[v, u], running_cost=u ** 2 / 2,
        initial_state=[0, 0], final_state={x: 1, v: 0}, final_time=1, control_sets=sets,
        path_constraints=[PathConstraint(v - V, "v_max", penalty=rho)])


def velocity_limited_move_bounded_exact(V, umax):
    """The closed form of :func:`velocity_limited_move` with ``|u| <= u_max``.

    The move is symmetric. Up to the limit, ``u = min(u_max, c (t1 - t))``:
    full force for ``s``, then a linear ramp of length ``tau = u_max / c`` down
    to ``u = 0`` at ``t1 = s + tau``, where ``v`` reaches ``V`` (the costate
    ``lambda_v`` vanishes there by continuity of ``H``). Then
    ``v(t1) = u_max (s + tau / 2) = V`` and
    ``x(t1) = u_max (s^2 / 2 + s tau + tau^2 / 3)``, and the cruise covers the
    rest: ``2 x(t1) + V (1 - 2 t1) = 1``. Solving that for ``tau`` gives
    ``J = u_max^2 (s + tau / 3)``.

    Returns ``(J, t1, s)``. Valid for ``V (1 - V / u_max) > 1`` (otherwise
    even full force cannot cover the distance under the limit) and while
    ``s > 0`` (``u_max < 2 V / t1`` of the unbounded move).
    """
    from scipy.optimize import brentq

    def parts(tau):
        s = V / umax - tau / 2
        return s, umax * (s ** 2 / 2 + s * tau + tau ** 2 / 3)

    def gap(tau):
        s, x1 = parts(tau)
        return 2 * x1 + V * (1 - 2 * (s + tau)) - 1

    if not V * (1 - V / umax) > 1:
        raise ValueError("infeasible: the distance cannot be covered under both limits")
    tau = brentq(gap, 1e-12, 2 * V / umax)
    s, _ = parts(tau)
    if not s > 0:
        raise ValueError("u_max is not active: use the unbounded closed form")
    return umax ** 2 * (s + tau / 3), s + tau, s


def speed_limited_min_time(smoothed=False):
    """Minimum-time rest-to-rest run over a distance ``D`` with ``|u| <= u_max`` and ``v <= V``.

    ``x'' = u``: a train between two stations, with a speed limit. The
    control is bang-bang on ``S = lambda_v``: full power to the limit (at
    ``t1 = V / u_max``), coast at ``v = V`` with ``u = 0`` (a boundary arc of
    the order-1 limit ``v - V <= 0``), then full braking. While
    ``D > V^2 / u_max``, ``t_f = D / V + V / u_max``.

    With ``smoothed=True`` the running cost is ``1 + eps u^2 / 2`` instead of
    ``1``: the control is then clipped-linear, and ``eps -> 0`` recovers the
    bang-bang run. That variant also has a penalty weight ``rho`` on the
    limit, for exterior-penalty continuation.

    params: ``V, D, u_max`` (plus ``eps, rho`` if smoothed).
    """
    t, x, v, u = sp.symbols("t x v u", real=True)
    V, D, umax = sp.symbols("V D u_max", positive=True)
    params, cost, penalty = [V, D, umax], sp.S.One, None
    if smoothed:
        eps, rho = sp.symbols("eps rho", positive=True)
        params += [eps, rho]
        cost, penalty = 1 + eps * u ** 2 / 2, rho
    return OptimalControlProblem(
        name="speed_limited_min_time" + ("_smoothed" if smoothed else ""), time=t,
        states=[x, v], controls=[u], params=params, dynamics=[v, u], running_cost=cost,
        initial_state=[0, 0], final_state={x: D, v: 0}, control_sets=[Box(u, -umax, umax)],
        path_constraints=[PathConstraint(v - V, "v_max", penalty=penalty)])


def singular_regulator(smoothed=False):
    """``min int x^2``, ``x' = u``, ``|u| <= 1``, from ``x(0) = a`` to ``x(T) = a``.

    The simplest singular arc. ``S = lambda``, ``S' = -2 x`` and
    ``S'' = -2 u``, so the singular arc is ``x = 0`` with ``u_s = 0``, and
    its Legendre-Clebsch coefficient ``-dS''/du = 2`` is positive. For
    ``T > 2 a > 0``: ``u = -1`` to ``t = a``, singular to ``T - a``, then
    ``u = +1``. ``lambda(0) = a^2``, ``nu = -a^2`` and the cost is
    ``2 a^3 / 3``.

    With ``smoothed=True`` the running cost is ``x^2 + eps u^2 / 2``: the
    control is then clipped-linear, and ``eps -> 0`` approaches the
    bang-singular one.

    params: ``T, a`` (plus ``eps`` if smoothed).
    """
    t, x, u = sp.symbols("t x u", real=True)
    T, a = sp.symbols("T a", positive=True)
    params, cost = [T, a], x ** 2
    if smoothed:
        eps = sp.Symbol("eps", positive=True)
        params.append(eps)
        cost += eps * u ** 2 / 2
    return OptimalControlProblem(
        name="singular_regulator" + ("_smoothed" if smoothed else ""), time=t, states=[x],
        controls=[u], params=params, dynamics=[u], running_cost=cost, initial_state=[a],
        final_state={x: a}, final_time=T, control_sets=[Box(u, -1, 1)])


def goddard(dynamic_pressure=False):
    """Goddard's rocket: the highest climb on a given propellant, through drag.

    Vertical flight in normalized units (Seywald & Cliff 1992; Graichen &
    Petit 2008): ``h' = v``, ``v' = (u - D) / m - 1 / h^2``, ``m' = -u / c``
    with drag ``D = D_0 v^2 exp(-beta (h - 1))`` and thrust ``0 <= u <=
    u_max``. From rest at ``h = 1``, ``m = 1``, burn down to ``m(t_f) = m_f``
    and maximize ``h(t_f)``, with ``t_f`` free. The standard values are
    ``GODDARD_PARAMS``: ``D_0 = 310``, ``beta = 500``, ``c = 0.5``,
    ``u_max = 3.5``, ``m_f = 0.6``.

    Full thrust is too much in the dense air: the optimum is full thrust, a
    singular arc on which the thrust balances drag against the climb, then
    a coast. ``h(t_f) = 1.01283692`` at ``t_f = 0.19885626`` (Graichen &
    Petit 2008).

    With ``dynamic_pressure=True`` the path constraint
    ``q = rho_0 v^2 exp(-beta (h - 1)) / 2 <= q_max`` (order 1) is added,
    with ``rho_0 = 12400`` (so ``q = 20 D``) and ``q_max = 10`` in
    ``GODDARD_Q_PARAMS``. Unconstrained, ``q`` peaks near 14.9 at the
    singular entry, so the limit is ridden between the full-thrust and the
    singular arcs: ``h(t_f) = 1.01271727`` at ``t_f = 0.20407108``
    (Graichen & Petit 2008). ``rho`` weighs its exterior penalty.

    params: ``D_0, beta, c, u_max, m_f`` (plus ``rho_0, q_max, rho`` with the
    limit).
    """
    t, v, u = sp.symbols("t v u", real=True)
    h, m = sp.symbols("h m", positive=True)       # radius and mass
    D0, beta, c, umax, mf = sp.symbols("D_0 beta c u_max m_f", positive=True)
    params, limits = [D0, beta, c, umax, mf], []
    if dynamic_pressure:
        rho0, qmax = sp.symbols("rho_0 q_max", positive=True)
        rho = sp.Symbol("rho", nonnegative=True)
        params += [rho0, qmax, rho]
        q = rho0 * v ** 2 * sp.exp(-beta * (h - 1)) / 2
        limits.append(PathConstraint(q - qmax, "q_max", penalty=rho))
    drag = D0 * v ** 2 * sp.exp(-beta * (h - 1))
    return OptimalControlProblem(
        name="goddard" + ("_q" if dynamic_pressure else ""), time=t, states=[h, v, m],
        controls=[u], params=params, dynamics=[v, (u - drag) / m - 1 / h ** 2, -u / c],
        terminal_cost=-h, initial_state=[1, 0, 1], final_state={m: mf},
        control_sets=[Box(u, 0, umax)], path_constraints=limits)


GODDARD_PARAMS = (310.0, 500.0, 0.5, 3.5, 0.6)
GODDARD_Q_PARAMS = GODDARD_PARAMS + (12400.0, 10.0, 0.0)


def all_examples():
    """Every example, by name (the bounded and smoothed variants included)."""
    return {
        "zermelo": zermelo(),
        "min_time": double_integrator_min_time(),
        "min_energy": double_integrator_min_energy(),
        "min_energy_bounded": double_integrator_min_energy(bounded=True),
        "soft_terminal": soft_terminal_lq(),
        "low_thrust": planar_low_thrust_fuel(),
        "bryson_denham": bryson_denham(),
        "velocity_limited_move": velocity_limited_move(),
        "velocity_limited_move_bounded": velocity_limited_move(bounded=True),
        "speed_limited_min_time": speed_limited_min_time(),
        "speed_limited_min_time_smoothed": speed_limited_min_time(smoothed=True),
        "singular_regulator": singular_regulator(),
        "singular_regulator_smoothed": singular_regulator(smoothed=True),
        "goddard": goddard(),
        "goddard_q": goddard(dynamic_pressure=True),
    }
