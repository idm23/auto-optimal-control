"""Record and animate how the Earth-Mars fuel-optimal rendezvous is solved.

1. **Energy-optimal shooting.** Random costates in [0, 1]^7 until one
   converges. Every Newton evaluation of the winning start is a real
   trajectory from Earth; it misses Mars until the residual vanishes.
2. **Energy-to-fuel continuation.** ``L = (T/c)(delta - eps delta (1 - delta))``
   from ``eps = 1`` (smooth throttle) to ``1e-4``: the throttle sharpens into
   bang-bang, and the final mass rises.
3. **The fuel-optimal flight**, solved exactly with located switches: the
   spacecraft thrusts where the switching function is negative and coasts
   where it is positive.
"""

import numpy as np

from problems import (MARS, TU, ShootingError, SingleShooting, _kernels, continuation,
                      earth_mars_problem)
from visuals.common import (BLUE, GRID, INK, INK2, MUTED, ORANGE, RAMP, Problem, header, load, recording,
                            save, style, write)

__all__ = ["record", "animate", "save", "load", "PROBLEM"]
DAY = 86400.0 / TU


PROBLEM = Problem(
    name="Earth-to-Mars low-thrust rendezvous",
    story=("A 1000 kg spacecraft with a 0.5 N ion engine (specific impulse 2000 s) starts at "
           "Earth's position and velocity and must\nmatch Mars's 348.8 days later, using the "
           "least propellant. A benchmark of Taheri et al. (arXiv:2409.01490)."),
    formulation=[
        ("maximize", r"$m(t_f)$,  i.e.  minimize $\int_0^{t_f} \frac{T_{max}}{c}\,\delta\,dt$"),
        ("dynamics", r"$\ddot{\mathbf{r}} = -\frac{\mu}{r^3}\,\mathbf{r} + \frac{T_{max}\,"
                     r"\delta}{m}\,\hat{\mathbf{a}}, \quad \dot m = -\frac{T_{max}\,\delta}{c}$"),
        ("controls", r"throttle $\delta \in [0, 1]$,  thrust direction $|\hat{\mathbf{a}}| = 1$"),
        ("boundary", r"$(\mathbf{r}, \mathbf{v})(0)$ = Earth,  $(\mathbf{r}, \mathbf{v})(t_f)$ "
                     r"= Mars,  $t_f = 348.795$ days"),
    ],
    solved_by=[
        "Energy-optimal first: random costates in [0, 1]⁷ until shooting converges (a smooth "
        "throttle).",
        "Energy-to-fuel continuation: L = (T/c)(δ − ε δ(1 − δ)), ε lowered from 1 to 1e-4.",
        "Fuel-optimal bang-bang shooting, every throttle switch located exactly.",
    ],
    summary="max m(t_f),  Earth → Mars rendezvous in 348.8 days,  0.5 N,  Isp 2000 s,  "
            "throttle δ ∈ [0, 1]",
    source="Reference: m(t_f) = 603.935 kg, arXiv:2409.01490 Table 2",
)

def _kepler(r0, v0, ts, mu):
    """Two-body positions at times ts (sorted either way) from (r0, v0) at ts[0]."""
    from scipy.integrate import solve_ivp

    def f(t, y):
        r = y[:3]
        return np.concatenate([y[3:], -mu * r / np.linalg.norm(r) ** 3])

    sol = solve_ivp(f, (ts[0], ts[-1]), np.concatenate([r0, v0]), t_eval=ts, rtol=1e-11,
                    atol=1e-12)
    return sol.y[:3].T


def _period(r0, v0, mu):
    a = 1.0 / (2.0 / np.linalg.norm(r0) - np.dot(v0, v0) / mu)
    return 2 * np.pi * np.sqrt(a ** 3 / mu)


def _sample(k, z0, params, tf, n):
    try:
        tr = k.trajectory(0.0, z0, tf, params, n_points=n)
    except (RuntimeError, ValueError):
        return None
    if tr["status"] != 0 or not np.all(np.isfinite(tr["z"])):
        return None
    return tr


def record(cache_dir=None, seed=0, n_tries=200, verbose=True):
    def say(msg):
        if verbose:
            print(msg, flush=True)

    kh = _kernels(earth_mars_problem(True), cache_dir)
    kf = _kernels(earth_mars_problem(False), cache_dir)
    M = MARS
    x0 = np.concatenate([M["rE"], M["vE"], [1.0]])
    base = [M["mu"], M["T"], M["c"], M["tf"], *x0, *M["rM"], *M["vM"]]
    tf = M["tf"]
    out = {"tf": tf, "m0": M["m0"]}

    # 1. Energy-optimal shooting from random costates.
    energy = recording(SingleShooting)(kh, base + [1.0])
    rng = np.random.default_rng(seed)
    tries_R, win = [], None
    for _ in range(n_tries):
        try:
            guess = energy.guess_multipliers(rng.uniform(0, 1, 7))
        except (ShootingError, RuntimeError, ValueError, np.linalg.LinAlgError):
            tries_R.append(np.nan)
            continue
        energy.evaluations.clear()
        sol = energy.solve(guess)
        tries_R.append(sol.residual_norm)
        if sol.success:
            win = (guess, list(energy.evaluations), sol)
            break
    guess, evals, sol = win
    out["tries_R"] = np.array(tries_R)
    pos, thr, mass, Rs = [], [], [], []
    for v, R in [(guess, np.nan)] + evals:
        tr = _sample(kh, np.concatenate([x0, v[:7]]), base + [1.0], tf, 201)
        if tr is None:
            continue
        pos.append(tr["z"][:, :3])
        thr.append(tr["u"][:, 3])
        mass.append(tr["z"][:, 6] * M["m0"])
        Rs.append(R)
    out["shoot_t"] = np.linspace(0, tf, 201)
    out["shoot_r"], out["shoot_delta"], out["shoot_R"] = np.array(pos), np.array(thr), np.array(Rs)
    out["shoot_m"] = np.array(mass)
    say(f"energy: {len(tries_R)} random starts, {len(evals)} evaluations in the winner")

    # 2. Continuation in eps.
    def at(eps, g):
        return SingleShooting(kh, base + [eps]).solve(g, tol=1e-9)

    path = continuation(at, 1.0, [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 1e-3, 1e-4], sol.v)
    path = [(1.0, sol)] + path
    eps, cr, cd, cm = [], [], [], []
    for e, s in path:
        tr = _sample(kh, s.z0, base + [e], tf, 601)
        eps.append(e)
        cr.append(tr["z"][:, :3])
        cd.append(tr["u"][:, 3])
        cm.append(tr["z"][:, 6] * M["m0"])
    out["cont_t"] = np.linspace(0, tf, 601)
    out["cont_eps"], out["cont_r"] = np.array(eps), np.array(cr)
    out["cont_delta"], out["cont_m"] = np.array(cd), np.array(cm)
    say(f"continuation: {len(path) - 1} steps, final mass {cm[-1][-1]:.3f} kg")

    # 3. The exact fuel-optimal flight.
    fuel = SingleShooting(kf, base).solve(path[-1][1].v)
    tr = _sample(kf, fuel.z0, base, tf, 721)
    S = np.array([kf.switching(ti, zi, base)[0] for ti, zi in zip(tr["t"], tr["z"])])
    r = kf.propagate(fuel.t0, fuel.z0, tf, base)
    out["fuel_t"], out["fuel_r"] = tr["t"], tr["z"][:, :3]
    out["fuel_delta"], out["fuel_dir"] = tr["u"][:, 3], tr["u"][:, :3]
    out["fuel_m"], out["fuel_S"] = tr["z"][:, 6] * M["m0"], S
    out["switch_t"] = r["switch_times"]
    out["fuel_residual"] = fuel.residual_norm
    say(f"fuel: m(tf) = {out['fuel_m'][-1]:.4f} kg, switches at "
        f"{np.round(out['switch_t'] / DAY, 1).tolist()} days")

    # Planets along the flight, and their full orbits.
    out["earth"] = _kepler(M["rE"], M["vE"], tr["t"], M["mu"])
    out["mars"] = _kepler(M["rM"], M["vM"], tr["t"][::-1], M["mu"])[::-1]
    for name, r0, v0 in (("earth_orbit", M["rE"], M["vE"]), ("mars_orbit", M["rM"], M["vM"])):
        P = _period(r0, v0, M["mu"])
        out[name] = _kepler(r0, v0, np.linspace(0, P, 361), M["mu"])
    return out


CAPTIONS = {
    "shoot": ("1  Energy-optimal shooting",
              "Random initial costates until one converges ({n} start{s}). Every Newton "
              "evaluation is a real flight from\nEarth with a smooth throttle; it misses Mars "
              "until the shooting residual vanishes."),
    "cont": ("2  Energy-to-fuel continuation",
             "The cost moves from energy to fuel, L = (T/c)(δ − ε δ(1 − δ)), as ε goes from 1 to "
             "1e-4. The throttle\nsharpens from smooth into full-on / off, and the spacecraft "
             "arrives heavier."),
    "flight": ("3  The fuel-optimal flight",
               "Solved exactly: the throttle is full on where the switching function S < 0 and "
               "off where S > 0, with\nthe {n} switches located by the integrator. Arrival "
               "mass {m:.3f} kg of {m0:.0f} kg."),
}


def _plan(data, fps):
    plan = []
    n = len(data["shoot_R"])
    for i in range(n):
        plan += [("shoot", i)] * max(2, int(75 / n))
    plan += [("shoot", n - 1)] * fps
    for i in range(len(data["cont_eps"])):
        plan += [("cont", i)] * 12
    plan += [("cont", len(data["cont_eps"]) - 1)] * fps
    n = len(data["fuel_t"])
    plan += [("flight", int(i)) for i in np.linspace(0, n - 1, 8 * fps)]
    plan += [("flight", n - 1)] * fps
    return plan


def animate(data, path, fps=30, dpi=100, stills_dir=None):
    plt = style()
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("throttle", [RAMP[0], RAMP[-1]])
    days = lambda t: np.asarray(t) / DAY          # noqa: E731
    T_days = days(data["tf"])
    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], left=0.05, right=0.97, top=0.80,
                          bottom=0.105, hspace=0.45, wspace=0.16)
    sky = fig.add_subplot(gs[:, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[1, 1], sharex=ax_d)
    ax_k = fig.add_subplot(gs[2, 1])
    title, caption = header(fig)
    status = sky.text(0.02, 0.98, "", transform=sky.transAxes, va="top", fontsize=11)

    sky.set_aspect("equal")
    sky.set(xlim=(-1.9, 1.9), ylim=(-1.9, 1.9), xlabel="x [AU]", ylabel="y [AU]")
    sky.grid(color=GRID, lw=0.6)
    sky.plot(*data["earth_orbit"][:, :2].T, color=MUTED, lw=1)
    sky.plot(*data["mars_orbit"][:, :2].T, color=MUTED, lw=1, ls="--")
    sky.plot(0, 0, "o", color="#eda100", ms=12)
    e0, mf = data["earth"][0], data["mars"][-1]
    sky.plot(*e0[:2], "o", color=INK2, ms=6, mfc="none")
    sky.plot(*mf[:2], "o", color=ORANGE, ms=7, mfc="none")
    sky.text(e0[0] + 0.08, e0[1] - 0.15, "Earth at departure", color=INK2, fontsize=9)
    sky.text(mf[0] + 0.08, mf[1] + 0.05, "Mars at arrival", color=ORANGE, fontsize=9)
    hist = LineCollection([], colors=MUTED, linewidths=0.8, alpha=0.6)
    traj = LineCollection([], linewidths=2.2, cmap=cmap)
    traj.set_clim(0, 1)
    sky.add_collection(hist)
    sky.add_collection(traj)
    craft, = sky.plot([], [], "o", color=INK, ms=6, zorder=6)
    thrust, = sky.plot([], [], color=ORANGE, lw=2.5, zorder=5)
    earth, = sky.plot([], [], "o", color=BLUE, ms=8, zorder=5, label="Earth")
    mars, = sky.plot([], [], "o", color=ORANGE, ms=7, zorder=5, label="Mars")
    craft.set_label("spacecraft (thrust in orange)")
    sky.legend(handles=[earth, mars, craft], loc="lower left", frameon=False, fontsize=9)
    cax = sky.inset_axes([0.6, 0.13, 0.36, 0.025])
    cb = fig.colorbar(traj, cax=cax, orientation="horizontal")
    cb.set_label("throttle δ along the path", fontsize=9)
    cax.tick_params(labelsize=8)

    for ax in (ax_d, ax_m):
        ax.grid(color=GRID, lw=0.8)
        ax.set_xlim(0, T_days)
    ax_d.set(ylim=(-0.05, 1.08), ylabel="throttle δ")
    m_all = np.concatenate([data["cont_m"].ravel(), data["fuel_m"], data["shoot_m"][-1]])
    ax_m.set(ylim=(m_all.min() - 15, data["m0"] + 10), ylabel="mass [kg]", xlabel="t [days]")
    hist_d = LineCollection([], colors=MUTED, linewidths=1, alpha=0.6)
    hist_m = LineCollection([], colors=MUTED, linewidths=1, alpha=0.6)
    ax_d.add_collection(hist_d)
    ax_m.add_collection(hist_m)
    cur_d, = ax_d.plot([], [], color=BLUE, lw=2)
    cur_m, = ax_m.plot([], [], color=BLUE, lw=2)
    heads = [ax.axvline(np.nan, color=INK2, lw=1) for ax in (ax_d, ax_m)]
    state = {"stage": None}

    def colored(r, delta):
        pts = r[:, :2]
        traj.set_segments(np.stack([pts[:-1], pts[1:]], 1))
        traj.set_array(0.5 * (delta[:-1] + delta[1:]))

    def setup(stage):
        ax_k.cla()
        ax_k.grid(color=GRID, lw=0.8)
        for art in (hist, hist_d, hist_m, traj):
            art.set_segments([])
        for art in (cur_d, cur_m, craft, thrust, earth, mars):
            art.set_data([], [])
        for h in heads:
            h.set_xdata([np.nan])
        head, body = CAPTIONS[stage]
        if stage == "shoot":
            n = len(data["tries_R"])
            body = body.format(n=n, s="" if n == 1 else "s")
            R = data["shoot_R"]
            fin = R[np.isfinite(R) & (R > 0)]
            ax_k.set(yscale="log", xlim=(-0.5, len(R) - 0.5), ylim=(fin.min() * 0.3, fin.max() * 3),
                     xlabel="residual evaluation", ylabel="‖shooting residual‖")
            state["k"], = ax_k.plot([], [], color=BLUE, lw=2, marker="o", ms=4)
        elif stage == "cont":
            mf_ = data["cont_m"][:, -1]
            ax_k.set(xlim=(-0.5, len(mf_) - 0.5), ylim=(mf_.min() - 3, data["fuel_m"][-1] + 3),
                     xlabel="continuation step", ylabel="arrival mass [kg]")
            ax_k.set_xticks(range(len(mf_)), [f"{e:g}" for e in data["cont_eps"]], fontsize=8)
            ax_k.set_xlabel("ε")
            ax_k.axhline(data["fuel_m"][-1], color=INK2, lw=1, ls="--")
            ax_k.text(0.01, 0.9, f"exact fuel-optimal: {data['fuel_m'][-1]:.3f} kg",
                      transform=ax_k.transAxes, color=INK2, fontsize=9)
            state["k"], = ax_k.plot([], [], color=BLUE, lw=2, marker="o", ms=5)
        else:
            body = body.format(n=len(data["switch_t"]), m=data["fuel_m"][-1], m0=data["m0"])
            t = days(data["fuel_t"])
            S = data["fuel_S"]
            ax_k.plot(t, S, color=BLUE, lw=2)
            ax_k.axhline(0, color=INK2, lw=1)
            ax_k.fill_between(t, S, 0, where=S < 0, color=BLUE, alpha=0.15, lw=0)
            for ts in days(data["switch_t"]):
                ax_k.axvline(ts, color=INK2, lw=0.8, ls=":")
            ax_k.set(xlim=(0, T_days), xlabel="t [days]", ylabel="switching function S")
            state["k"] = ax_k.axvline(0, color=INK2, lw=1)
            cur_d.set_data(t, data["fuel_delta"])
            cur_m.set_data(t, data["fuel_m"])
        title.set_text(head)
        caption.set_text(body)
        state["stage"] = stage

    def draw(frame):
        stage, i = frame
        if stage == "__reset__":
            state["stage"] = None
            return ()
        if stage != state["stage"]:
            setup(stage)
        if stage == "shoot":
            t = days(data["shoot_t"])
            r, d = data["shoot_r"], data["shoot_delta"]
            hist.set_segments([r[j][:, :2] for j in range(max(0, i - 8), i)])
            colored(r[i], d[i])
            hist_d.set_segments([np.column_stack([t, d[j]]) for j in range(max(0, i - 8), i)])
            cur_d.set_data(t, d[i])
            m = data["shoot_m"]
            hist_m.set_segments([np.column_stack([t, m[j]]) for j in range(max(0, i - 8), i)])
            cur_m.set_data(t, m[i])
            R = data["shoot_R"][: i + 1]
            state["k"].set_data(np.arange(i + 1), R)
            miss = np.linalg.norm(r[i][-1] - data["mars"][-1])
            r_txt = "initial guess" if not np.isfinite(R[-1]) else f"‖R‖ = {R[-1]:.1e}"
            status.set_text(f"evaluation {i}   {r_txt}   miss at arrival = {miss:.3f} AU")
        elif stage == "cont":
            t = days(data["cont_t"])
            r, d, m = data["cont_r"], data["cont_delta"], data["cont_m"]
            hist.set_segments([r[j][:, :2] for j in range(i)])
            colored(r[i], d[i])
            hist_d.set_segments([np.column_stack([t, d[j]]) for j in range(i)])
            hist_m.set_segments([np.column_stack([t, m[j]]) for j in range(i)])
            cur_d.set_data(t, d[i])
            cur_m.set_data(t, m[i])
            state["k"].set_data(np.arange(i + 1), m[: i + 1, -1])
            status.set_text(f"ε = {data['cont_eps'][i]:g}   arrival mass = {m[i][-1]:.3f} kg")
        else:
            r, d, a = data["fuel_r"], data["fuel_delta"], data["fuel_dir"]
            colored(r[: i + 1], d[: i + 1]) if i > 0 else traj.set_segments([])
            craft.set_data([r[i, 0]], [r[i, 1]])
            L = 0.25 * d[i]
            thrust.set_data([r[i, 0], r[i, 0] + L * a[i, 0]], [r[i, 1], r[i, 1] + L * a[i, 1]])
            earth.set_data([data["earth"][i, 0]], [data["earth"][i, 1]])
            mars.set_data([data["mars"][i, 0]], [data["mars"][i, 1]])
            tt = days(data["fuel_t"][i])
            for h in heads:
                h.set_xdata([tt])
            state["k"].set_xdata([tt])
            status.set_text(f"day {tt:5.1f}   {'THRUST' if d[i] > 0.5 else 'coast'}   "
                            f"mass = {data['fuel_m'][i]:.2f} kg")
        return ()

    plan = _plan(data, fps)
    mid = next(j for j, f in enumerate(plan) if f[0] == "flight" and f[1] >= 0.62 * len(data["fuel_t"]))
    return write(fig, draw, plan, path, fps, stills_dir, prefix="earth_mars_",
                 still_frames={"flight": mid}, problem=PROBLEM)
