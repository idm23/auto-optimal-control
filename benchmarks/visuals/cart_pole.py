"""Record and animate how the cart-pole benchmark is solved.

The solve has three stages, and the animation adds a fourth:

1. **Direct method.** Kelly's Hermite-Simpson collocation, from his
   straight-line guess (SLSQP). The early iterates are not dynamically
   feasible: the pole turns at a constant rate while the cart slides.
2. **Indirect polish.** Costates lifted from the direct solution, then Newton
   on the shooting residual. Every evaluation is a real extremal of the cart
   and pole, which misses the target until the residual vanishes.
3. **Random costate search.** Shooting from random costates finds several
   extremals, one cheaper than the published solution.
4. **Both extremals**, swinging up side by side with their controls.

:func:`record` runs the solves and returns plain arrays (it can be saved with
:func:`save` and reloaded with :func:`load`). :func:`animate` renders them
with matplotlib.
"""

import time

import numpy as np

from problems import (CART_POLE_PARAMS, ShootingError, SingleShooting, _kernels,
                      cart_pole_problem, costates_from_direct)
from visuals.common import BLUE, GRID, INK, INK2, MUTED, ORANGE, SURF, Problem, write

PUBLISHED_J = 58.80819
N_SAMPLES = 121          # time samples per recorded trajectory

PROBLEM = Problem(
    name="Cart-pole swing-up",
    story=("Swing a pole from hanging straight down to balanced upright while moving the cart "
           "1 m to the right, in exactly 2 s,\nwith the least force effort. The benchmark of "
           "Kelly's direct-collocation tutorial (SIAM Review, 2017)."),
    formulation=[
        ("minimize", r"$J = \int_0^{2} u^2\,dt$"),
        ("dynamics", r"$\ddot x = \frac{l m_2 \sin\theta\,\dot\theta^2 + u + m_2 g \cos\theta"
                     r"\sin\theta}{m_1 + m_2 \sin^2\theta}, \quad \ddot\theta = -\frac{l m_2"
                     r"\cos\theta\sin\theta\,\dot\theta^2 + u\cos\theta + (m_1 + m_2) g"
                     r"\sin\theta}{l\,(m_1 + m_2 \sin^2\theta)}$"),
        ("force limit", r"$|u| \leq 20$ N"),
        ("boundary", r"$x: 0 \to 1$ m,  $\theta: 0 \to \pi$,  at rest at $t = 0$ and "
                     r"$t = 2$ s"),
        ("constants", r"$m_1 = 1$ kg (cart),  $m_2 = 0.3$ kg (pole),  $l = 0.5$ m,  "
                      r"$g = 9.81$ m/s$^2$"),
    ],
    solved_by=[
        "Direct: Kelly's Hermite-Simpson collocation from his straight-line guess (SLSQP).",
        "Indirect polish: costates lifted from the direct solution, then Newton on the "
        "shooting residual.",
        "Random costate search: shooting from random costates finds other extremals, one "
        "cheaper than the published one.",
    ],
    summary="min ∫u² dt,  cart-pole swing-up in T = 2 s,  |u| ≤ 20 N,  cart moves 1 m",
    source="Reference: M. Kelly, An introduction to trajectory optimization, SIAM Review 59 "
           "(2017), section 6",
)


# =============================================================================
# Recording
# =============================================================================

class _RecordingShooting(SingleShooting):
    """SingleShooting that keeps every point the root finder evaluates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluations = []

    def residual_jacobian(self, v):
        R, J = super().residual_jacobian(v)
        self.evaluations.append((np.array(v, dtype=float), float(np.linalg.norm(R))))
        return R, J

    __call__ = residual_jacobian


def _direct_iterates(p, N=40):
    """Kelly's direct method with every SLSQP iterate: (t, [(X, U, J, defect)])."""
    from scipy.optimize import minimize

    from problems import _cart_pole_plant

    T, umax, d = p[6], p[4], p[5]
    h = T / N

    def split(w):
        return w[:4 * (N + 1)].reshape(N + 1, 4), w[4 * (N + 1):]

    def defects(w):
        X, U = split(w)
        out = []
        for i in range(N):
            fk, fk1 = _cart_pole_plant(X[i], U[i], p), _cart_pole_plant(X[i + 1], U[i + 1], p)
            um = 0.5 * (U[i] + U[i + 1])
            xm = 0.5 * (X[i] + X[i + 1]) + h / 8 * (fk - fk1)
            out.append(X[i + 1] - X[i] - h / 6 * (fk + 4 * _cart_pole_plant(xm, um, p) + fk1))
        out += [X[0], X[-1] - [d, np.pi, 0, 0]]
        return np.concatenate(out)

    def cost(w):
        _, U = split(w)
        um = 0.5 * (U[:-1] + U[1:])
        return h / 6 * np.sum(U[:-1] ** 2 + 4 * um ** 2 + U[1:] ** 2)

    t = np.linspace(0, T, N + 1)
    w0 = np.concatenate([np.stack([d * t / T, np.pi * t / T, 0 * t, 0 * t], 1).ravel(),
                         np.zeros(N + 1)])
    iterates = []

    def keep(w):
        X, U = split(w)
        iterates.append((X.copy(), U.copy(), float(cost(w)),
                         float(np.linalg.norm(defects(w)))))

    keep(w0)
    bounds = ([(-2, 2), (None, None), (None, None), (None, None)] * (N + 1)
              + [(-umax, umax)] * (N + 1))
    minimize(cost, w0, method="SLSQP", constraints={"type": "eq", "fun": defects},
             bounds=bounds, callback=keep, options={"maxiter": 500, "ftol": 1e-12})
    return t, iterates


def _sample(k, z0, p, n=N_SAMPLES):
    """(t, z, u) of the extremal from z0, or None if it cannot be propagated."""
    try:
        tr = k.trajectory(0.0, z0, p[6], p, n_points=n)
    except (RuntimeError, ValueError):
        return None
    if tr["status"] != 0 or not np.all(np.isfinite(tr["z"])):
        return None
    return tr["t"], tr["z"][:, :4], tr["u"][:, 0]


def record(cache_dir=None, seed=0, n_random=400, verbose=True):
    """Run the three solve stages and keep what the animation needs."""
    def say(msg):
        if verbose:
            print(msg, flush=True)

    start = time.perf_counter()
    k = _kernels(cart_pole_problem(), cache_dir)
    p = np.array(CART_POLE_PARAMS)
    out = {"params": p}

    # 1. Direct method.
    t_dir, its = _direct_iterates(p)
    out["direct_t"] = t_dir
    out["direct_X"] = np.array([X for X, *_ in its])
    out["direct_U"] = np.array([U for _, U, *_ in its])
    out["direct_J"] = np.array([J for *_, J, _ in its])
    out["direct_defect"] = np.array([d for *_, d in its])
    say(f"direct: {len(its) - 1} SLSQP iterations, J = {its[-1][2]:.4f} "
        f"({time.perf_counter() - start:.0f} s)")

    # 2. Indirect polish from the lifted costates.
    shoot = _RecordingShooting(k, p)
    lam0 = costates_from_direct(t_dir, its[-1][0], its[-1][1], p)
    v0 = shoot.guess_multipliers(lam0)
    shoot.evaluations.clear()
    sol = shoot.solve(v0)
    polish_t, polish_z, polish_u, polish_R = [], [], [], []
    for v, R in [(v0, np.nan)] + shoot.evaluations:
        s = _sample(k, np.concatenate([np.zeros(4), v[:4]]), p)
        if s is None:
            continue
        polish_t.append(s[0])
        polish_z.append(s[1])
        polish_u.append(s[2])
        polish_R.append(R)
    out["polish_t"], out["polish_z"] = polish_t[0], np.array(polish_z)
    out["polish_u"], out["polish_R"] = np.array(polish_u), np.array(polish_R)
    out["published_lam0"] = sol.lam0
    out["published_J"] = k.propagate(0, sol.z0, p[6], p, cost=True)["cost"]
    say(f"polish: {len(shoot.evaluations)} evaluations, J = {out['published_J']:.6f}")

    # 3. Random costate search.
    plain = SingleShooting(k, p)
    rng = np.random.default_rng(seed)
    tries_J, tries_z, tries_ok = [], [], []
    for _ in range(n_random):
        lam0 = rng.uniform(-60, 60, 4)
        try:
            s = plain.solve(plain.guess_multipliers(lam0))
        except (ShootingError, RuntimeError, ValueError, np.linalg.LinAlgError):
            s = None
        ok = s is not None and s.residual_norm < 1e-10
        J = k.propagate(0, s.z0, p[6], p, cost=True)["cost"] if ok else np.nan
        sample = _sample(k, s.z0, p) if ok else None
        tries_ok.append(ok and sample is not None)
        tries_J.append(J if tries_ok[-1] else np.nan)
        tries_z.append(sample[1] if tries_ok[-1] else np.full((N_SAMPLES, 4), np.nan))
    out["search_J"], out["search_ok"] = np.array(tries_J), np.array(tries_ok)
    out["search_z"] = np.array(tries_z)
    out["best_found"] = bool(out["search_ok"].any())
    if not out["best_found"]:
        # Too few starts to converge: show the published extremal in its place.
        say(f"search: none of {n_random} starts converged")
        out["best_J"] = out["published_J"]
        lam_best = sol.lam0
    else:
        best = int(np.nanargmin(out["search_J"]))
        out["best_J"] = out["search_J"][best]
        say(f"search: {int(out['search_ok'].sum())}/{n_random} converged, "
            f"best J = {out['best_J']:.6f} ({time.perf_counter() - start:.0f} s)")
        rng = np.random.default_rng(seed)
        for _ in range(best + 1):           # replay the draw that found the best one
            lam0 = rng.uniform(-60, 60, 4)
        lam_best = plain.solve(plain.guess_multipliers(lam0)).lam0

    # 4. Both extremals, finely sampled for the motion.
    for name, lam in (("published", sol.lam0), ("best", lam_best)):
        t, z, u = _sample(k, np.concatenate([np.zeros(4), lam]), p, n=241)
        out[f"{name}_t"], out[f"{name}_z"], out[f"{name}_u"] = t, z, u
    return out


def save(data, path):
    np.savez_compressed(path, **data)


def load(path):
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


# =============================================================================
# Rendering
# =============================================================================

POLE = 0.5                                     # pole length l (params[2])


def _poses(z, k=9):
    """Cart and pole-tip positions at k evenly spaced times of a trajectory z (n, 4)."""
    idx = np.linspace(0, len(z) - 1, k).round().astype(int)
    x, q = z[idx, 0], z[idx, 1]
    return x, np.stack([x + POLE * np.sin(q), -POLE * np.cos(q)], 1)


def _frame_plan(data, fps):
    plan = []
    n_dir = len(data["direct_J"])
    for i in np.unique(np.linspace(0, n_dir - 1, min(n_dir, 45)).round().astype(int)):
        plan += [("direct", int(i))] * 2
    plan += [("direct", n_dir - 1)] * fps
    n_pol = len(data["polish_R"])
    for i in range(n_pol):
        plan += [("polish", i)] * max(2, int(60 / n_pol))
    plan += [("polish", n_pol - 1)] * fps
    n_try = len(data["search_J"])
    per = max(1, int(np.ceil(n_try / 150)))
    for i in range(per - 1, n_try + per - 1, per):
        plan.append(("search", min(i, n_try - 1)))
    plan += [("search", n_try - 1)] * fps
    n_mot = len(data["best_t"])
    for _ in range(2):
        plan += [("motion", int(i)) for i in np.linspace(0, n_mot - 1, 2 * fps)]
        plan += [("motion", n_mot - 1)] * (fps // 2)
    return plan


CAPTIONS = {
    "direct": ("1  Direct method (Kelly 2017)",
               "Hermite-Simpson collocation from a straight-line guess. Early iterates are not "
               "dynamically feasible:\nthe defects measure how far each is from the equations of "
               "motion."),
    "polish": ("2  Indirect polish",
               "Costates lifted from the direct solution, then Newton on the shooting residual. "
               "Every evaluation is a\nreal extremal of the cart and pole; it misses the target "
               "until the residual vanishes."),
    "search": ("3  Random costate search",
               "Shooting from random costates. {n_ok} of {n} starts converge, {to_best} to an "
               "extremal that never\npushes the cart past x = 1 and costs {gain:.1f}% less "
               "than the published one."),
    "motion": ("4  The two extremals",
               "Published (orange) and the cheaper one found by search (blue), in real time "
               "(slowed 2x)."),
}


def animate(data, path, fps=30, dpi=100, stills_dir=None):
    """Render ``data`` from :func:`record` to ``path`` (.mp4 needs ffmpeg; .gif needs Pillow).

    With ``stills_dir``, also save the last frame of each stage as a PNG.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF})
    p = data["params"]
    T, d = p[6], p[5]
    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], left=0.05, right=0.97, top=0.80,
                          bottom=0.08, hspace=0.45, wspace=0.18)
    scene = fig.add_subplot(gs[:, 0])
    ax_q = fig.add_subplot(gs[0, 1])
    ax_x = fig.add_subplot(gs[1, 1], sharex=ax_q)
    ax_m = fig.add_subplot(gs[2, 1])
    title = fig.text(0.05, 0.94, "", fontsize=17, weight="bold")
    caption = fig.text(0.05, 0.85, "", fontsize=10.5, color=INK2, va="bottom")
    status = scene.text(0.02, 0.97, "", transform=scene.transAxes, va="top", fontsize=11)

    # The scene: track, target pose (dashed), and the moving or strobed cart-poles.
    scene.set_aspect("equal")
    scene.set(xlim=(-0.75, 1.75), ylim=(-0.75, 0.75), xticks=[0, 0.5, 1], yticks=[])
    scene.spines[["left"]].set_visible(False)
    scene.axhline(0, color=INK2, lw=1.2, zorder=1)
    scene.plot([d, d], [0, POLE], color=INK2, lw=2, ls="--", zorder=1)
    scene.plot([d - 0.12, d + 0.12], [0, 0], color=INK2, lw=10, alpha=0.15, zorder=1,
               solid_capstyle="butt")
    scene.text(d, POLE + 0.06, "target", ha="center", color=INK2)
    key = scene.text(0.02, 0.03, "poses at 9 evenly spaced times; darker = later",
                     transform=scene.transAxes, color=INK2, fontsize=9)
    scene.set_xlabel("cart position x [m]")
    strobe_poles = LineCollection([], linewidths=2.5, zorder=3)
    strobe_carts = LineCollection([], linewidths=12, zorder=2, capstyle="butt")
    scene.add_collection(strobe_carts)
    scene.add_collection(strobe_poles)
    movers = {}
    for name, color in (("published", ORANGE), ("best", BLUE)):
        cart, = scene.plot([], [], color=color, lw=16, solid_capstyle="butt", alpha=0.85,
                           zorder=4)
        pole, = scene.plot([], [], color=color, lw=3.5, marker="o", markevery=[1], ms=9,
                           zorder=5)
        movers[name] = (cart, pole)

    for ax, label, target in ((ax_q, "pole angle θ [rad]", np.pi), (ax_x, "cart x [m]", d)):
        ax.axhline(target, color=INK2, lw=1, ls="--")
        ax.set_ylabel(label)
        ax.grid(color=GRID, lw=0.8)
        ax.set_xlim(0, T)
    ax_x.set_xlabel("t [s]")
    ax_q.set_ylim(-1.5, 4.8)
    ax_x.set_ylim(-0.8, 1.8)
    ax_m.grid(color=GRID, lw=0.8)
    hist_q = LineCollection([], colors=MUTED, linewidths=1, alpha=0.5)
    hist_x = LineCollection([], colors=MUTED, linewidths=1, alpha=0.5)
    ax_q.add_collection(hist_q)
    ax_x.add_collection(hist_x)
    cur_q, = ax_q.plot([], [], color=BLUE, lw=2)
    cur_x, = ax_x.plot([], [], color=BLUE, lw=2)
    ref_q, = ax_q.plot([], [], color=ORANGE, lw=2)
    ref_x, = ax_x.plot([], [], color=ORANGE, lw=2)
    head = [ax.axvline(np.nan, color=INK2, lw=1) for ax in (ax_q, ax_x)]

    def strobe(z, color):
        xs, tips = _poses(z)
        alphas = np.linspace(0.15, 1.0, len(xs))
        rgba = [(*plt.matplotlib.colors.to_rgb(color), a) for a in alphas]
        strobe_poles.set_segments([[(x, 0), tuple(tp)] for x, tp in zip(xs, tips)])
        strobe_carts.set_segments([[(x - 0.12, 0), (x + 0.12, 0)] for x in xs])
        strobe_poles.set_colors(rgba)
        strobe_carts.set_colors([(*c[:3], c[3] * 0.6) for c in rgba])

    def clear_movers():
        for cart, pole in movers.values():
            cart.set_data([], [])
            pole.set_data([], [])

    state = {"stage": None}

    def setup(stage):
        """Reset the stage-dependent artists when a stage begins."""
        ax_m.cla()
        ax_m.grid(color=GRID, lw=0.8)
        for art in (cur_q, cur_x, ref_q, ref_x):
            art.set_data([], [])
        hist_q.set_segments([])
        hist_x.set_segments([])
        for h in head:
            h.set_xdata([np.nan])
        clear_movers()
        key.set_visible(stage != "motion")
        strobe_poles.set_segments([])
        strobe_carts.set_segments([])
        t_head, body = CAPTIONS[stage]
        if stage == "search":
            ok = data["search_ok"]
            J = data["search_J"][ok]
            n_best = int(np.sum(np.abs(J - data["best_J"]) < 1e-6))
            if len(J):
                body = body.format(n_ok=int(ok.sum()), n=len(ok),
                                   to_best="all" if n_best == len(J) else f"{n_best} of them",
                                   gain=100 * (1 - data["best_J"] / data["published_J"]))
            else:
                body = (f"Shooting from random costates. None of {len(ok)} starts converged "
                        "(use more starts).")
        title.set_text(t_head)
        caption.set_text(body)
        if stage == "direct":
            n = len(data["direct_defect"])
            ax_m.set(yscale="log", xlim=(-1, n), xlabel="SLSQP iteration",
                     ylabel="‖defects‖")
            ax_m.set_ylim(max(np.nanmin(data["direct_defect"]) * 0.3, 1e-14),
                          np.nanmax(data["direct_defect"]) * 3)
            state["m"], = ax_m.plot([], [], color=BLUE, lw=2, marker="o", ms=3)
        elif stage == "polish":
            R = data["polish_R"]
            n = len(R)
            ax_m.set(yscale="log", xlim=(-0.5, n - 0.5), xlabel="residual evaluation",
                     ylabel="‖shooting residual‖")
            finite = R[np.isfinite(R) & (R > 0)]
            ax_m.set_ylim(max(finite.min() * 0.3, 1e-16), finite.max() * 3)
            state["m"], = ax_m.plot([], [], color=BLUE, lw=2, marker="o", ms=4)
        elif stage == "search":
            J = data["search_J"][data["search_ok"]]
            if not len(J):
                J = np.array([data["published_J"]])
            lo = min(np.nanmin(J), data["published_J"])
            hi = max(np.nanpercentile(J, 95), data["published_J"])
            pad = 0.25 * (hi - lo) + 0.1
            ax_m.set(xlim=(0, len(data["search_J"])), ylim=(lo - pad, hi + pad),
                     xlabel="random start", ylabel="cost J of the extremal")
            ax_m.axhline(data["published_J"], color=ORANGE, lw=1.5, ls="--")
            ax_m.text(0.99, (data["published_J"] - lo + pad) / (hi - lo + 2 * pad) + 0.04,
                      f"published J = {data['published_J']:.3f}", transform=ax_m.transAxes,
                      ha="right", color=ORANGE)
            state["m"] = ax_m.scatter([], [], s=12, color=BLUE)
            state["best"] = ax_m.text(0.01, 0.06, "", transform=ax_m.transAxes, color=BLUE)
        else:
            ax_m.set(xlim=(0, T), xlabel="t [s]", ylabel="force u [N]")
            ax_m.axhline(0, color=INK2, lw=0.8)
            ax_m.plot(data["published_t"], data["published_u"], color=ORANGE, lw=2)
            ax_m.plot(data["best_t"], data["best_u"], color=BLUE, lw=2)
            state["mhead"] = ax_m.axvline(0, color=INK2, lw=1)
            for z, lq, lx, tt in ((data["published_z"], ref_q, ref_x, data["published_t"]),
                                  (data["best_z"], cur_q, cur_x, data["best_t"])):
                lq.set_data(tt, z[:, 1])
                lx.set_data(tt, z[:, 0])
        state["stage"] = stage

    def draw(frame):
        stage, i = frame
        if stage == "__reset__":
            state["stage"] = None
            return ()
        if stage != state["stage"]:
            setup(stage)
        if stage == "direct":
            t = data["direct_t"]
            X = data["direct_X"]
            lo = max(0, i - 6)
            hist_q.set_segments([np.column_stack([t, X[j, :, 1]]) for j in range(lo, i)])
            hist_x.set_segments([np.column_stack([t, X[j, :, 0]]) for j in range(lo, i)])
            cur_q.set_data(t, X[i, :, 1])
            cur_x.set_data(t, X[i, :, 0])
            strobe(X[i], BLUE)
            dfc = data["direct_defect"][: i + 1]
            state["m"].set_data(np.arange(i + 1), dfc)
            status.set_text(f"iteration {i}   J = {data['direct_J'][i]:.3f}   "
                            f"‖defects‖ = {dfc[-1]:.1e}")
        elif stage == "polish":
            t, Z = data["polish_t"], data["polish_z"]
            hist_q.set_segments([np.column_stack([t, Z[j, :, 1]]) for j in range(i)])
            hist_x.set_segments([np.column_stack([t, Z[j, :, 0]]) for j in range(i)])
            cur_q.set_data(t, Z[i, :, 1])
            cur_x.set_data(t, Z[i, :, 0])
            strobe(Z[i], BLUE)
            R = data["polish_R"][: i + 1]
            state["m"].set_data(np.arange(i + 1), R)
            miss = np.abs(Z[i, -1] - [d, np.pi, 0, 0]).max()
            r_txt = "initial guess" if not np.isfinite(R[-1]) else f"‖R‖ = {R[-1]:.1e}"
            status.set_text(f"evaluation {i}   {r_txt}   end-state miss = {miss:.1e}")
        elif stage == "search":
            ok, J, Z = data["search_ok"], data["search_J"], data["search_z"]
            idx = np.flatnonzero(ok[: i + 1])
            t = data["polish_t"]
            shown = idx[-40:]
            hist_q.set_segments([np.column_stack([t, Z[j, :, 1]]) for j in shown])
            hist_x.set_segments([np.column_stack([t, Z[j, :, 0]]) for j in shown])
            if len(idx):
                j = idx[-1]
                cur_q.set_data(t, Z[j, :, 1])
                cur_x.set_data(t, Z[j, :, 0])
                strobe(Z[j], BLUE)
                state["m"].set_offsets(np.column_stack([idx, J[idx]]))
                best = np.nanmin(J[idx])
                state["best"].set_text(f"best so far J = {best:.3f}")
            status.set_text(f"start {i + 1} / {len(J)}   converged: {len(idx)}")
        else:
            strobe_poles.set_segments([])
            strobe_carts.set_segments([])
            for name in ("published", "best"):
                z = data[f"{name}_z"][i]
                cart, pole = movers[name]
                cart.set_data([z[0] - 0.12, z[0] + 0.12], [0, 0])
                pole.set_data([z[0], z[0] + POLE * np.sin(z[1])], [0, -POLE * np.cos(z[1])])
            tt = data["best_t"][i]
            for h in head:
                h.set_xdata([tt])
            state["mhead"].set_xdata([tt])
            status.set_text(f"t = {tt:.2f} s   published J = {data['published_J']:.3f}   "
                            f"found J = {data['best_J']:.3f}")
        return ()

    plan = _frame_plan(data, fps)
    # The motion's still is mid-swing, not at the target where both coincide.
    mid = next(j for j, f in enumerate(plan) if f[0] == "motion" and f[1] >= len(data["best_t"]) // 2)
    return write(fig, draw, plan, path, fps, stills_dir, prefix="cart_pole_",
                 still_frames={"motion": mid}, problem=PROBLEM)
