"""Record and animate Goddard's rocket: a singular arc, then a dynamic-pressure limit.

A rocket climbs vertically from rest through an exponential atmosphere and
burns 40% of its mass as propellant; the thrust is limited to ``u <= 3.5``.
Maximize the final altitude. Full thrust wastes propellant on drag low down,
so the optimum throttles back on a singular arc, where the switching
function stays at zero, before coasting to the top.

1. **Singular arc.** Exact multi-arc shooting on full thrust, singular,
   coast, from a rough guess: every residual evaluation is drawn.
2. **Dynamic-pressure limit** ``q <= 10``. The free extremal peaks at
   ``q = 14.9``; its crossings of the limit seed the structure full thrust,
   ride ``q = 10``, singular, coast.
3. **The climb**, with and without the limit.
"""

import numpy as np

from problems import GODDARD_REFERENCE, goddard_free, goddard_kernels, goddard_q
from visuals.common import (BLUE, GRID, INK2, MUTED, ORANGE, Problem, header, load,
                            recording, save, style, write)

from auto_optimal_control import examples

__all__ = ["record", "animate", "save", "load", "PROBLEM"]
N_T = 401
P = examples.GODDARD_Q_PARAMS
RHO0, QMAX, BETA, UMAX = P[5], P[6], P[1], P[3]

PROBLEM = Problem(
    name="Goddard's rocket: the highest climb through drag",
    story=("A rocket climbs from rest through an exponential atmosphere, burning 40% of its "
           "mass. Thrust is limited to 3.5 (normalized units).\nFull thrust wastes propellant "
           "fighting drag low down: the optimum throttles back on a singular arc, where the "
           "switching function\nstays at zero, then coasts. Then a structural limit on the "
           "dynamic pressure q ≤ 10 makes it ride the limit before the singular arc."),
    formulation=[
        ("maximize", r"$h(t_f)$,   $t_f$ free"),
        ("dynamics", r"$\dot h = v, \quad \dot v = \frac{u - D}{m} - \frac{1}{h^2}, \quad "
                     r"\dot m = -\frac{u}{c}, \quad D = 310\, v^2 e^{-500 (h - 1)}$"),
        ("thrust", r"$0 \leq u \leq 3.5$   ($H$ linear in $u$: bang-bang or singular, "
                   r"$S'' = 0$ on the singular arc)"),
        ("pressure", r"$q = \frac{1}{2}\rho_0 v^2 e^{-500 (h - 1)} \leq 10$   (order 1; "
                     r"second solve)"),
        ("boundary", r"$h(0) = 1, \ v(0) = 0, \ m(0) = 1; \quad m(t_f) = 0.6$"),
    ],
    solved_by=[
        "Exact multi-arc shooting on full thrust → singular → coast, from a rough guess: "
        "S = S' = 0 at the singular entry.",
        "Add q ≤ 10: the free extremal's crossings of the limit seed full thrust → ride q = 10 "
        "→ singular → coast.",
    ],
    summary="max h(t_f),  Goddard rocket,  0 ≤ u ≤ 3.5,  m: 1 → 0.6,  then q ≤ 10",
    source="Seywald & Cliff (1992); Graichen & Petit (2008): h(t_f) = 1.01283692 free, "
           "1.01271727 with q ≤ 10",
)


def _q(h, v):
    return RHO0 * v ** 2 * np.exp(-BETA * (h - 1)) / 2


def _sample(shoot, v, tt):
    """h, v, m, u, q on ``tt * t_f``, and t_f."""
    tr = shoot.trajectory(v, n_points=1500)
    _, idx = np.unique(tr["t"], return_index=True)
    T, Z, U = tr["t"][idx], tr["z"][idx], tr["u"][idx, 0]
    tf = T[-1]
    s = tt * tf
    h, vel, m = (np.interp(s, T, Z[:, i]) for i in range(3))
    return np.column_stack([h, vel, m, np.interp(s, T, U), _q(h, vel)]), tf


def _iterates(shoot, seed, evaluations, tt):
    X, TF, R = [], [], []
    for w, r in [(seed, float(np.linalg.norm(shoot.residual(seed))))] + evaluations:
        try:
            x, tf = _sample(shoot, w, tt)
        except Exception:                    # times out of order, failed propagation
            continue
        if not np.isfinite(x).all():
            continue
        X.append(x)
        TF.append(tf)
        R.append(r)
    return np.array(X), np.array(TF), np.array(R)


def record(cache_dir=None, verbose=True):
    k, kq = goddard_kernels(cache_dir)
    tt = np.linspace(0, 1, N_T)
    out = {}
    free, fsol = goddard_free(k, record=recording)
    seed = free.pack_arcs(free.pack([-2.5, -0.18, 0.005], [-0.03], 0.2), [(0.03, 0.1)])
    X, TF, R = _iterates(free, seed, free.evaluations, tt)
    out.update(free_x=X, free_tf=TF, free_R=R,
               free_junctions=np.array(free.unpack_arcs(fsol.v)[0]))
    qshoot, qsol, qseed = goddard_q(kq, free, fsol, record=recording)
    X, TF, R = _iterates(qshoot, qseed, qshoot.evaluations, tt)
    (b_in, b_out, _), s_out = qshoot.unpack_arcs(qsol.v)
    out.update(q_x=X, q_tf=TF, q_R=R, q_junctions=np.array([b_in, b_out, s_out]))
    # The climb on one clock, to the later arrival.
    T = max(fsol.tf, qsol.tf)
    clock = np.linspace(0, T, 600)
    for key, shoot, sol in (("run_free", free, fsol), ("run_q", qshoot, qsol)):
        x, tf = _sample(shoot, sol.v, clock[clock <= sol.tf] / sol.tf)
        pad = np.repeat(x[-1:], len(clock) - len(x), axis=0)
        pad[:, 3] = 0.0
        out[key] = np.vstack([x, pad])
        out[key + "_tf"] = tf
    out["clock"] = clock
    if verbose:
        print(f"free: {len(free.evaluations)} evaluations, h(t_f) = {out['free_x'][-1][-1, 0]:.10f}"
              f" ({GODDARD_REFERENCE['free'][0]});  q <= 10: {len(qshoot.evaluations)} "
              f"evaluations, h(t_f) = {out['q_x'][-1][-1, 0]:.10f} "
              f"({GODDARD_REFERENCE['q'][0]})", flush=True)
    return out


CAPTIONS = {
    "free": ("1  The singular arc",
             "Exact multi-arc shooting on full thrust → singular → coast, from a rough guess "
             "(every residual evaluation drawn). On the singular arc\nS = S' = 0 and the thrust "
             "is u_s from S'' = 0; the free extremal's dynamic pressure peaks at q = {qpk:.1f}, "
             "far above 10."),
    "q": ("2  With the dynamic-pressure limit q ≤ 10",
          "Seeded from the free solution, where its q crosses 10: full thrust, ride q = 10 "
          "(boundary arc of order 1), a joined singular arc,\nthen the coast. h(t_f) = {hq:.8f}, "
          "{dh:.1e} below the free climb."),
    "motion": ("3  The climb",
               "Blue: with q ≤ 10. Orange: without it. Flame length is the thrust; altitude in "
               "thousandths of an Earth radius."),
}


def _phases(junctions, tf, limited):
    if limited:
        b_in, b_out, s_out = junctions
        return [(0, b_in, "full"), (b_in, b_out, "ride q = 10"), (b_out, s_out, "sing."),
                (s_out, tf, "coast")]
    t_in, t_out = junctions
    return [(0, t_in, "full"), (t_in, t_out, "singular"), (t_out, tf, "coast")]


def _plan(data, fps):
    def steps(n, frames):
        idx = np.unique(np.linspace(0, n - 1, min(n, 60)).round().astype(int))
        return [int(i) for i in idx for _ in range(max(3, frames // len(idx)))]

    plan = []
    for stage, secs in (("free", 5), ("q", 5)):
        n = len(data[f"{stage}_R"])
        plan += [(stage, i) for i in steps(n, secs * fps)] + [(stage, n - 1)] * (2 * fps)
    n = len(data["clock"])
    plan += [("motion", int(i)) for i in np.linspace(0, n - 1, 9 * fps)]
    plan += [("motion", n - 1)] * (2 * fps)
    return plan


def _rocket(ax, plt, x, color):
    body = plt.Polygon([[x - 0.09, 0], [x + 0.09, 0], [x + 0.09, 0.5], [x, 0.72],
                        [x - 0.09, 0.5]], closed=True, color=color)
    flame = plt.Polygon([[x - 0.06, 0], [x + 0.06, 0], [x, -0.1]], closed=True, color="#f2a541")
    ax.add_patch(body)
    ax.add_patch(flame)
    return body, flame


def animate(data, path, fps=30, dpi=100, stills_dir=None):
    plt = style()
    from matplotlib.collections import LineCollection

    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], left=0.07, right=0.97, top=0.80,
                          bottom=0.105, hspace=0.45, wspace=0.2)
    ax_h = fig.add_subplot(gs[:, 0])
    ax_u = fig.add_subplot(gs[0, 1])
    ax_q = fig.add_subplot(gs[1, 1])
    ax_k = fig.add_subplot(gs[2, 1])
    title, caption = header(fig)
    status = ax_h.text(0.02, 0.98, "", transform=ax_h.transAxes, va="top", fontsize=11)
    keep = {status}
    state = {"stage": None}
    tmax = 0.215
    hscale = 1e3                              # altitude in thousandths of an Earth radius
    hmax = (data["free_x"][-1][:, 0].max() - 1) * hscale * 1.08
    qtop = 1.15 * max(data["free_x"][-1][:, 4].max(), QMAX)

    def clear():
        for ax in (ax_h, ax_u, ax_q, ax_k):
            for a in list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.texts):
                if a not in keep:
                    a.remove()
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.set_visible(True)

    def setup(stage):
        clear()
        for ax in (ax_h, ax_u, ax_q, ax_k):
            ax.grid(color=GRID, lw=0.8)
        ax_u.set(xlim=(0, tmax), ylim=(-0.3, UMAX * 1.2), ylabel="thrust u")
        ax_u.axhline(UMAX, color=INK2, lw=1, ls="--")
        ax_u.axhline(0, color=INK2, lw=0.8)
        ax_q.set(xlim=(0, tmax), ylim=(0, qtop), ylabel="dynamic pressure q", xlabel="t")
        ax_q.axhline(QMAX, color=ORANGE if stage != "free" else MUTED, lw=1.5, ls="--")
        ax_q.text(tmax * 0.98, QMAX + 1.2, "q_max = 10", ha="right", color=INK2, fontsize=9)
        state["hist"] = {}
        state["cur"] = {}
        if stage == "motion":
            ax_h.set(xlim=(-0.6, 1.6), ylim=(-0.15 * hmax, hmax * 1.12), xticks=[],
                     xlabel="", ylabel="altitude h − 1  (×10⁻³)")
            ax_h.grid(False)
            ax_h.axhline(0, color=INK2, lw=1.5)
            state["rockets"] = [_rocket(ax_h, plt, x, c) for x, c in ((0.0, BLUE), (1.0, ORANGE))]
            state["labels"] = [ax_h.text(x, -0.1 * hmax, name, ha="center", color=c, fontsize=11)
                               for x, c, name in ((0.0, BLUE, "q ≤ 10"), (1.0, ORANGE, "no limit"))]
            state["gauges"] = [ax_h.text(x + 0.18, 0, "", fontsize=9.5, color=c, va="center")
                               for x, c in ((0.0, BLUE), (1.0, ORANGE))]
            for key, c in (("run_q", BLUE), ("run_free", ORANGE)):
                h_top = (data[key][:, 0].max() - 1) * hscale
                ax_h.axhline(h_top, color=c, lw=0.8, ls=":")
            clock = data["clock"]
            for key, c in (("run_q", BLUE), ("run_free", ORANGE)):
                ax_u.plot(clock, data[key][:, 3], color=c, lw=2)
                ax_q.plot(clock, data[key][:, 4], color=c, lw=2)
            state["heads"] = [ax.axvline(0, color=INK2, lw=1) for ax in (ax_u, ax_q)]
            ax_k.set_visible(False)
        else:
            ax_h.set(xlim=(0, tmax), ylim=(0, hmax), xlabel="t",
                     ylabel="altitude h − 1  (×10⁻³)")
            for ax in (ax_h, ax_u, ax_q):
                c = LineCollection([], colors=MUTED, linewidths=1, alpha=0.55)
                ax.add_collection(c)
                state["hist"][ax] = c
                state["cur"][ax] = ax.plot([], [], color=BLUE, lw=2.2)[0]
            if stage == "q":
                ref = data["free_x"][-1]
                grid = np.linspace(0, 1, ref.shape[0]) * data["free_tf"][-1]
                for ax, col in ((ax_h, 0), (ax_u, 3), (ax_q, 4)):
                    y = (ref[:, 0] - 1) * hscale if col == 0 else ref[:, col]
                    ax.plot(grid, y, color=ORANGE, lw=1.3, alpha=0.8)
            R = data[f"{stage}_R"]
            fin = R[np.isfinite(R) & (R > 0)]
            ax_k.set(yscale="log", xlim=(-0.5, len(R) - 0.5),
                     ylim=(max(fin.min() * 0.3, 1e-16), fin.max() * 3),
                     xlabel="residual evaluation", ylabel="‖multi-arc residual‖")
            state["k"], = ax_k.plot([], [], color=BLUE, lw=2, marker="o", ms=4)
            state["phases"] = []
        head, body = CAPTIONS[stage]
        title.set_text(head)
        hq = data["q_x"][-1][-1, 0]
        caption.set_text(body.format(qpk=data["free_x"][-1][:, 4].max(), hq=hq,
                                     dh=data["free_x"][-1][-1, 0] - hq))
        state["stage"] = stage

    def show(stage, i):
        X, TF = data[f"{stage}_x"], data[f"{stage}_tf"]
        grid = np.linspace(0, 1, X.shape[1])
        prev = range(max(0, i - 10), i)
        for ax, col in ((ax_h, 0), (ax_u, 3), (ax_q, 4)):
            def y(x):
                return (x[:, 0] - 1) * hscale if col == 0 else x[:, col]
            state["hist"][ax].set_segments([np.column_stack([grid * TF[j], y(X[j])])
                                            for j in prev])
            state["cur"][ax].set_data(grid * TF[i], y(X[i]))

    def draw(frame):
        stage, i = frame
        if stage == "__reset__":
            state["stage"] = None
            return ()
        if stage != state["stage"]:
            setup(stage)
        if stage in ("free", "q"):
            show(stage, i)
            R = data[f"{stage}_R"]
            state["k"].set_data(np.arange(i + 1), R[:i + 1])
            for a in state["phases"]:
                a.remove()
            state["phases"] = []
            last = i == len(R) - 1
            if last:
                tf = data[f"{stage}_tf"][-1]
                for j, (lo, hi, name) in enumerate(_phases(data[f"{stage}_junctions"], tf,
                                                           stage == "q")):
                    state["phases"].append(ax_u.axvspan(lo, hi, color=BLUE,
                                                        alpha=0.05 + 0.05 * (j % 2), lw=0))
                    state["phases"].append(ax_u.text((lo + hi) / 2, UMAX * 1.07, name,
                                                     ha="center", fontsize=8.5, color=INK2))
            x = data[f"{stage}_x"][i]
            done = (f"\nh(t_f) = {x[-1, 0]:.10f}   (Graichen & Petit: "
                    f"{GODDARD_REFERENCE[stage][0]})") if last else ""
            status.set_text(f"evaluation {i}   ‖R‖ = {R[i]:.1e}{done}")
        else:
            clock = data["clock"]
            for (body, flame), gauge, key, x0 in zip(state["rockets"], state["gauges"],
                                                     ("run_q", "run_free"), (0.0, 1.0)):
                h, _, _, u, q = data[key][i]
                y = (h - 1) * hscale
                size = 0.09 * hmax
                body.set_xy([[x0 - 0.09, y], [x0 + 0.09, y], [x0 + 0.09, y + size],
                             [x0, y + 1.45 * size], [x0 - 0.09, y + size]])
                fl = 0.45 * size * u / UMAX
                flame.set_xy([[x0 - 0.06, y], [x0 + 0.06, y], [x0, y - fl]])
                gauge.set_position((x0 + 0.18, y + 0.6 * size))
                gauge.set_text(f"u = {u:.2f}\nq = {q:.1f}")
            for hd in state["heads"]:
                hd.set_xdata([clock[i]])
            status.set_text(f"t = {clock[i]:.4f}")
        return ()

    plan = _plan(data, fps)
    mid = next(j for j, f in enumerate(plan)
               if f[0] == "motion" and f[1] >= 0.3 * len(data["clock"]))
    return write(fig, draw, plan, path, fps, stills_dir, prefix="goddard_",
                 still_frames={"motion": mid}, problem=PROBLEM)
