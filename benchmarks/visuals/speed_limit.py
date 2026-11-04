"""Record and animate the minimum-time run with a speed limit: exact arcs with a
bang-bang control.

A train leaves a station at rest and must stop at the next one, ``D = 3``
away, as soon as possible, with ``|u| <= 1`` and ``v <= V = 1``. The answer
is full power to the limit, a cruise at ``v = V`` (``u = 0``, a boundary arc
of an order-1 limit) and full braking: ``t_f = D/V + V/u_max = 4``. Penalties
do not work here: the penalized optimum would ride the limit on a singular
arc. So the problem is smoothed first.

1. **Smoothed, with a penalty.** The running cost ``1 + eps u^2/2`` at
   ``eps = 1`` makes the control a clipped-linear law; ``rho`` is raised on
   ``rho/2 max(0, v - V)^2``.
2. **Smoothing continuation with exact arcs.** The penalty solution seeds
   multi-arc shooting with one boundary arc, and ``eps`` is lowered from 1 to
   ``1e-4``: the control sharpens into bang-bang and ``t_f`` falls to 4.
3. **Bang-bang exact arcs.** The last smoothed solution seeds the bang-bang
   problem; its switches sit on the junctions, and ``t_f = 4`` to round-off.
4. **The run**, beside the run without a speed limit.
"""

import numpy as np

from problems import speed_limit_record
from visuals.common import (BLUE, GRID, INK2, MUTED, ORANGE, Problem, header, load,
                            recording, save, style, write)

__all__ = ["record", "animate", "save", "load", "PROBLEM"]
V, D, UMAX = 1.0, 3.0, 1.0
N_T = 401

PROBLEM = Problem(
    name="Train between stations: minimum time under a speed limit",
    story=("A train leaves a station at rest and must stop at the next one, D = 3 away, as soon "
           "as possible. It can accelerate and brake\nat most |u| ≤ 1, and its speed is "
           "limited to V = 1 (units scaled to the limits). The optimum is bang-bang with a\n"
           "cruise on the limit: full power, coast at v = V, full brakes, arriving at "
           "t_f = D/V + V/u_max = 4."),
    formulation=[
        ("minimize", r"$t_f = \int_0^{t_f} 1\,dt$"),
        ("dynamics", r"$\dot x = v, \quad \dot v = u$"),
        ("speed limit", r"$v(t) \leq V = 1$   (order 1)"),
        ("control", r"$|u(t)| \leq u_{max} = 1$   ($H$ is linear in $u$: bang-bang on "
                    r"$S = \lambda_v$)"),
        ("boundary", r"$x(0) = 0, \ v(0) = 0; \quad x(t_f) = 3, \ v(t_f) = 0$"),
    ],
    solved_by=[
        "Smooth it: running cost 1 + ε u²/2 with ε = 1 (a clipped-linear control) and an "
        "exterior penalty on v ≤ V, raised by continuation.",
        "Multi-arc shooting on the smoothed problem, seeded from the penalty solution; then ε "
        "lowered to 1e-4 with exact arcs.",
        "Bang-bang multi-arc shooting from the last smoothed solution: switches located, the "
        "cruise exact, t_f = 4 to round-off.",
    ],
    summary="min t_f,  ẍ = u,  |u| ≤ 1,  v ≤ 1,  x: 0 → 3, rest to rest",
    source="Closed form: t_f = D/V + V/u_max while D > V²/u_max",
)


def _no_limit(t):
    """Bang-bang without the speed limit (closed form): switch at sqrt(D/u_max)."""
    ts = np.sqrt(D / UMAX)
    x = np.where(t <= ts, UMAX * t ** 2 / 2, D - UMAX * np.clip(2 * ts - t, 0, None) ** 2 / 2)
    v = np.where(t <= ts, UMAX * t, UMAX * np.clip(2 * ts - t, 0, None))
    u = np.where(t <= ts, UMAX, np.where(t <= 2 * ts, -UMAX, 0.0))
    return x, v, u, 2 * ts


def _sample(shoot, v, tt):
    tr = shoot.trajectory(v, n_points=1200)
    _, idx = np.unique(tr["t"], return_index=True)
    T, Z, U = tr["t"][idx], tr["z"][idx], tr["u"][idx, 0]
    tf = T[-1]
    s = tt * tf                          # on [0, t_f], resampled on a fixed grid of N_T points
    return (np.column_stack([np.interp(s, T, Z[:, i]) for i in range(4)]),
            np.interp(s, T, U), tf, s)


def record(cache_dir=None, verbose=True):
    bang, sol, parts, ks = speed_limit_record(V, D, UMAX, cache_dir, record=recording)
    tt = np.linspace(0, 1, N_T)
    out = {}
    # 1. Smoothed penalty continuation (eps = 1).
    Z, U, TF, viol = [], [], [], []
    for rho, s in parts["penalty"]:
        tr = ks.trajectory(0, s.z0, s.tf, [V, D, UMAX, 1.0, rho], t_eval=tt * s.tf)
        Z.append(tr["z"])
        U.append(tr["u"][:, 0])
        TF.append(s.tf)
        viol.append(s.max_violation)
    out.update(rho=np.array([r for r, _ in parts["penalty"]]), pen_z=np.array(Z),
               pen_u=np.array(U), pen_tf=np.array(TF), pen_viol=np.array(viol))
    # 2. Smoothing continuation with exact arcs.
    from problems import MultiArcShooting

    Z, U, TF, EPS = [], [], [], []
    for eps, s in parts["eps"]:
        shoot = MultiArcShooting(ks, [V, D, UMAX, eps, 0.0], [("boundary", "v_max")])
        z, u, tf, _ = _sample(shoot, s.v, tt)
        Z.append(z)
        U.append(u)
        TF.append(tf)
        EPS.append(eps)
    out.update(eps=np.array(EPS), eps_z=np.array(Z), eps_u=np.array(U), eps_tf=np.array(TF))
    # 3. Bang-bang exact arcs: every residual evaluation.
    Z, U, TF, R = [], [], [], []
    seed = parts["eps"][-1][1].v
    for w, r in [(seed, float(np.linalg.norm(bang.residual(seed))))] + bang.evaluations:
        try:
            z, u, tf, _ = _sample(bang, w, tt)
        except Exception:                    # times out of order, failed propagation
            continue
        Z.append(z)
        U.append(u)
        TF.append(tf)
        R.append(r)
    out.update(bang_z=np.array(Z), bang_u=np.array(U), bang_tf=np.array(TF),
               bang_R=np.array(R))
    out["junctions"] = np.array(bang.unpack_arcs(sol.v)[0][:2])
    out["tf"] = sol.tf
    # The run without a speed limit, on the same clock (it arrives first).
    tf = sol.tf
    clock = tt * tf
    out["clock"] = clock
    xr, vr, ur, tfr = _no_limit(clock)
    out.update(ref_x=xr, ref_v=vr, ref_u=ur, ref_tf=tfr)
    if verbose:
        print(f"{len(parts['penalty'])} penalty solves, {len(EPS)} smoothing steps to "
              f"eps = {EPS[-1]:.0e}, bang-bang in {len(bang.evaluations)} evaluations: "
              f"t_f = {sol.tf:.15f} (closed form {D / V + V / UMAX})", flush=True)
    return out


CAPTIONS = {
    "pen": ("1  Smoothed problem, penalty continuation",
            "The running cost 1 + ε u²/2 (ε = 1) makes the control a clipped-linear law. "
            "ρ/2 max(0, v − V)² is raised from 1 to 30:\nthe speed is pushed down toward the "
            "limit (shaded: over it)."),
    "eps": ("2  Smoothing continuation with exact arcs",
            "The penalty solution seeds multi-arc shooting with one boundary arc, and ε is "
            "lowered from 1 to 1e-4. The ramps of u\nshorten into jumps (bang-bang) and t_f "
            "falls to 4; t_f − 4 shrinks in proportion to ε."),
    "bang": ("3  Bang-bang exact arcs",
             "The last smoothed solution seeds the bang-bang problem (H linear in u). The "
             "switching function S = λ_v vanishes at both\njunctions: full power until t = 1, "
             "cruise at v = V with u = 0, brakes from t = 3. t_f = 4 to round-off."),
    "motion": ("4  The run",
               "Blue: with the speed limit, t_f = {tf:.4f}. Orange: without it, peaking at "
               "v = {vmax:.2f} and arriving at t_f = {tfr:.4f}. Real time, slowed 2x."),
}


def _plan(data, fps):
    def steps(n, most, frames):
        idx = np.unique(np.linspace(0, n - 1, min(n, most)).round().astype(int))
        return [int(i) for i in idx for _ in range(max(2, frames // len(idx)))]

    plan = []
    for stage, key, secs in (("pen", "rho", 3), ("eps", "eps", 5), ("bang", "bang_R", 3)):
        n = len(data[key])
        plan += [(stage, i) for i in steps(n, 40, secs * fps)] + [(stage, n - 1)] * fps
    n = len(data["clock"])
    plan += [("motion", int(i)) for i in np.linspace(0, n - 1, int(2 * data["tf"] * fps))]
    plan += [("motion", n - 1)] * fps
    return plan


def animate(data, path, fps=30, dpi=100, stills_dir=None):
    plt = style()
    from matplotlib.collections import LineCollection

    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], left=0.06, right=0.97, top=0.80,
                          bottom=0.105, hspace=0.45, wspace=0.18)
    ax_v = fig.add_subplot(gs[:, 0])
    ax_u = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1])
    ax_k = fig.add_subplot(gs[2, 1])
    title, caption = header(fig)
    status = ax_v.text(0.02, 0.98, "", transform=ax_v.transAxes, va="top", fontsize=11)
    keep = {status}
    state = {"stage": None}
    tmax = 4.6

    def clear():
        for ax in (ax_v, ax_u, ax_s, ax_k):
            for a in list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.texts):
                if a not in keep:
                    a.remove()
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.set_visible(True)

    def setup(stage):
        clear()
        state["hist"] = {}
        for ax in (ax_v, ax_u, ax_s):
            c = LineCollection([], colors=MUTED, linewidths=1, alpha=0.55)
            ax.add_collection(c)
            state["hist"][ax] = c
        state["cur"] = {ax: ax.plot([], [], color=BLUE, lw=2.2)[0] for ax in (ax_v, ax_u, ax_s)}
        for ax in (ax_v, ax_u, ax_s, ax_k):
            ax.grid(color=GRID, lw=0.8)
        ax_u.set(xlim=(0, tmax), ylim=(-1.35, 1.35), ylabel="control u")
        for s in (1, -1):
            ax_u.axhline(s * UMAX, color=ORANGE, lw=1.2, ls="--")
        if stage == "motion":
            ax_v.set(xlim=(-0.35, D + 0.35), ylim=(-1.0, 1.4), yticks=[], xlabel="position x",
                     ylabel="")
            ax_v.grid(False)
            for y, c, name in ((0.55, BLUE, f"speed limit v ≤ {V:g}"),
                               (-0.45, ORANGE, "no speed limit")):
                ax_v.plot([-0.3, D + 0.3], [y - 0.13, y - 0.13], color=INK2, lw=1.5)
                for xs in (0, D):
                    ax_v.add_patch(plt.Rectangle((xs - 0.12, y - 0.13), 0.24, 0.32, fill=False,
                                                 ec=INK2, lw=1))
                ax_v.text(-0.3, y + 0.3, name, color=c, fontsize=11)
            ax_v.text(0, 1.2, "station A", ha="center", color=INK2)
            ax_v.text(D, 1.2, "station B", ha="center", color=INK2)
            state["trains"] = [ax_v.add_patch(plt.Rectangle((-0.2, y - 0.1), 0.4, 0.16, color=c))
                               for y, c in ((0.55, BLUE), (-0.45, ORANGE))]
            state["gauges"] = [ax_v.text(D + 0.3, y + 0.3, "", ha="right", fontsize=10, color=c)
                               for y, c in ((0.55, BLUE), (-0.45, ORANGE))]
            ax_s.set(xlim=(0, tmax), ylim=(0, 1.9), ylabel="speed v", xlabel="t")
            ax_s.axhline(V, color=INK2, lw=1, ls="--")
            ax_k.set_visible(False)
            clock = data["clock"]
            ax_u.plot(clock, data["bang_u"][-1], color=BLUE, lw=2, drawstyle="steps-post")
            ax_u.plot(clock, data["ref_u"], color=ORANGE, lw=2, drawstyle="steps-post")
            ax_s.plot(clock, data["bang_z"][-1][:, 1], color=BLUE, lw=2)
            ax_s.plot(clock, data["ref_v"], color=ORANGE, lw=2)
            state["heads"] = [ax.axvline(0, color=INK2, lw=1) for ax in (ax_u, ax_s)]
        else:
            ax_v.set(xlim=(0, tmax), ylim=(0, 1.5), xlabel="t", ylabel="speed v")
            ax_v.axhline(V, color=ORANGE, lw=2, ls="--")
            ax_v.text(0.05, V + 0.02, f"speed limit V = {V:g}", color=ORANGE)
            ax_s.set(xlim=(0, tmax), ylabel="switching fn S = λ_v", xlabel="t")
            ax_s.axhline(0, color=INK2, lw=0.8)
            lv = np.concatenate([data[f"{k}_z"][..., 3].ravel() for k in ("pen", "eps", "bang")])
            lo, hi = np.percentile(lv, [1, 99])
            ax_s.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
            if stage == "pen":
                rho = data["rho"]
                ax_k.set(xscale="log", xlim=(rho[0] / 1.5, rho[-1] * 1.5),
                         xlabel="penalty weight ρ  (ε = 1)", ylabel="arrival time t_f")
                ax_k.set_ylim(data["pen_tf"].min() - 0.05, data["pen_tf"].max() + 0.05)
                state["k"], = ax_k.plot([], [], color=BLUE, lw=2, marker="o", ms=4)
            elif stage == "eps":
                eps = data["eps"]
                gap = data["eps_tf"] - 4.0
                ax_k.set(xscale="log", yscale="log", xlim=(eps[0] * 2, eps[-1] / 2),
                         ylim=(gap.min() / 3, gap.max() * 3), xlabel="smoothing ε",
                         ylabel="t_f − 4")
                state["k"], = ax_k.plot([], [], color=BLUE, lw=2, marker="o", ms=4)
            else:
                R = data["bang_R"]
                fin = R[np.isfinite(R) & (R > 0)]
                ax_k.set(yscale="log", xlim=(-0.5, len(R) - 0.5),
                         ylim=(max(fin.min() * 0.3, 1e-16), fin.max() * 3),
                         xlabel="residual evaluation", ylabel="‖multi-arc residual‖")
                state["k"], = ax_k.plot([], [], color=BLUE, lw=2, marker="o", ms=4)
                state["phases"] = []
        head, body = CAPTIONS[stage]
        title.set_text(head)
        caption.set_text(body.format(tf=data["tf"], tfr=data["ref_tf"],
                                     vmax=data["ref_v"].max()))
        state["stage"] = stage

    def show(Zs, Us, TF, i):
        hist, cur = state["hist"], state["cur"]
        prev = range(max(0, i - 10), i)
        grid = np.linspace(0, 1, Zs.shape[1])
        for ax, j in ((ax_v, 1), (ax_s, 3)):
            hist[ax].set_segments([np.column_stack([grid * TF[q], Zs[q][:, j]]) for q in prev])
            cur[ax].set_data(grid * TF[i], Zs[i][:, j])
        hist[ax_u].set_segments([np.column_stack([grid * TF[q], Us[q]]) for q in prev])
        cur[ax_u].set_data(grid * TF[i], Us[i])
        return grid * TF[i]

    def draw(frame):
        stage, i = frame
        if stage == "__reset__":
            state["stage"] = None
            return ()
        if stage != state["stage"]:
            setup(stage)
        if stage == "pen":
            Z = data["pen_z"]
            tt = show(Z, data["pen_u"], data["pen_tf"], i)
            for coll in list(ax_v.collections):
                if coll is not state["hist"][ax_v]:
                    coll.remove()
            ax_v.fill_between(tt, V, Z[i][:, 1], where=Z[i][:, 1] > V, color=ORANGE, alpha=0.3,
                              lw=0)
            state["k"].set_data(data["rho"][:i + 1], data["pen_tf"][:i + 1])
            status.set_text(f"ρ = {data['rho'][i]:.1f}   t_f = {data['pen_tf'][i]:.5f}   "
                            f"max(v − V) = {data['pen_viol'][i]:.1e}")
        elif stage == "eps":
            show(data["eps_z"], data["eps_u"], data["eps_tf"], i)
            state["k"].set_data(data["eps"][:i + 1], data["eps_tf"][:i + 1] - 4.0)
            status.set_text(f"ε = {data['eps'][i]:.1e}   t_f = {data['eps_tf'][i]:.10f}")
        elif stage == "bang":
            show(data["bang_z"], data["bang_u"], data["bang_tf"], i)
            R = data["bang_R"]
            state["k"].set_data(np.arange(i + 1), R[:i + 1])
            for a in state["phases"]:
                a.remove()
            state["phases"] = []
            if i == len(R) - 1:
                t_in, t_out = data["junctions"]
                for lo, hi, name in ((0, t_in, "full power"), (t_in, t_out, "cruise: v = V, u = 0"),
                                     (t_out, data["tf"], "brakes")):
                    state["phases"].append(ax_v.axvspan(lo, hi, color=BLUE, alpha=0.06, lw=0))
                    state["phases"].append(ax_v.text((lo + hi) / 2, 0.2, name, ha="center",
                                                     fontsize=9, color=INK2))
            r_txt = f"‖R‖ = {R[i]:.1e}" + ("  (seed: the smoothed solution at ε = 1e-4)"
                                           if i == 0 else "")
            done = f"   t_f = {data['tf']:.15f}" if i == len(R) - 1 else ""
            status.set_text(f"evaluation {i}   {r_txt}{done}")
        else:
            clock = data["clock"]
            for train, gauge, xs, vs in zip(state["trains"], state["gauges"],
                                            (data["bang_z"][-1][:, 0], data["ref_x"]),
                                            (data["bang_z"][-1][:, 1], data["ref_v"])):
                train.set_x(xs[i] - 0.2)
                gauge.set_text(f"v = {vs[i]:.2f}")
            for h in state["heads"]:
                h.set_xdata([clock[i]])
            status.set_text(f"t = {clock[i]:.2f}")
        return ()

    plan = _plan(data, fps)
    mid = next(j for j, f in enumerate(plan) if f[0] == "motion" and f[1] >= 0.55 * len(data["clock"]))
    return write(fig, draw, plan, path, fps, stills_dir, prefix="speed_limit_",
                 still_frames={"motion": mid}, problem=PROBLEM)
