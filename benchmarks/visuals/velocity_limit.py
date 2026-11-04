"""Record and animate the velocity-limited move with a force limit: exact arcs
with a bounded (clipped) control.

An elevator car moves one unit up in ``t = 1``, rest to rest, with the least
``int u^2/2``, under a speed limit ``v <= V`` and a force limit
``|u| <= u_max``. With the speed limit alone, the motor would need
``u(0) = 2 V / t1`` (9.6 at ``V = 1.2``); capped at 8, the car starts at full
force, ramps down onto the limit, cruises, and mirrors that on arrival.

1. **Penalty continuation.** ``rho/2 max(0, v - V)^2`` is added to the cost
   and ``rho`` raised from 1 to 100. The force limit holds exactly
   throughout, as the clipped law ``u = clip(-lambda_v, -u_max, u_max)``.
2. **Exact arcs.** The penalty solution seeds multi-arc shooting with one
   boundary arc. The free arcs keep the clipped law, and their kinks (where
   ``u`` meets ``u_max``) are located like switches; on the arc ``u = 0``.
3. **The move**, beside the same move without the force limit.
"""

import numpy as np

from problems import (MultiArcShooting, SingleShooting, _kernels, examples,
                      parameter_continuation)
from visuals.common import (BLUE, GRID, INK2, MUTED, ORANGE, Problem, header, load,
                            recording, save, style, write)

__all__ = ["record", "animate", "save", "load", "PROBLEM"]
V, UMAX = 1.2, 8.0
N_T = 401

PROBLEM = Problem(
    name="Elevator move with speed and force limits",
    story=("An elevator car rises one unit of height in 1 s, from rest to rest, with the least "
           "energy. Its speed is limited to V = 1.2 and\nits motor force to |u| ≤ 8. With the "
           "speed limit alone the motor would need a force of 9.6 at the start, so the force "
           "limit\nis active too: exact arcs with a bounded control."),
    formulation=[
        ("minimize", r"$J = \int_0^1 \frac{u^2}{2}\,dt$"),
        ("dynamics", r"$\dot x = v, \quad \dot v = u$"),
        ("speed limit", r"$v(t) \leq V = 1.2$   (order 1: $u$ appears in $\dot v$)"),
        ("force limit", r"$|u(t)| \leq u_{max} = 8$"),
        ("boundary", r"$x(0) = 0, \ v(0) = 0; \quad x(1) = 1, \ v(1) = 0$"),
    ],
    solved_by=[
        "Exterior penalty ρ/2 max(0, v − V)² on the speed limit, ρ raised from 1 to 100 by "
        "continuation; the force limit is exact\n      throughout, as the clipped law "
        "u = clip(−λ_v, −8, 8).",
        "Multi-arc shooting with one boundary arc (v = V, u = 0), seeded from the penalty "
        "solution; the integration restarts\n      where u meets its bound, and the cost "
        "matches the closed form to about 1e-12.",
    ],
    summary="min ∫u²/2 dt,  ẍ = u,  v ≤ 1.2,  |u| ≤ 8,  x: 0 → 1 in t = 1, rest to rest",
    source="Closed form: velocity_limited_move_bounded_exact in auto_optimal_control.examples",
)


def _unbounded(t):
    """The move with the speed limit only (closed form): x, v, u."""
    t1 = 3 * (V - 1) / (2 * V)
    c = 2 * V / t1 ** 2

    def up(s):
        return c * (t1 * s ** 2 / 2 - s ** 3 / 6), c * (t1 * s - s ** 2 / 2), c * (t1 - s)

    x1 = up(t1)[0]
    x, v, u = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    a, m, b = t <= t1, (t > t1) & (t < 1 - t1), t >= 1 - t1
    x[a], v[a], u[a] = up(t[a])
    x[m], v[m] = x1 + V * (t[m] - t1), V
    xe, ve, ue = up(1 - t[b])
    x[b], v[b], u[b] = 1 - xe, ve, -ue
    return x, v, u


def record(cache_dir=None, verbose=True):
    k = _kernels(examples.velocity_limited_move(bounded=True), cache_dir)
    tt = np.linspace(0, 1, N_T)
    out = {"t": tt}
    v = SingleShooting(k, [V, 1.0, UMAX]).solve([12.0, 6.0, 0, 0]).v
    path = parameter_continuation(k, [V, 1.0, UMAX], "rho", np.logspace(0, 2, 17), v)
    Z, U, J, viol = [], [], [], []
    for rho, s in path:
        tr = k.trajectory(0, s.z0, 1.0, [V, rho, UMAX], t_eval=tt)
        Z.append(tr["z"])
        U.append(tr["u"][:, 0])
        J.append(k.propagate(0, s.z0, 1.0, [V, rho, UMAX], cost=True)["cost"])
        viol.append(s.max_violation)
    out.update(rho=np.array([r for r, _ in path]), pen_z=np.array(Z), pen_u=np.array(U),
               pen_J=np.array(J), pen_viol=np.array(viol))

    rho, sol = path[-1]
    shoot, v0 = recording(MultiArcShooting).from_penalty(
        k, [V, 0.0, UMAX], [("boundary", "v_max")], sol, [V, rho, UMAX])
    ex = shoot.solve(v0)
    Ze, Ue, Re, Te = [], [], [], []
    for w, R in [(v0, float(np.linalg.norm(shoot.residual(v0))))] + shoot.evaluations:
        tr = shoot.trajectory(w, n_points=800)
        Ze.append(np.column_stack([np.interp(tt, tr["t"], tr["z"][:, i]) for i in range(4)]))
        Ue.append(np.interp(tt, tr["t"], tr["u"][:, 0]))
        Re.append(R)
        Te.append(shoot.unpack_arcs(w)[0][:2])
    J_exact, t1, s_full = examples.velocity_limited_move_bounded_exact(V, UMAX)
    out.update(exact_z=np.array(Ze), exact_u=np.array(Ue), exact_R=np.array(Re),
               exact_junctions=np.array(Te), cost=shoot.cost(ex.v), J_exact=J_exact,
               t1=t1, s_full=s_full, seed_rho=rho)
    out["ref_x"], out["ref_v"], out["ref_u"] = _unbounded(tt)
    out["ref_J"] = 4 * V ** 2 / (3 * (3 * (V - 1) / (2 * V)))
    if verbose:
        print(f"{len(path)} penalty solves to rho = {rho:.0e}; exact arcs in "
              f"{len(shoot.evaluations)} evaluations: J = {out['cost']:.13f} "
              f"(closed form {J_exact:.13f})", flush=True)
    return out


CAPTIONS = {
    "pen": ("1  Penalty continuation on the speed limit",
            "ρ/2 max(0, v − V)² is added to the cost and ρ raised from 1 to 100. The speed is "
            "pushed down toward V = 1.2 (shaded: over the\nlimit), while the force stays "
            "clipped at |u| ≤ 8 exactly: u = clip(−λ_v, −8, 8)."),
    "exact": ("2  Exact arcs with a bounded control",
              "Seeded from ρ = {rho:.0f}: full force for {s:.3f} s, a ramp to u = 0 at "
              "t₁ = {t1:.4f}, the limit, the mirror image. Convergence is linear,\nnot "
              "quadratic: at an order-1 entry with a smooth control, H continuity has a double "
              "root (λ_v → 0 from both sides)."),
    "motion": ("3  The move",
               "Blue: with the force limit (J = {J:.4f}). Orange: speed limit only, which needs "
               "9.6 at the start (J = {Jr:.4f}). Slowed 4x."),
}


def _plan(data, fps):
    def steps(n, most, frames):
        idx = np.unique(np.linspace(0, n - 1, min(n, most)).round().astype(int))
        return [int(i) for i in idx for _ in range(max(2, frames // len(idx)))]

    n = len(data["rho"])
    plan = [("pen", i) for i in steps(n, 40, 4 * fps)] + [("pen", n - 1)] * fps
    n = len(data["exact_R"])
    plan += [("exact", i) for i in steps(n, 30, 3 * fps)] + [("exact", n - 1)] * (2 * fps)
    n = len(data["t"])
    plan += [("motion", int(i)) for i in np.linspace(0, n - 1, 4 * fps)]
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
    ax_l = fig.add_subplot(gs[1, 1], sharex=ax_u)
    ax_k = fig.add_subplot(gs[2, 1])
    title, caption = header(fig)
    status = ax_v.text(0.02, 0.98, "", transform=ax_v.transAxes, va="top", fontsize=11)
    t = data["t"]
    keep = {status}
    state = {"stage": None}

    def clear():
        for ax in (ax_v, ax_u, ax_l, ax_k):
            for a in list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.texts):
                if a not in keep:
                    a.remove()
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.set_visible(True)

    def setup(stage):
        clear()
        state["hist"] = {}
        for ax in (ax_v, ax_u, ax_l):
            c = LineCollection([], colors=MUTED, linewidths=1, alpha=0.55)
            ax.add_collection(c)
            state["hist"][ax] = c
        state["cur"] = {ax: ax.plot([], [], color=BLUE, lw=2.2)[0] for ax in (ax_v, ax_u, ax_l)}
        for ax in (ax_v, ax_u, ax_l, ax_k):
            ax.grid(color=GRID, lw=0.8)
        if stage == "motion":
            ax_v.set(xlim=(-1.6, 1.6), ylim=(-0.12, 1.25), xticks=[], xlabel="", ylabel="height x")
            ax_v.grid(False)
            for x0, c, name in ((-0.7, BLUE, "|u| ≤ 8"), (0.7, ORANGE, "no force limit")):
                ax_v.add_patch(plt.Rectangle((x0 - 0.45, -0.1), 0.9, 1.3, fill=False,
                                             ec=INK2, lw=1))
                ax_v.text(x0, 1.21, name, color=c, ha="center", fontsize=11)
            state["cars"] = [ax_v.add_patch(plt.Rectangle((x0 - 0.35, 0), 0.7, 0.1, color=c))
                             for x0, c in ((-0.7, BLUE), (0.7, ORANGE))]
            state["gauges"] = [ax_v.text(x0, -0.07, "", ha="center", fontsize=10, color=c)
                               for x0, c in ((-0.7, BLUE), (0.7, ORANGE))]
            ax_u.set(xlim=(0, 1), ylim=(-10.5, 10.5), ylabel="force u")
            ax_l.set(xlim=(0, 1), ylim=(0, 1.35), ylabel="speed v", xlabel="t")
            ax_k.set_visible(False)
            for ax, lim in ((ax_u, UMAX), (ax_l, V)):
                ax.axhline(lim, color=INK2, lw=1, ls="--")
            ax_u.axhline(-UMAX, color=INK2, lw=1, ls="--")
            ax_u.plot(t, data["exact_u"][-1], color=BLUE, lw=2)
            ax_u.plot(t, data["ref_u"], color=ORANGE, lw=2)
            ax_l.plot(t, data["exact_z"][-1][:, 1], color=BLUE, lw=2)
            ax_l.plot(t, data["ref_v"], color=ORANGE, lw=2)
            state["heads"] = [ax.axvline(0, color=INK2, lw=1) for ax in (ax_u, ax_l)]
        else:
            ax_v.set(xlim=(0, 1), ylim=(0, 1.62), xlabel="t", ylabel="speed v")
            ax_v.axhline(V, color=ORANGE, lw=2, ls="--")
            ax_v.text(0.01, V + 0.015, f"speed limit V = {V}", color=ORANGE)
            ax_u.set(xlim=(0, 1), ylim=(-10.5, 10.5), ylabel="force u")
            for s in (1, -1):
                ax_u.axhline(s * UMAX, color=ORANGE, lw=1.5, ls="--")
            ax_u.text(0.99, UMAX + 0.6, "|u| ≤ 8", color=ORANGE, ha="right", fontsize=9)
            lv = np.concatenate([data["exact_z"][..., 3].ravel(), data["pen_z"][..., 3].ravel()])
            lo, hi = np.percentile(lv, [1, 99])
            ax_l.set(xlim=(0, 1), ylim=(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)),
                     ylabel="costate λ_v", xlabel="t")
            if stage == "pen":
                rho = data["rho"]
                ax_k.set(xscale="log", xlim=(rho[0] / 1.5, rho[-1] * 1.5),
                         xlabel="penalty weight ρ", ylabel="cost J")
                Jp = data["pen_J"]
                ax_k.set_ylim(Jp.min() - 0.1 * (data["J_exact"] - Jp.min()),
                              data["J_exact"] + 0.15 * (data["J_exact"] - Jp.min()))
                ax_k.axhline(data["J_exact"], color=ORANGE, lw=1.5, ls="--")
                ax_k.text(rho[0], data["J_exact"], f"closed form {data['J_exact']:.4f}",
                          color=ORANGE, fontsize=9, va="bottom")
                state["k"], = ax_k.plot([], [], color=BLUE, lw=2, marker="o", ms=3)
            else:
                R = data["exact_R"]
                fin = R[np.isfinite(R) & (R > 0)]
                ax_k.set(yscale="log", xlim=(-0.5, len(R) - 0.5),
                         ylim=(max(fin.min() * 0.3, 1e-16), fin.max() * 3),
                         xlabel="residual evaluation", ylabel="‖multi-arc residual‖")
                state["k"], = ax_k.plot([], [], color=BLUE, lw=2, marker="o", ms=4)
                state["phases"] = []
        head, body = CAPTIONS[stage]
        title.set_text(head)
        caption.set_text(body.format(rho=data["seed_rho"], s=data["s_full"], t1=data["t1"],
                                     J=data["cost"], Jr=data["ref_J"]))
        state["stage"] = stage

    def show(Zs, Us, i):
        hist, cur = state["hist"], state["cur"]
        prev = range(max(0, i - 10), i)
        for ax, j in ((ax_v, 1), (ax_l, 3)):
            hist[ax].set_segments([np.column_stack([t, Zs[q][:, j]]) for q in prev])
            cur[ax].set_data(t, Zs[i][:, j])
        hist[ax_u].set_segments([np.column_stack([t, Us[q]]) for q in prev])
        cur[ax_u].set_data(t, Us[i])

    def draw(frame):
        stage, i = frame
        if stage == "__reset__":
            state["stage"] = None
            return ()
        if stage != state["stage"]:
            setup(stage)
        if stage == "pen":
            Z = data["pen_z"]
            show(Z, data["pen_u"], i)
            for coll in list(ax_v.collections):
                if coll is not state["hist"][ax_v]:
                    coll.remove()
            ax_v.fill_between(t, V, Z[i][:, 1], where=Z[i][:, 1] > V, color=ORANGE, alpha=0.3,
                              lw=0)
            state["k"].set_data(data["rho"][:i + 1], data["pen_J"][:i + 1])
            status.set_text(f"ρ = {data['rho'][i]:.1f}   J = {data['pen_J'][i]:.5f}   "
                            f"max(v − V) = {data['pen_viol'][i]:.1e}")
        elif stage == "exact":
            show(data["exact_z"], data["exact_u"], i)
            R = data["exact_R"]
            state["k"].set_data(np.arange(i + 1), R[:i + 1])
            for a in state["phases"]:
                a.remove()
            state["phases"] = []
            if i == len(R) - 1:
                s, t1 = data["s_full"], data["t1"]
                for lo, hi, name in ((0, s, "full\nforce"), (s, t1, "ramp"),
                                     (t1, 1 - t1, "boundary arc: v = V, u = 0"),
                                     (1 - t1, 1 - s, "ramp"), (1 - s, 1, "full\nbrake")):
                    state["phases"].append(ax_v.axvspan(lo, hi, color=BLUE, alpha=0.06, lw=0))
                    state["phases"].append(ax_v.text((lo + hi) / 2, 0.25, name, ha="center",
                                                     fontsize=8.5, color=INK2))
            r_txt = f"‖R‖ = {R[i]:.1e}" + ("  (seed from the penalty solution)" if i == 0 else "")
            t_in, t_out = data["exact_junctions"][i]
            done = (f"   J = {data['cost']:.11f} (exact {data['J_exact']:.11f})"
                    if i == len(R) - 1 else f"   arc [{t_in:.4f}, {t_out:.4f}]")
            status.set_text(f"evaluation {i}   {r_txt}{done}")
        else:
            x = data["exact_z"][-1][:, 0]
            for car, gauge, xs, vs in zip(state["cars"], state["gauges"], (x, data["ref_x"]),
                                          (data["exact_z"][-1][:, 1], data["ref_v"])):
                car.set_y(xs[i])
                gauge.set_text(f"v = {vs[i]:.2f}")
            for h in state["heads"]:
                h.set_xdata([t[i]])
            status.set_text(f"t = {t[i]:.2f}")
        return ()

    plan = _plan(data, fps)
    mid = next(j for j, f in enumerate(plan) if f[0] == "motion" and f[1] >= 0.4 * len(t))
    return write(fig, draw, plan, path, fps, stills_dir, prefix="velocity_limit_",
                 still_frames={"motion": mid}, problem=PROBLEM)
