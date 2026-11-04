"""Record and animate how the Bryson-Denham problem is solved, in both regimes.

A unit mass leaves ``x = 0`` at speed 1 and must come back at speed -1 at
``t = 1`` without passing a wall at ``x = l``, using the least ``int u^2/2``.
The limit has order 2: the wall is touched at one instant for
``1/6 <= l < 1/4`` and ridden for a while below that.

For each regime (``l = 0.2``, a touch, and ``l = 0.1``, a boundary arc):

1. **Penalty continuation.** ``rho/2 max(0, x - l)^2`` is added to the cost
   and ``rho`` raised from 1. The solution is pushed back toward the wall,
   and the cost climbs toward the closed form from below.
2. **Exact arcs.** The penalty solution seeds multi-arc shooting with the
   structure (a touch, or a boundary arc): Newton then lands on the closed
   form, with the costate ``lambda_x`` jumping where the wall is met.

Then both solutions move side by side.
"""

import numpy as np

from problems import (MultiArcShooting, SingleShooting, _kernels, bryson_denham_exact, examples,
                      parameter_continuation)
from visuals.common import (BLUE, GRID, INK, INK2, MUTED, ORANGE, RAMP, Problem, header, load, recording,
                            save, style, write)

__all__ = ["record", "animate", "save", "load", "PROBLEM"]
CASES = (("touch", 0.2, 8, 1e-10, 3), ("arc", 0.1, 7, 1e-8, 3))   # name, l, log10 rho, tol, seed exp


PROBLEM = Problem(
    name="The Bryson-Denham problem",
    story=("A unit mass leaves x = 0 at speed 1 and must come back to x = 0 at speed −1 at "
           "t = 1, without passing a wall at x = l,\nwith the least control effort. The limit "
           "has order 2: for 1/6 ≤ l < 1/4 the wall is touched at one instant, below that it "
           "is\nridden for a while (Bryson, Denham & Dreyfus 1963)."),
    formulation=[
        ("minimize", r"$J = \int_0^1 \frac{u^2}{2}\,dt$"),
        ("dynamics", r"$\dot x = v, \quad \dot v = u$"),
        ("path limit", r"$x(t) \leq l$   ($l = 0.2$: one touch;  $l = 0.1$: a boundary arc)"),
        ("boundary", r"$x(0) = 0, \ v(0) = 1; \quad x(1) = 0, \ v(1) = -1$"),
    ],
    solved_by=[
        "Exterior penalty ρ/2 max(0, x − l)², with ρ raised by continuation.",
        "Multi-arc shooting with the structure (a touch, or a boundary arc), seeded from the "
        "penalty solution: the closed form.",
    ],
    summary="min ∫u²/2 dt,  ẍ = u,  x ≤ l,  x(0) = 0,  v: 1 → −1 at t = 1",
    source="Closed form: J = 2.24 (l = 0.2, touch),  J = 4/(9l) (l = 0.1, boundary arc)",
)

def record(cache_dir=None, verbose=True):
    k = _kernels(examples.bryson_denham(), cache_dir)
    out = {}
    for name, ell, rmax, tol, seed_exp in CASES:
        v = SingleShooting(k, [ell, 1.0]).solve([0, 2.0, 0, 0]).v
        path = parameter_continuation(k, [ell, 1.0], "rho", np.logspace(0, rmax, 4 * rmax + 1), v,
                                      solve_options=dict(tol=tol))
        rho, Z, U, J, viol = [], [], [], [], []
        for r_, s in path:
            tr = k.trajectory(0, s.z0, 1.0, [ell, r_], n_points=401)
            rho.append(r_)
            Z.append(tr["z"])
            U.append(tr["u"][:, 0])
            J.append(k.propagate(0, s.z0, 1.0, [ell, r_], cost=True)["cost"])
            viol.append(s.max_violation)
        out[f"{name}_t"] = tr["t"]
        out[f"{name}_rho"], out[f"{name}_pen_z"] = np.array(rho), np.array(Z)
        out[f"{name}_pen_u"], out[f"{name}_pen_J"] = np.array(U), np.array(J)
        out[f"{name}_pen_viol"] = np.array(viol)
        out[f"{name}_ell"] = ell
        out[f"{name}_exact_J"] = bryson_denham_exact(ell)

        # Exact arcs, seeded from a moderate penalty so the Newton steps show.
        j = int(np.argmin(np.abs(np.log10(out[f"{name}_rho"]) - seed_exp)))
        arcs = [("touch" if name == "touch" else "boundary", "x_max")]
        shoot, v0 = recording(MultiArcShooting).from_penalty(k, [ell, 0.0], arcs, path[j][1],
                                                            [ell, rho[j]])
        sol = shoot.solve(v0)
        tt = np.linspace(0, 1, 401)
        Ze, Ue, Re = [], [], []
        for v, R in [(v0, np.nan)] + shoot.evaluations:
            try:
                tr = shoot.trajectory(v, n_points=600)
            except Exception:                   # times out of order, failed propagation
                continue
            Ze.append(np.column_stack([np.interp(tt, tr["t"], tr["z"][:, i]) for i in range(4)]))
            Ue.append(np.interp(tt, tr["t"], tr["u"][:, 0]))
            Re.append(R)
        out[f"{name}_exact_z"], out[f"{name}_exact_u"] = np.array(Ze), np.array(Ue)
        out[f"{name}_exact_R"] = np.array(Re)
        out[f"{name}_seed_rho"] = rho[j]
        out[f"{name}_cost"] = shoot.cost(sol.v)
        out[f"{name}_junctions"] = np.array(shoot.unpack_arcs(sol.v)[0][:2])
        if verbose:
            print(f"{name}: {len(path)} penalty solves to rho = {rho[-1]:.0e}, exact from "
                  f"rho = {rho[j]:.0e} in {len(shoot.evaluations)} evaluations, "
                  f"J = {out[f'{name}_cost']:.12f} (exact {bryson_denham_exact(ell):.12f})",
                  flush=True)
    return out


CAPTIONS = {
    "touch_pen": ("1  Touch regime (l = 0.2): penalty continuation",
                  "ρ/2 max(0, x − l)² is added to the cost and ρ raised from 1 to 1e8. The mass is "
                  "pushed back toward the wall,\nand the cost climbs toward the closed form 2.24 "
                  "from below."),
    "touch_exact": ("2  Touch regime: exact arcs",
                    "The penalty solution at ρ = {rho:.0e} seeds multi-arc shooting with one touch "
                    "(x = l and v = 0 at one instant).\nNewton lands on the closed form; λ_x "
                    "jumps by ν = 9.6 at the contact."),
    "arc_pen": ("3  Boundary-arc regime (l = 0.1): penalty continuation",
                "Here the optimum rides the wall from t = 0.3 to 0.7. The penalized mass sinks into "
                "the wall and wobbles\nalong it; the cost closes in on 4/(9l) from below."),
    "arc_exact": ("4  Boundary-arc regime: exact arcs",
                  "Seeded from ρ = {rho:.0e}: entry and exit times, and the jump π at entry, are "
                  "unknowns. On the arc u = 0 holds\nthe mass at the wall exactly; the cost is "
                  "4/(9l) to round-off."),
    "motion": ("5  Both solutions",
               "Touch (l = 0.2, orange wall) and boundary arc (l = 0.1, blue wall), in real time "
               "(slowed 4x)."),
}


def _plan(data, fps):
    plan = []
    def steps(n, most, frames):
        idx = np.unique(np.linspace(0, n - 1, min(n, most)).round().astype(int))
        return [int(i) for i in idx for _ in range(max(2, frames // len(idx)))]

    for name in ("touch", "arc"):
        n = len(data[f"{name}_rho"])
        plan += [(f"{name}_pen", i) for i in steps(n, 40, 90)]
        plan += [(f"{name}_pen", n - 1)] * fps
        n = len(data[f"{name}_exact_R"])
        plan += [(f"{name}_exact", i) for i in steps(n, 30, 60)]
        plan += [(f"{name}_exact", n - 1)] * fps
    n = len(data["touch_t"])
    plan += [("motion", int(i)) for i in np.linspace(0, n - 1, 4 * fps)]
    plan += [("motion", n - 1)] * fps
    return plan


def animate(data, path, fps=30, dpi=100, stills_dir=None):
    plt = style()
    from matplotlib.collections import LineCollection

    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], left=0.06, right=0.97, top=0.80,
                          bottom=0.105, hspace=0.45, wspace=0.18)
    ax_x = fig.add_subplot(gs[:, 0])
    ax_u = fig.add_subplot(gs[0, 1])
    ax_l = fig.add_subplot(gs[1, 1], sharex=ax_u)
    ax_k = fig.add_subplot(gs[2, 1])
    title, caption = header(fig)
    status = ax_x.text(0.02, 0.98, "", transform=ax_x.transAxes, va="top", fontsize=11)
    t = data["touch_t"]
    hist = {ax: LineCollection([], colors=MUTED, linewidths=1, alpha=0.55)
            for ax in (ax_x, ax_u, ax_l)}
    for ax, c in hist.items():
        ax.add_collection(c)
    cur = {ax: ax.plot([], [], color=BLUE, lw=2.2)[0] for ax in (ax_x, ax_u, ax_l)}
    second = {ax: ax.plot([], [], color=ORANGE, lw=2.2)[0] for ax in (ax_x, ax_u, ax_l)}
    state = {"stage": None, "art": []}

    def frame_axes():
        for ax in (ax_x, ax_u, ax_l, ax_k):
            for a in list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.texts):
                if a not in (status,) and a not in cur.values() and a not in second.values() \
                        and a not in hist.values():
                    a.remove()
        ax_k.cla()

    def setup(stage):
        frame_axes()
        for c in hist.values():
            c.set_segments([])
        for line in list(cur.values()) + list(second.values()):
            line.set_data([], [])
        case = "touch" if stage.startswith("touch") or stage == "motion" else "arc"
        ell = data[f"{case}_ell"]
        state["case"] = case
        for ax in (ax_x, ax_u, ax_l):
            ax.grid(color=GRID, lw=0.8)
            ax.set_xlim(0, 1)
        if stage == "motion":
            ax_x.set(xlim=(-0.05, 0.3), ylim=(-0.6, 1.6), xlabel="position x", ylabel="",
                     yticks=[])
            ax_x.grid(False)
            for y, c, name in ((1.0, ORANGE, "touch"), (0.0, BLUE, "arc")):
                e = data[f"{name}_ell"]
                ax_x.axhline(y, color=INK2, lw=1)
                ax_x.add_patch(plt.Rectangle((e, y - 0.35), 0.01, 0.7, color=c))
                ax_x.text(-0.04, y + 0.25, f"l = {e}", color=c)
            state["blocks"] = [ax_x.add_patch(plt.Rectangle((0, y - 0.12), 0.03, 0.24, color=c))
                               for y, c in ((1.0, ORANGE), (0.0, BLUE))]
            ax_u.set(ylabel="force u", ylim=(-7.5, 1))
            ax_l.set(ylabel="x(t)", ylim=(-0.02, 0.27))
            ax_k.set(xlim=(0, 1), ylim=(-1.15, 1.15), ylabel="velocity v(t)", xlabel="t")
            ax_k.axhline(0, color=INK2, lw=0.8)
            for name, c in (("touch", ORANGE), ("arc", BLUE)):
                z = data[f"{name}_exact_z"][-1]
                ax_u.plot(t, data[f"{name}_exact_u"][-1], color=c, lw=2)
                ax_l.plot(t, z[:, 0], color=c, lw=2)
                ax_l.axhline(data[f"{name}_ell"], color=c, lw=1, ls="--")
                ax_k.plot(t, z[:, 1], color=c, lw=2)
            state["heads"] = [ax.axvline(0, color=INK2, lw=1) for ax in (ax_u, ax_l, ax_k)]
        else:
            ax_x.set(xlim=(0, 1), ylim=(-0.02, 0.27), xlabel="t", ylabel="position x")
            ax_x.axhline(ell, color=ORANGE, lw=2, ls="--")
            ax_x.text(0.01, ell + 0.004, f"wall x = l = {ell}", color=ORANGE)
            ax_u.set(ylabel="force u", ylim=(-7.5 if case == "arc" else -4, 1.5))
            lx = np.concatenate([data[f"{case}_exact_z"][..., 2].ravel(),
                                 data[f"{case}_pen_z"][..., 2].ravel()])
            lo, hi = np.percentile(lx, [1, 99])
            ax_l.set(ylabel="costate λ_x", xlabel="t", ylim=(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)))
            if stage.endswith("pen"):
                rho = data[f"{case}_rho"]
                ax_k.set(xscale="log", xlim=(rho[0] / 2, rho[-1] * 2), xlabel="penalty weight ρ",
                         ylabel="cost J")
                Jp = data[f"{case}_pen_J"]
                ax_k.set_ylim(Jp.min() - 0.1 * (data[f"{case}_exact_J"] - Jp.min()),
                              data[f"{case}_exact_J"] * 1.01)
                ax_k.axhline(data[f"{case}_exact_J"], color=ORANGE, lw=1.5, ls="--")
                ax_k.text(rho[0], data[f"{case}_exact_J"] * 1.002,
                          f"closed form {data[f'{case}_exact_J']:.4f}", color=ORANGE, fontsize=9)
                state["k"], = ax_k.plot([], [], color=BLUE, lw=2, marker="o", ms=3)
            else:
                R = data[f"{case}_exact_R"]
                fin = R[np.isfinite(R) & (R > 0)]
                ax_k.set(yscale="log", xlim=(-0.5, len(R) - 0.5),
                         ylim=(max(fin.min() * 0.3, 1e-16), fin.max() * 3),
                         xlabel="residual evaluation", ylabel="‖multi-arc residual‖")
                state["k"], = ax_k.plot([], [], color=BLUE, lw=2, marker="o", ms=4)
        ax_k.grid(color=GRID, lw=0.8)
        head, body = CAPTIONS[stage]
        if "exact" in stage:
            body = body.format(rho=data[f"{case}_seed_rho"])
        title.set_text(head)
        caption.set_text(body)
        state["stage"] = stage

    def show(z, u, i_hist, Zs, Us):
        for ax, j in ((ax_x, 0), (ax_l, 2)):
            hist[ax].set_segments([np.column_stack([t, Zs[q][:, j]]) for q in i_hist])
            cur[ax].set_data(t, z[:, j])
        hist[ax_u].set_segments([np.column_stack([t, Us[q]]) for q in i_hist])
        cur[ax_u].set_data(t, u)

    def draw(frame):
        stage, i = frame
        if stage == "__reset__":
            state["stage"] = None
            return ()
        if stage != state["stage"]:
            setup(stage)
        case = state["case"]
        if stage.endswith("pen"):
            Z, U = data[f"{case}_pen_z"], data[f"{case}_pen_u"]
            show(Z[i], U[i], range(max(0, i - 10), i), Z, U)
            for coll in list(ax_x.collections):
                if coll not in hist.values():
                    coll.remove()
            x = Z[i][:, 0]
            ax_x.fill_between(t, data[f"{case}_ell"], x, where=x > data[f"{case}_ell"],
                              color=ORANGE, alpha=0.3, lw=0)
            state["k"].set_data(data[f"{case}_rho"][: i + 1], data[f"{case}_pen_J"][: i + 1])
            status.set_text(f"ρ = {data[f'{case}_rho'][i]:.1e}   J = {data[f'{case}_pen_J'][i]:.5f}"
                            f"   max violation = {data[f'{case}_pen_viol'][i]:.1e}")
        elif stage.endswith("exact"):
            Z, U, R = data[f"{case}_exact_z"], data[f"{case}_exact_u"], data[f"{case}_exact_R"]
            show(Z[i], U[i], range(max(0, i - 10), i), Z, U)
            state["k"].set_data(np.arange(i + 1), R[: i + 1])
            r_txt = "seed from the penalty solution" if not np.isfinite(R[i]) else f"‖R‖ = {R[i]:.1e}"
            done = f"   J = {data[f'{case}_cost']:.10f}" if i == len(R) - 1 else ""
            status.set_text(f"evaluation {i}   {r_txt}{done}")
        else:
            tt = t[i]
            for blk, name in zip(state["blocks"], ("touch", "arc")):
                blk.set_x(data[f"{name}_exact_z"][-1][i, 0] - 0.03)
            for h in state["heads"]:
                h.set_xdata([tt])
            status.set_text(f"t = {tt:.2f}   J = {data['touch_cost']:.4f} (touch), "
                            f"{data['arc_cost']:.4f} (arc)")
        return ()

    plan = _plan(data, fps)
    mid = next(j for j, f in enumerate(plan) if f[0] == "motion" and f[1] >= 0.45 * len(t))
    return write(fig, draw, plan, path, fps, stills_dir, prefix="bryson_denham_",
                 still_frames={"motion": mid}, problem=PROBLEM)
