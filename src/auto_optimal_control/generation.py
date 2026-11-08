"""Extremals without a boundary-value problem: integrate the canonical system from one end.

A solve fixes both ends of the motion and searches for the costates that join
them. Generation fixes only the start. From ``x0`` and any costates ``lam0``
(on ``H = 0`` when the final time is free) the state and costate equations
integrate to *some* end ``x(T)``, and the extremal found is exact for the
problem from ``x0`` to that end: when both ends are fixed states, the costates
are free at both ends, so nothing else has to hold there. Many extremals, from
any starts to wherever they lead, are the data of
:mod:`~auto_optimal_control.embedding`.

Path constraints are what make this more than one integration. A trajectory
from arbitrary costates either stays inside its limits (then it is a valid
extremal as it stands) or breaks one, and then it is not an extremal of the
constrained problem at all. :class:`ExtremalGenerator` turns each violation
into a boundary arc and moves the unknowns ``(lam0, t_in, t_out, pi)`` by the
smallest step (minimum-norm Newton) that makes every junction condition hold:
the same rows :class:`~auto_optimal_control.shooting.MultiArcShooting`
imposes, tangency (``dg/dt = 0``) at an order-1 entry where ``H`` is regular
and ``H`` continuity where a control may jump, and ``mu = 0`` at an exit. The
end conditions are not among them, so this projection is not a solve: the end
is whatever the extremal reaches. All the junctions are local conditions;
none couples to the far end.

Only order-1 limits are supported, and arcs must lie strictly inside the
motion (a sample whose limit is violated at either end is rejected).
"""

import time
from dataclasses import dataclass, field

import numpy as np

from auto_optimal_control.homotopy import OFF
from auto_optimal_control.shooting import MultiArcShooting, ShootingError

_FAILURES = (ShootingError, RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError)


class Rejected(Exception):
    """A draw that does not give a valid extremal; ``reason`` says why."""

    def __init__(self, reason, detail=""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass
class Extremal:
    """One exact extremal from ``x0`` at ``t0`` to ``xf`` at ``tf``.

    ``arcs`` are its :class:`~auto_optimal_control.shooting.MultiArcShooting`
    entries (``("enter", c)`` and ``("exit", c)`` in time order) and ``values``
    per arc, in the order entered, ``(t_in, t_out, pi)``. ``junction_residual``
    is the largest junction condition (and ``H(t0)`` when the final time is
    free) left after the projection, ``cost`` the running cost, and
    ``checks`` what :meth:`ExtremalGenerator.check` found.
    """

    x0: np.ndarray
    lam0: np.ndarray
    t0: float
    tf: float
    xf: np.ndarray
    lamf: np.ndarray
    arcs: tuple
    values: tuple
    cost: float
    junction_residual: float
    checks: dict = field(default_factory=dict)

    @property
    def structure(self):
        """The arcs as a hashable tuple of ``(event, constraint)``."""
        return tuple((e, int(c)) for e, c in self.arcs)


class ExtremalGenerator:
    """Generate exact extremals of one problem by integrating from the start.

    Parameters
    ----------
    kernels, params
        The exact problem's kernels and runtime parameters, as a solve uses
        them (:meth:`from_homotopy` takes both from a
        :class:`~auto_optimal_control.homotopy.Homotopy`). Only the dynamics,
        cost and limits matter here: the problem's end conditions are not used.
    n_points : int
        Samples per motion when looking for violated limits; each local peak
        is then refined to its exact height.
    tol : float
        The junction conditions are met to this (absolute) level.
    """

    def __init__(self, kernels, params, *, rtol=1e-12, atol=1e-12, n_points=1201, tol=1e-10,
                 max_rounds=None, max_newton=60, truncate=True, vary_start="none",
                 start_scale=1e-2):
        self.kernels = k = kernels
        self.params = np.asarray(params, dtype=float)
        self.options = dict(rtol=rtol, atol=atol)
        self.n_points = n_points
        self.tol = tol
        self.max_rounds = max_rounds or 4 * max(k.N_CONSTRAINTS, 1) + 8
        self.max_newton = max_newton
        # When an arc cannot be made valid, end the motion just before it instead.
        self.truncate = truncate
        # Which parts of the start the projection may move: "none", "positions" (the
        # first third of the states, for a jerk-controlled chain: rest stays rest) or "all".
        n = k.N_STATES
        self._vary = {"none": np.zeros(0, dtype=int), "positions": np.arange(n // 3),
                      "all": np.arange(n)}[vary_start]
        self.start_scale = start_scale
        self.debug = False
        # The states an anchor may move to reach its limit (None: all). For a torque limit
        # under jerk control the accelerations reach it with the least disturbance.
        self.anchor_states = None
        # Seconds one draw may take before it is rejected (None: no limit).
        self.max_seconds = None
        self._deadline = np.inf
        bad = [k.CONSTRAINT_NAMES[c] for c in range(k.N_CONSTRAINTS)
               if k.CONSTRAINT_ORDERS[c] != 1 or not k.EXACT_ARCS[c]]
        self.unsupported = bad
        # Regular H (no bang-bang control): an order-1 entry is tangential even
        # with a control at its bound, since the clipped law is continuous.
        self.regular = not k.N_SWITCHING
        self.free_time = bool(k.FREE_FINAL_TIME)
        probe = MultiArcShooting(k, self.params)
        self.t0 = probe.t0
        self.fixed_tf = probe.fixed_tf
        # Offsets of the limits (g - off <= 0) among the parameters, when the kernels have
        # them (the regularized kernels a regular problem is solved on): an arc is then born
        # at the peak of its violation, with the limit raised to it, and grown as the offset
        # falls back to zero.
        names = list(k.param_names)
        self._offsets = {c: names.index(OFF + k.CONSTRAINT_NAMES[c])
                         for c in range(k.N_CONSTRAINTS) if OFF + k.CONSTRAINT_NAMES[c] in names}
        self.birth_steps = 200
        # A new arc's length, as a fraction of the motion, is drawn from this range.
        self.arc_fraction = (0.02, 0.3)
        # How far past the drawn duration a motion may be extended to reach a peak.
        self.extend = 3.0

    def _out_of_time(self):
        if time.perf_counter() > self._deadline:
            raise Rejected("out of time", f"{self.max_seconds} s")

    @classmethod
    def from_homotopy(cls, homotopy, params, **kwargs):
        """The generator of the problem ``homotopy`` solves (its policy applied)."""
        homotopy.params = np.asarray(params, dtype=float)
        return cls(homotopy.exact_kernels, homotopy.exact_params(), **kwargs)

    # -- the shooting of one structure -----------------------------------------

    def _shooting(self, x0, arcs):
        """``MultiArcShooting`` for arcs ``[(c, t_in, t_out, pi), ...]`` and the order of their events."""
        events = []
        for i, (c, t_in, t_out, _) in enumerate(arcs):
            events.append((t_in, 0, "enter", c, i))
            if not np.isnan(t_out):           # an arc open at the end has no exit
                events.append((t_out, 1, "exit", c, i))
        events.sort(key=lambda e: (e[0], e[1]))
        entries = [(kind, c) for _, _, kind, c, _ in events]
        order = [i for _, _, kind, _, i in events if kind == "enter"]
        shoot = MultiArcShooting(self.kernels, self.params, entries, x0=x0, open_end=True,
                                 **self.options)
        if self.regular:
            shoot.tangent_junctions = True
        shoot.max_nfev = 200000
        return shoot, order

    def _pack(self, shoot, order, lam0, arcs, tf):
        base = shoot.pack(lam0, None, tf if self.free_time else None)
        values = [(arcs[i][1], arcs[i][2], [arcs[i][3]]) for i in order]
        return shoot.pack_arcs(base, values)

    def _rows(self, shoot, order, x0, lam0, arcs, tf, jac=True):
        """Junction rows (and ``H(t0)``), and their Jacobian in ``(lam0, per arc t_in, t_out, pi)``."""
        k, n = self.kernels, self.kernels.N_STATES
        v = self._pack(shoot, order, lam0, arcs, tf)
        R, J, segs = shoot._evaluate(v, jac=jac)
        nb = len(shoot._rows)
        rows = list(R[nb:])
        z0 = np.concatenate([x0, lam0])
        cols = shoot.n_free + np.arange(n)
        jac_rows = None
        if jac:
            # Columns: lam0, then the arcs in the caller's order.
            m = len(arcs)
            Jw = np.zeros((len(R) - nb, n + 3 * m))
            Jw[:, :n] = J[nb:, cols]
            for pos, i in enumerate(order):
                Jw[:, self._arc_columns(n, i, arcs[i])] = J[nb:, shoot._arc_slices[pos]]
            jac_rows = Jw
        if self.free_time:
            rows.append(k.hamiltonian(self.t0, z0, self.params))
            if jac:
                _, Hz = k.hamiltonian_gradient(self.t0, z0, self.params)
                extra = np.zeros(jac_rows.shape[1])
                extra[:n] = Hz[n:]
                jac_rows = np.vstack([jac_rows, extra])
        return np.asarray(rows, dtype=float), jac_rows, v, segs

    @staticmethod
    def _arc_columns(n, i, arc):
        """Columns of arc ``i`` among ``(lam0, per arc t_in, t_out, pi)``; no ``t_out`` if open."""
        base = n + 3 * i
        return [base, base + 2] if np.isnan(arc[2]) else [base, base + 1, base + 2]

    def _settle_exits(self, x, lam, arcs, tf, skip=()):
        """Each arc's exit at the first zero of its ``mu``, found along the arc.

        An exit is an event, not an unknown: riding until the multiplier
        reaches zero is a simple root in time, where ``mu(t_out) = 0`` as a
        Newton row jumps whenever the riding control's active bounds change.
        Arcs with another event inside them keep their exit as an unknown.
        Returns ``(arcs, settled)``: the arcs, and the indices whose exit was set."""
        k, p = self.kernels, self.params
        span = tf - self.t0
        arcs = list(arcs)
        settled = set()
        for i in sorted(range(len(arcs)), key=lambda j: arcs[j][1]):
            if i in skip:
                continue
            c, t_in, t_out, pi = arcs[i]
            others = [a[1] for j, a in enumerate(arcs) if j != i and a[1] > t_in] \
                + [a[2] for j, a in enumerate(arcs) if j != i and not np.isnan(a[2]) and a[2] > t_in]
            horizon = min(others + [tf])
            if not np.isnan(t_out) and t_out > horizon:
                continue                      # another event inside the arc: Newton keeps t_out
            probe = list(arcs)
            # Only the arc's first moment: its start state after the jump.
            probe[i] = (c, t_in, min(t_in + 1e-6 * span, horizon), pi)
            shoot, order = self._shooting(x, probe)
            _, _, v, segs = self._rows(shoot, order, x, lam, probe, tf, jac=False)
            seg = next(sg for sg in segs if sg[0] == "boundary" and abs(sg[2] - t_in) < 1e-14 * max(1, abs(t_in)) + 1e-15)
            aid, za = seg[1], seg[4]
            where = list(k.arc_constraints(aid)).index(c)

            def mu_at(t, z):
                return float(np.atleast_1d(k.boundary_multiplier(aid, t, z, p))[where])

            n_grid = 48
            grid = np.linspace(t_in, horizon, n_grid + 1)
            z, t_prev = za, t_in
            exit_at = None
            riding = True
            for tg in grid[1:]:
                try:
                    r = k.propagate_boundary(aid, t_prev, z, tg, p, **self.options)
                    m = mu_at(tg, r["z"]) if r["status"] == 0 else np.nan
                except RuntimeError:
                    m = np.nan
                if not np.isfinite(m):
                    riding = False            # the arc cannot be ridden this far
                    break
                if m < 0:
                    lo, hi, zlo = t_prev, tg, z
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        rm = k.propagate_boundary(aid, lo, zlo, mid, p, **self.options)
                        if mu_at(mid, rm["z"]) < 0:
                            hi = mid
                        else:
                            lo, zlo = mid, rm["z"]
                        if hi - lo < 1e-14 * max(1.0, abs(hi)):
                            break
                    exit_at = 0.5 * (lo + hi)
                    break
                z, t_prev = r["z"], tg
            if exit_at is None and not riding:
                # Riding fails before mu reaches zero: leave the exit to Newton, from there.
                arcs[i] = (c, t_in, t_prev if np.isnan(t_out) else min(t_out, t_prev), pi)
                continue
            if exit_at is None:
                if horizon < tf:
                    continue                  # mu stays positive up to the next event
                exit_at = np.nan              # it rides to the end
            elif exit_at - t_in < 1e-9 * span:
                exit_at = t_in + 1e-9 * span  # it dies: the caller removes it
            arcs[i] = (c, t_in, exit_at, pi)
            settled.add(i)
        return arcs, settled

    def _project(self, x0, lam0, arcs, tf, touch=()):
        """Minimum-norm Newton onto the entry conditions, with the exits settled as events.

        The unknowns are ``lam0``, each arc's ``(t_in, pi)`` (and ``t_out``
        where it cannot be settled) and the components of the start ``x0``
        that :attr:`vary_start` frees. The exits move with them: their
        dependence enters the Jacobian through the exit rows (a Schur
        complement). Arcs in ``touch`` are contacts: zero jump, a fixed tiny
        width, and only their entry rows (the peak of ``g`` at the limit).
        Returns ``(x0, lam0, arcs, residual)``."""
        n = self.kernels.N_STATES
        s_lam = max(float(np.linalg.norm(lam0)) / np.sqrt(n), 1e-12)
        span = tf - self.t0
        vary = self._vary
        nv = len(vary)
        m = len(arcs)

        def unknowns(x, lam, arcs_):
            return np.concatenate([lam] + [[a[1], a[2], a[3]] for a in arcs_] + [x[vary]])

        def split(w, arcs_):
            lam = w[:n]
            out = [(c, w[n + 3 * i], w[n + 3 * i + 1], w[n + 3 * i + 2])
                   for i, (c, *_) in enumerate(arcs_)]
            x = x0.copy()
            x[vary] = w[len(w) - nv:]
            return x, lam, out

        def check(arcs_):
            for c, t_in, t_out, _ in arcs_:
                end = tf if np.isnan(t_out) else t_out
                if not (self.t0 < t_in < end <= tf) or (end == tf and not np.isnan(t_out)):
                    raise ShootingError("an arc left the motion or closed")

        def rows_at(x, lam, arcs_, jac):
            check(arcs_)
            shoot, order = self._shooting(x, arcs_)
            r, J, _, _ = self._rows(shoot, order, x, lam, arcs_, tf, jac=jac)
            # Which row is each arc's exit row (events: an entry has q + 1 rows, an exit one).
            exit_row, row = {}, 0
            for ev, pos in shoot.events:
                if ev == "out":
                    exit_row[order[pos]] = row
                    row += 1
                else:
                    row += shoot.arcs[pos][2] + 1
            return r, J, exit_row

        eps = 1e-7 * span

        def evaluate(x, lam, arcs_):
            arcs_ = [(a[0], a[1], a[1] + eps, 0.0) if i in touch else a for i, a in enumerate(arcs_)]
            arcs_, settled = self._settle_exits(x, lam, arcs_, tf, skip=touch)
            r, J, exit_row = rows_at(x, lam, arcs_, True)
            if nv:
                # The start's columns by differences: the shooting holds x0 fixed. The
                # exits are held too; the Schur step below moves them with the start.
                cols = []
                for i in vary:
                    h = 1e-7 * max(1.0, abs(x[i]))
                    xh = x.copy()
                    xh[i] += h
                    cols.append((rows_at(xh, lam, arcs_, False)[0] - r) / h)
                J = np.hstack([J, np.array(cols).T])
            # Eliminate the settled exits: their rows hold (mu = 0), and t_out moves with
            # the other unknowns through them.
            el = sorted(i for i in settled if i in exit_row)
            e_rows = [exit_row[i] for i in el]
            e_cols = [n + 3 * i + 1 for i in el]
            # A contact's exit follows its entry, and its jump stays zero.
            for i in touch:
                J[:, n + 3 * i] += J[:, n + 3 * i + 1]
            touch_rows = [exit_row[i] for i in touch if i in exit_row]
            keep_rows = [j for j in range(len(r)) if j not in e_rows and j not in touch_rows]
            dropped = set(e_cols) | {n + 3 * i + 1 for i in settled if np.isnan(arcs_[i][2])} \
                | {n + 3 * i + d for i in touch for d in (1, 2)}
            free = [j for j in range(J.shape[1]) if j not in dropped]
            if e_rows:
                Jmt = J[np.ix_(e_rows, e_cols)]
                Jmf = J[np.ix_(e_rows, free)]
                Jkt = J[np.ix_(keep_rows, e_cols)]
                Jred = J[np.ix_(keep_rows, free)] - Jkt @ np.linalg.lstsq(Jmt, Jmf, rcond=None)[0]
            else:
                Jred = J[np.ix_(keep_rows, free)]
            return r[keep_rows], Jred, free, arcs_

        scales = np.concatenate([np.full(n, s_lam)] + [[span, span, s_lam] for _ in range(m)]
                                + [self.start_scale * np.ones(nv)])
        try:
            r, J, free, arcs = evaluate(x0, lam0, arcs)
        except _FAILURES as exc:
            raise Rejected("propagation failed", str(exc)) from exc
        err = float(np.max(np.abs(r), initial=0.0))
        w = unknowns(x0, lam0, arcs)
        # Levenberg-Marquardt in the scaled unknowns: with a small damping it is the
        # minimum-norm Gauss-Newton step (more unknowns than rows), with a large one a
        # short gradient step, for corrections too large for the linearization.
        damping = 1e-6
        for _ in range(self.max_newton):
            if err < self.tol:
                break
            self._out_of_time()
            sc = scales[free]
            A = np.where(np.isfinite(J), J, 0.0) * sc
            merit = float(r @ r)
            improved = False
            for _ in range(16):
                if damping < 1e-9:
                    y = np.linalg.lstsq(A, -r, rcond=None)[0]
                else:
                    aug = np.vstack([A, np.sqrt(damping) * np.eye(A.shape[1])])
                    y = np.linalg.lstsq(aug, np.concatenate([-r, np.zeros(A.shape[1])]),
                                        rcond=None)[0]
                wt = w.copy()
                wt[free] += sc * y
                x_t, lam_t, arcs_t = split(wt, arcs)
                try:
                    rt, Jt, free_t, arcs_t = evaluate(x_t, lam_t, arcs_t)
                except _FAILURES:
                    damping = max(damping * 10, 1e-6)
                    continue
                if float(rt @ rt) < merit:
                    improved = True
                    damping = max(damping / 10, 1e-12)
                    break
                damping = max(damping * 10, 1e-6)
            if self.debug:
                print(f"    LM err {err:.2e} -> {float(np.max(np.abs(rt))) if improved else float('nan'):.2e}"
                      f" damping {damping:.0e} rows {np.round(r, 6)}")
            if not improved:
                raise Rejected("projection stalled", f"junction residual {err:.1e}")
            r, J, free, arcs = rt, Jt, free_t, arcs_t
            x0, lam0 = x_t, lam_t
            w = unknowns(x0, lam0, arcs)
            err = float(np.max(np.abs(r), initial=0.0))
        if err >= self.tol:
            raise Rejected("projection stalled", f"junction residual {err:.1e}")
        return x0, lam0, arcs, err

    # -- violations ------------------------------------------------------------

    def _state_at(self, shoot, seg, t):
        k, p = self.kernels, self.params
        kind, aid, ta, tb, za, flags, _ = seg
        if t <= ta:
            return za
        if kind is None:
            arc = shoot._free_arc(ta, za, flags)
            r = k.propagate(ta, za, t, p, constraint_events=False, arc=arc,
                            frozen=sorted(flags), **self.options)
        else:
            r = k.propagate_boundary(aid, ta, za, t, p, **self.options)
        if r["status"] != 0:
            raise ShootingError(f"propagation failed with status {r['status']}")
        return r["z"]

    def _violations(self, shoot, v, segs):
        """``(c, t_start, t_end, peak)`` of the earliest violated limit off its arcs, or None."""
        k = self.kernels
        tr = shoot.trajectory(v, self.n_points)
        t, g, active = tr["t"], tr["g"], tr["active"]
        tol_g = 1e-9 * np.maximum(1.0, np.max(np.abs(g), axis=0))
        found = []
        for c in range(k.N_CONSTRAINTS):
            off = ~active[:, c]
            gc = np.where(off, g[:, c], -np.inf)
            # Local peaks off the arcs, refined to their exact height.
            idx = [i for i in range(1, len(t) - 1)
                   if off[i] and gc[i] >= gc[i - 1] and gc[i] >= gc[i + 1]
                   and gc[i] > -1e-3 * max(1.0, abs(gc[i]))]
            for i in idx:
                peak, tp = self._refine_peak(shoot, segs, c, t[i - 1], t[i + 1], gc[i], t[i])
                if peak > tol_g[c]:
                    lo, hi = i, i
                    while lo > 0 and off[lo - 1] and g[lo - 1, c] > 0:
                        lo -= 1
                    while hi < len(t) - 1 and off[hi + 1] and g[hi + 1, c] > 0:
                        hi += 1
                    a, b = t[max(lo - 1, 0)], t[min(hi + 1, len(t) - 1)]
                    found.append((a, c, a, b, peak, tp))
            if off[0] and g[0, c] > tol_g[c]:
                raise Rejected("violated at the start", k.CONSTRAINT_NAMES[c])
            if off[-1] and g[-1, c] > tol_g[c]:
                # Still violated at the end: the motion ends on an arc of c.
                lo = len(t) - 1
                while lo > 0 and off[lo - 1] and g[lo - 1, c] > 0:
                    lo -= 1
                found.append((t[max(lo - 1, 0)], c, t[max(lo - 1, 0)], np.nan, g[-1, c], t[-1]))
        if not found:
            return None
        found.sort()
        _, c, a, b, peak, tp = found[0]
        return c, a, b, peak, tp

    def _negative_multiplier(self, shoot, v, arcs, order):
        """``(arc index, t)``: the first arc whose ``mu`` falls below zero, and where it
        crosses zero (the arc must end there), or None."""
        k, p = self.kernels, self.params
        tr = shoot.trajectory(v, self.n_points)
        worst = None
        for pos, i in enumerate(order):
            c = arcs[i][0]
            on = np.flatnonzero(tr["active"][:, c])
            if not len(on):
                continue
            ts, mus = [], []
            for j in on:
                aid = tr["arc"][j]
                m_ = np.atleast_1d(k.boundary_multiplier(aid, tr["t"][j], tr["z"][j], p))
                ts.append(tr["t"][j])
                mus.append(float(m_[list(k.arc_constraints(aid)).index(c)]))
            ts, mus = np.array(ts), np.array(mus)
            scale = max(1e-12, float(np.max(np.abs(mus))))
            neg = np.flatnonzero(mus < -1e-8 * scale)
            # Only inside the arc: the exit's own mu = 0 is not a break.
            neg = [j for j in neg if ts[j] < (arcs[i][2] if not np.isnan(arcs[i][2]) else np.inf)
                   - 1e-9 * (ts[-1] - ts[0] + 1e-300)]
            if not neg:
                continue
            j = neg[0]
            t_zero = ts[j - 1] + (ts[j] - ts[j - 1]) * mus[j - 1] / (mus[j - 1] - mus[j]) \
                if j > 0 else ts[0]
            if worst is None or t_zero < worst[1]:
                worst = (i, t_zero)
        return worst

    def _refine_peak(self, shoot, segs, c, ta, tb, g_mid, t_mid):
        """The largest ``g_c`` on ``[ta, tb]`` by golden section on exact states."""
        k, p = self.kernels, self.params

        def seg_of(tt):
            for s in segs:
                if s[2] <= tt <= s[3]:
                    return s
            return segs[-1]

        def g_at(tt):
            s = seg_of(tt)
            return float(k.constraint(tt, self._state_at(shoot, s, tt), p)[c])

        phi = (np.sqrt(5) - 1) / 2
        a, b = ta, tb
        x1, x2 = b - phi * (b - a), a + phi * (b - a)
        f1, f2 = g_at(x1), g_at(x2)
        for _ in range(40):
            if b - a < 1e-12 * max(1.0, abs(b)):
                break
            if f1 > f2:
                b, x2, f2 = x2, x1, f1
                x1 = b - phi * (b - a)
                f1 = g_at(x1)
            else:
                a, x1, f1 = x1, x2, f2
                x2 = a + phi * (b - a)
                f2 = g_at(x2)
        best = max((f1, x1), (f2, x2), (g_mid, t_mid))
        return best

    # -- generation ------------------------------------------------------------

    def on_level_set(self, x0, lam0):
        """``lam0`` scaled along its ray so that ``H(t0) = 0`` (free final time).

        Along ``s lam0`` the Hamiltonian ``min_u L + s lam0 . f`` is concave
        in ``s`` and starts at ``min_u L``, so there is at most one positive
        root; :class:`Rejected` when there is none."""
        k, p = self.kernels, self.params
        lam0 = np.asarray(lam0, dtype=float)
        if not self.free_time:
            return lam0

        def H(s):
            return k.hamiltonian(self.t0, np.concatenate([x0, s * lam0]), p)

        lo, hi = 0.0, 1.0
        if H(lo) <= 0:
            raise Rejected("no level set", "H(0) <= 0")
        while H(hi) > 0:
            hi *= 2
            if hi > 1e8:
                raise Rejected("no level set", "H stays positive along the ray")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if H(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-15 * hi:
                break
        return hi * lam0

    def generate(self, x0, lam0, tf=None, rng=None):
        """The exact extremal from ``x0`` nearest the costates ``lam0``, over ``[t0, tf]``.

        ``tf`` is required when the final time is free (it is then the end of
        the motion generated) and must be omitted otherwise. Raises
        :class:`Rejected` when the draw gives no valid extremal.
        """
        n = self.kernels.N_STATES
        x0 = np.asarray(x0, dtype=float).reshape(n)
        if self.free_time:
            if tf is None:
                raise ValueError("the final time is free: give tf")
            tf = float(tf)
        else:
            if tf is not None:
                raise ValueError("the final time is fixed by the problem")
            tf = self.fixed_tf
        self._rng = np.random.default_rng(rng)
        self._deadline = (time.perf_counter() + self.max_seconds if self.max_seconds
                          else np.inf)
        lam0 = self.on_level_set(x0, lam0)
        return self._grow(x0, lam0, [], tf)

    def _grow(self, x0, lam0, arcs, tf):
        """From ``(x0, lam0)`` and the arcs ``(c, t_in, t_out, pi)`` so far, the
        extremal with every violated limit made an arc (or truncated before it)."""
        k = self.kernels
        span = tf - self.t0
        tf_cap = self.t0 + self.extend * span
        valid = None                 # (x0, lam0, arcs, err, t_violation): valid up to t_violation
        for _ in range(self.max_rounds + 12):
            self._out_of_time()
            try:
                x0, lam0, arcs, err = self._project(x0, lam0, arcs, tf)
            except Rejected:
                if valid is None or not self.truncate:
                    raise
                return self._truncated(*valid, why="projection")
            dead = [i for i, a in enumerate(arcs) if a[3] < -1e-9 * max(1.0, abs(a[3]))
                    or a[2] - a[1] < 1e-9 * span]
            if dead:
                arcs = [a for i, a in enumerate(arcs) if i not in dead]
                continue
            try:
                shoot, order = self._shooting(x0, arcs)
                _, _, v, segs = self._rows(shoot, order, x0, lam0, arcs, tf, jac=False)
                broken = self._negative_multiplier(shoot, v, arcs, order)
            except _FAILURES as exc:
                if valid is None or not self.truncate:
                    raise Rejected("propagation failed", str(exc)) from exc
                return self._truncated(*valid, why="propagation failed")
            if broken is not None:
                # mu crosses zero inside the arc: it ends there (the free arc after is checked next).
                i, t_zero = broken
                c, t_in, _, pi = arcs[i]
                if t_zero <= t_in + 1e-6 * span:
                    arcs.pop(i)
                else:
                    arcs[i] = (c, t_in, t_zero, pi)
                continue
            try:
                hit = self._violations(shoot, v, segs)
            except _FAILURES as exc:
                if valid is None or not self.truncate:
                    raise Rejected("propagation failed", str(exc)) from exc
                return self._truncated(*valid, why="propagation failed")
            if hit is None:
                return self._finish(x0, lam0, arcs, tf, err)
            c, a, b, peak, tp = hit
            valid = (x0, lam0, list(arcs), err, a)
            if k.CONSTRAINT_NAMES[c] in self.unsupported:
                if self.truncate:
                    return self._truncated(*valid, why="unsupported limit")
                raise Rejected("unsupported limit", k.CONSTRAINT_NAMES[c])
            if c in self._offsets and self.regular:
                if np.isnan(b):
                    # Still rising at the end: its peak, where the arc is born, lies
                    # beyond. The duration is free, so extend the motion past it.
                    if tf < tf_cap:
                        tf = min(1.25 * tf - 0.25 * self.t0, tf_cap)
                        span = tf - self.t0
                        continue
                    if self.truncate:
                        return self._truncated(*valid, why="rising at the end")
                    raise Rejected("violated at the end", k.CONSTRAINT_NAMES[c])
                try:
                    x0, lam0, arcs = self._birth(x0, lam0, arcs, tf, c, peak, tp, rng=self._rng)
                except Rejected as exc:
                    if self.truncate:
                        return self._truncated(*valid, why="birth: " + exc.reason)
                    raise
                continue
            near = 1e-3 * span
            reach = lambda arc: tf if np.isnan(arc[2]) else arc[2]
            same = [i for i, arc in enumerate(arcs)
                    if arc[0] == c and arc[1] - near <= (tf if np.isnan(b) else b)
                    and reach(arc) + near >= a]
            if same:
                i = same[0]
                cc, t_in, t_out, pi = arcs[i]
                t_out = np.nan if np.isnan(b) or np.isnan(t_out) else max(t_out, b)
                arcs[i] = (cc, min(t_in, a), t_out, pi)
            else:
                if not np.isnan(b) and b - a < 1e-6 * span:
                    a, b = tp - 1e-4 * span, tp + 1e-4 * span
                a = max(a, self.t0 + 1e-4 * span)
                if not np.isnan(b):
                    b = min(max(b, a + 1e-4 * span), tf - 1e-4 * span)
                arcs.append((c, a, b, 0.0))
            arcs.sort(key=lambda arc: arc[1])
        if valid is not None and self.truncate:
            return self._truncated(*valid, why="too many rounds")
        raise Rejected("too many rounds", f"{len(arcs)} arcs")

    # -- anchored at a junction ----------------------------------------------------

    def _entry_rows(self, c, t, z, gap=False):
        """The conditions for ``(t, z)`` to be an entry onto limit ``c``: ``g = 0``,
        the arrival (tangential, or ``H`` continuous when a control may jump) and
        ``H = 0`` when the final time is free.

        With ``gap`` the arrival row is ``H(u_b) - H(u*) >= 0``, which vanishes
        exactly when the free control already rides the limit. Unlike
        ``dg/dt`` it still moves with the costates while that control is
        saturated, where ``dg/dt`` is flat."""
        k, p = self.kernels, self.params
        rows = [float(k.constraint(t, z, p)[c])]
        if self.regular and gap:
            d = float(k.boundary_hamiltonian(c, t, z, p) - k.hamiltonian(t, z, p))
            rows.append(max(d, 0.0))
        elif self.regular:
            rows.append(float(np.atleast_1d(k.constraint_rate(t, z, p))[c]))
        else:
            rows.append(float(k.hamiltonian(t, z, p) - k.boundary_hamiltonian(c, t, z, p)))
        if self.free_time:
            rows.append(float(k.hamiltonian(t, z, p)))
        return np.array(rows)

    def _construct_anchor(self, c, t, z, iterations=60):
        """An entry onto limit ``c`` built in steps, for a continuous control law.

        The state is moved onto the limit; the control that rides it there (at
        the drawn costates) is the target, which must lie strictly inside its
        bounds; then costates are found whose free control *is* that target, on
        ``H = 0``. A free control that rides the limit has ``dg/dt = 0``: the
        entry is tangential. Each step is smooth with simple roots, because
        nothing is saturated."""
        k, p, n = self.kernels, self.params, self.kernels.N_STATES
        x, lam = z[:n].copy(), z[n:].copy()
        # 1. The state onto g_c = 0, the other limits held where the draw has them (pulled
        # inside if close), by Newton in the states anchor_states allows.
        movable = list(range(n) if self.anchor_states is None else self.anchor_states)
        g_draw = np.asarray(k.constraint(t, np.r_[x, lam], p), dtype=float)
        hold = np.minimum(g_draw, -0.05 * np.abs(g_draw))

        def all_rows(xx):
            g = np.asarray(k.constraint(t, np.r_[xx, lam], p), dtype=float)
            r = g - hold
            r[c] = g[c]
            return r

        # A limit moved by the same combination of states as c (its opposite, for a
        # torque) cannot be held: drop the rows whose gradient is parallel to c's.
        r0 = all_rows(x)
        G = np.zeros((len(r0), len(movable)))
        for jj, i in enumerate(movable):
            h = 1e-7 * max(1.0, abs(x[i]))
            xh = x.copy()
            xh[i] += h
            G[:, jj] = (all_rows(xh) - r0) / h
        gc = G[c] / max(float(np.linalg.norm(G[c])), 1e-300)
        keep = [j for j in range(len(r0)) if j == c or
                abs(float(G[j] @ gc)) < 0.999 * max(float(np.linalg.norm(G[j])), 1e-300)]

        def state_rows(xx):
            return all_rows(xx)[keep]

        r = state_rows(x)
        for _ in range(iterations):
            if np.max(np.abs(r)) < self.tol:
                break
            J = np.zeros((len(r), len(movable)))
            for jj, i in enumerate(movable):
                h = 1e-7 * max(1.0, abs(x[i]))
                xh = x.copy()
                xh[i] += h
                J[:, jj] = (state_rows(xh) - r) / h
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
            x[movable] += step
            r = state_rows(x)
        else:
            raise Rejected("no anchor", "the state does not reach the limit")
        # 2. The riding control there is the target; it must be inside its bounds. Large
        # costates push it to them, so the draw is shrunk along its ray until it is not,
        # and until the free control is inside too: where it is saturated it does not
        # move with the costates, and step 3 could not steer it (step 3 then restores
        # H = 0).
        def inside(u, lo, hi):
            margin = 1e-3 * np.maximum(hi - lo, 1e-12)
            return np.all(u > lo + margin) and np.all(u < hi - margin)

        for shrink in 10.0 ** -np.arange(0, 10, 0.5):
            zz = np.r_[x, shrink * lam]
            try:
                target = np.asarray(k.boundary_control(c, t, zz, p), dtype=float)
                free = np.asarray(k.control(t, zz, p), dtype=float)
            except _FAILURES:
                continue
            lo, hi = (np.asarray(b, dtype=float) for b in k.control_bounds(t, zz, p))
            if inside(target, lo, hi) and inside(free, lo, hi):
                lam = shrink * lam
                break
        else:
            raise Rejected("no anchor", "riding there needs a control at its bound")

        # 3. Costates whose free control is the target, on H = 0.
        def rows(l, bend=None):
            zl = np.r_[x, l]
            r = np.asarray(k.control(t, zl, p), dtype=float) - target
            if self.free_time:
                r = np.r_[r, k.hamiltonian(t, zl, p)]
            if bend is not None:
                r = np.r_[r, (self._curvature(c, t, zl) - bend) / bend_scale]
            return r

        def solve(lam, bend=None):
            scale = max(float(np.sqrt(np.mean(lam ** 2))), 1e-6)
            r = rows(lam, bend)
            for _ in range(iterations):
                if np.max(np.abs(r)) < self.tol:
                    return lam
                J = np.zeros((len(r), n))
                for i in range(n):
                    h = 1e-7 * max(scale, abs(lam[i]))
                    lh = lam.copy()
                    lh[i] += h
                    J[:, i] = (rows(lh, bend) - r) / h
                step = np.linalg.lstsq(J, -r, rcond=None)[0]
                for _ in range(20):
                    rt = rows(lam + step, bend)
                    if rt @ rt < r @ r:
                        lam, r = lam + step, rt
                        break
                    step = step / 2
                else:
                    break
            raise Rejected("no anchor", "no costates give that control on H = 0")

        bend_scale = 1.0
        lam = solve(lam)
        # 4. A peak, not a trough. With g = 0 and dg/dt = 0 the free motion meets the
        # limit from inside only if d2g/dt2 < 0; at a trough it arrives from outside.
        # The costates left free by step 3 (all but m + 1 of them) can bend it: the
        # curvature is mirrored, so the entry keeps its scale.
        bend = self._curvature(c, t, np.r_[x, lam])
        if bend >= 0:
            bend_scale = max(abs(bend), 1e-6)
            lam = solve(lam, -bend_scale)
        return np.r_[x, lam]

    def _curvature(self, c, t, z):
        """``d2g/dt2`` of limit ``c`` along the free flow at ``(t, z)``."""
        k, p = self.kernels, self.params
        f = np.asarray(k.rhs(t, z, p), dtype=float)
        h = 1e-6 / max(float(np.linalg.norm(f)), 1e-12) * max(1.0, float(np.linalg.norm(z)))
        up = np.atleast_1d(k.constraint_rate(t + h, z + h * f, p))[c]
        down = np.atleast_1d(k.constraint_rate(t - h, z - h * f, p))[c]
        return float((up - down) / (2 * h))

    def anchor(self, c, z, t=None, iterations=200):
        """``z = (x, lam)`` moved (minimum norm) onto the entry conditions of limit ``c``.

        These conditions involve one point only, so they need no integration.
        Raises :class:`Rejected` if they cannot be met, if another limit is
        broken there, or if the free motion would not arrive from inside."""
        k, p = self.kernels, self.params
        t = self.t0 if t is None else float(t)
        z = np.asarray(z, dtype=float).copy()
        n = k.N_STATES
        built = False
        if self.regular:
            try:
                z, built = self._construct_anchor(c, t, z), True
            except Rejected:
                pass                      # fall back to projecting the draw itself
        # One scale for the states and one for the costates: per-component scales make
        # small components (often the ones the limit depends on) too dear to move.
        scale = np.r_[np.full(n, max(float(np.sqrt(np.mean(z[:n] ** 2))), 1e-3)),
                      np.full(n, max(float(np.sqrt(np.mean(z[n:] ** 2))), 1e-3))]

        def rows(zz, gap):
            try:
                r = self._entry_rows(c, t, zz, gap)
            except _FAILURES:
                return None
            return r if np.all(np.isfinite(r)) else None

        # First with the gap (it moves with saturated controls), then with dg/dt itself.
        phases = ((True, 1e-12), (False, self.tol)) if self.regular and not built else ((False, self.tol),)
        for gap, tol in phases:
            r = rows(z, gap)
            if r is None:
                raise Rejected("no anchor", "the limit cannot be ridden at the draw")
            damping = 1e-6
            for _ in range(iterations):
                if np.max(np.abs(r)) < tol:
                    break
                J = np.zeros((len(r), len(z)))
                for i in range(len(z)):
                    h = 1e-7 * max(1.0, abs(z[i]))
                    zh = z.copy()
                    zh[i] += h
                    rh = rows(zh, gap)
                    J[:, i] = 0.0 if rh is None else (rh - r) / h
                A = J * scale
                for _ in range(20):
                    aug = np.vstack([A, np.sqrt(damping) * np.eye(len(z))])
                    y = np.linalg.lstsq(aug, np.r_[-r, np.zeros(len(z))], rcond=None)[0]
                    zt = z + scale * y
                    rt = rows(zt, gap)
                    if rt is not None and rt @ rt < r @ r:
                        z, r, damping = zt, rt, max(damping / 10, 1e-12)
                        break
                    damping *= 10
                else:
                    raise Rejected("no anchor", f"entry residual {np.max(np.abs(r)):.1e}")
            else:
                raise Rejected("no anchor", f"entry residual {np.max(np.abs(r)):.1e}")
        g = k.constraint(t, z, p)
        others = [j for j in range(k.N_CONSTRAINTS) if j != c]
        if others and np.max(g[others]) > -1e-9:
            raise Rejected("anchor breaks another limit")
        # The free motion must reach the limit from inside: just before, g < 0.
        h = 1e-4
        r = k.propagate(t, z, t - h, p, constraint_events=False, **self.options)
        if r["status"] != 0 or k.constraint(t - h, r["z"], p)[c] >= 0:
            raise Rejected("anchor not reached from inside")
        return z

    def generate_anchored(self, c, z, before, after, rng=None):
        """An extremal through an entry onto limit ``c`` at ``z = (x, lam)`` (anchored first).

        From the anchor the free equations integrate backward for ``before``,
        which gives the start (cut short at the latest broken limit, so the
        start is wherever the motion can begin), and the arc and the rest of
        the motion run forward for ``after``, its jump chosen for an arc of a
        length drawn from :attr:`arc_fraction`. The entry is exact by
        construction, so the saturated controls and the out-of-reach peaks that
        stop :meth:`generate` do not arise there; later limits are handled as
        there. Requires a free final time."""
        k, p, n = self.kernels, self.params, self.kernels.N_STATES
        if not self.free_time:
            raise NotImplementedError("anchored generation needs a free final time")
        self._rng = np.random.default_rng(rng)
        self._deadline = (time.perf_counter() + self.max_seconds if self.max_seconds
                          else np.inf)
        za = self.anchor(c, z)
        t_a = self.t0 + float(before)
        # Backward to the start. Events are not used: integrating backward a limit is
        # crossed the other way. Instead the stretch is sampled, and the motion starts
        # just after the last point where a limit is broken.
        eps = 1e-7 * before
        r = k.propagate(t_a, za, t_a - eps, p, constraint_events=False, **self.options)
        rb = k.propagate(t_a - eps, r["z"], self.t0, p, constraint_events=False, **self.options)
        if rb["status"] != 0:
            raise Rejected("propagation failed", f"status {rb['status']}")
        ts = np.linspace(self.t0, t_a - eps, 400)
        tr = k.trajectory(self.t0, rb["z"], t_a - eps, p, t_eval=ts, constraint_events=False,
                          **self.options)
        tol_g = 1e-9 * np.maximum(1.0, np.abs(tr["g"]).max(axis=0))
        broken = np.flatnonzero((tr["g"] > tol_g).any(axis=1))
        t_start = self.t0
        if len(broken):
            lo, hi = ts[broken[-1]], ts[min(broken[-1] + 1, len(ts) - 1)]
            for _ in range(50):                      # the last crossing, by bisection
                mid = 0.5 * (lo + hi)
                zm = k.propagate(t_a - eps, r["z"], mid, p, constraint_events=False,
                                 **self.options)["z"]
                if (k.constraint(mid, zm, p) > tol_g).any():
                    lo = mid
                else:
                    hi = mid
            t_start = hi + 1e-3 * (t_a - hi)
            if t_a - t_start < 1e-3 * before:
                raise Rejected("violated just before the anchor")
            rb = k.propagate(t_a - eps, r["z"], t_start, p, constraint_events=False, **self.options)
        # Shift time so the motion starts at t0 (the problem does not depend on time).
        shift = t_start - self.t0
        x0, lam0 = rb["z"][:n], rb["z"][n:]
        t_in = t_a - shift
        tf = t_in + float(after)
        arcs = [(c, t_in, np.nan, 0.0)]
        span = tf - self.t0
        target = self._rng.uniform(*self.arc_fraction) * span
        pi = self._jump_for_length(x0, lam0, arcs, tf, 0, min(target, 0.9 * (tf - t_in)))
        arcs, _ = self._settle_exits(x0, lam0, [(c, t_in, t_in + 1e-6 * span, pi)], tf)
        ext = self._grow(x0, lam0, arcs, tf)
        # The anchored arc, if it survived the rest of the motion (later limits can close it).
        kept = [val for (kind, cc), val in zip([a for a in ext.arcs if a[0] == "enter"], ext.values)
                if cc == c and abs(val[0] - t_in) < 1e-6 * span]
        ext.checks["anchored"] = k.CONSTRAINT_NAMES[c] if kept else None
        return ext

    def _birth(self, x0, lam0, arcs, tf, c, peak, tp, rng=None):
        """Arcs with a new one of limit ``c``, whose free trajectory peaks above it at ``tp``.

        First the peak is brought down to the limit: with the limit raised to
        just below the peak, the contact (a touch, zero jump) is nearly
        satisfied already, and lowering the raise to zero in adaptive steps,
        projecting at each, moves the costates (and the start, if free) until
        the free trajectory peaks exactly on the limit. Then the arc: its jump
        ``pi`` (``mu`` at the entry) is free going forward, so it is chosen for
        an arc of a sampled length, and the arc ends where ``mu`` reaches zero."""
        span = tf - self.t0
        i_off = self._offsets[c]
        base = self.params.copy()
        o0 = float(peak) * (1 - 1e-4)
        arcs = sorted(arcs + [(c, tp, tp + 1e-7 * span, 0.0)], key=lambda a: a[1])
        j = next(i for i, a in enumerate(arcs) if a[0] == c and a[1] == tp)
        s, ds = 0.0, 0.05
        x_s, lam_s, arcs_s = x0, lam0, arcs
        try:
            for _ in range(self.birth_steps):
                self._out_of_time()
                f = min(s + ds, 1.0) if s > 0 else 0.0
                self.params = base.copy()
                self.params[i_off] = base[i_off] + (1 - f) * o0
                try:
                    x_t, lam_t, arcs_t, _ = self._project(x_s, lam_s, arcs_s, tf, touch={j})
                except Rejected:
                    if s == 0.0 or ds < 1e-5:
                        raise
                    ds /= 2
                    continue
                x_s, lam_s, arcs_s = x_t, lam_t, arcs_t
                if f >= 1.0:
                    break
                s = f if f > 0 else 1e-12
                ds = min(ds * 1.5, 0.5)
            else:
                raise Rejected("birth too slow", f"offset fraction {s:.3f}")
        finally:
            self.params = base
        rng = np.random.default_rng() if rng is None else rng
        target = rng.uniform(*self.arc_fraction) * span
        pi = self._jump_for_length(x_s, lam_s, arcs_s, tf, j, target)
        cc, t_in, _, _ = arcs_s[j]
        arcs_s[j] = (cc, t_in, t_in + 1e-6 * span, pi)
        arcs_s, _ = self._settle_exits(x_s, lam_s, arcs_s, tf)
        return x_s, lam_s, arcs_s

    def _jump_for_length(self, x, lam, arcs, tf, j, target):
        """The jump ``pi`` of arc ``j`` that makes it last about ``target``, by bisection."""
        span = tf - self.t0
        c, t_in, _, _ = arcs[j]

        def length(pi):
            trial = list(arcs)
            trial[j] = (c, t_in, t_in + 1e-6 * span, pi)
            try:
                out, settled = self._settle_exits(x, lam, trial, tf)
            except _FAILURES:
                return None
            if j not in settled:
                return None                     # the arc cannot be ridden that long
            t_out = out[j][2]
            return (tf - t_in) if np.isnan(t_out) else t_out - t_in

        scale = max(1e-12, float(np.max(np.abs(lam))))
        lo, hi = 0.0, 1e-3 * scale
        L = length(hi)
        for _ in range(60):
            if L is None or L >= target:
                break
            lo, hi = hi, 2 * hi
            L = length(hi)
        if L is None:
            hi = lo if lo > 0 else hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            Lm = length(mid)
            if Lm is None or Lm > target:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-6 * hi:
                break
        return lo if lo > 0 else hi

    def _truncated(self, x0, lam0, arcs, err, t_violation, why=""):
        """The valid extremal ``(lam0, arcs)`` ended just before its first violation.

        Nothing before ``t_violation`` changes, so its junction conditions still
        hold; an arc still ridden there stays open at the new end."""
        span = t_violation - self.t0
        if span <= 1e-6:
            raise Rejected("violated at the start")
        tf = t_violation - 1e-6 * span
        kept = []
        for c, t_in, t_out, pi in arcs:
            if t_in >= tf:
                continue
            kept.append((c, t_in, np.nan if np.isnan(t_out) or t_out >= tf else t_out, pi))
        if not self.free_time:
            raise Rejected("cannot truncate", "the final time is fixed")
        ext = self._finish(x0, lam0, kept, tf, err)
        ext.checks["truncated"] = why or True
        return ext

    def _finish(self, x0, lam0, arcs, tf, err):
        n = self.kernels.N_STATES
        shoot, order = self._shooting(x0, arcs)
        v = self._pack(shoot, order, lam0, arcs, tf)
        _, _, segs = shoot._evaluate(v, jac=False, record=True)
        zf = segs[-1][-1]["z"]
        cost = float(sum(r["cost"] for *_, r in segs))
        entries = tuple((kind, int(c)) for kind, c in shoot.entries)
        values = tuple((arcs[i][1], arcs[i][2], arcs[i][3]) for i in order)
        ext = Extremal(x0=x0.copy(), lam0=np.asarray(lam0, float).copy(), t0=self.t0, tf=tf,
                       xf=zf[:n].copy(), lamf=zf[n:].copy(), arcs=entries, values=values,
                       cost=cost, junction_residual=err)
        ext.checks = self.check(ext)
        if ext.checks["multiplier_min"] < -1e-8 or ext.checks["bound_violation"] > 1e-8:
            raise Rejected("multiplier or bound", str(ext.checks))
        return ext

    # -- checks ----------------------------------------------------------------

    def shooting_of(self, ext):
        """``(MultiArcShooting, v)`` of an extremal, on this generator's problem."""
        arcs = [(c, a, b, pi) for (c, (a, b, pi)) in
                zip([c for e, c in ext.arcs if e == "enter"], ext.values)]
        shoot, order = self._shooting(ext.x0, sorted(arcs, key=lambda a: a[1]))
        arcs_sorted = sorted(arcs, key=lambda a: a[1])
        return shoot, self._pack(shoot, order, ext.lam0, arcs_sorted, ext.tf)

    def check(self, ext, n_points=None):
        """What a solution must satisfy beyond its junction rows.

        ``violation``: the largest ``g`` off its arcs; ``multiplier_min``: the
        smallest ``mu`` on them (``>= 0``); ``pi_min``: the smallest jump
        (``>= 0``); ``bound_violation``: how far a riding control leaves its
        bounds; ``hamiltonian``: the spread of ``H`` along the motion (zero
        when the final time is free)."""
        shoot, v = self.shooting_of(ext)
        d = shoot.diagnostics(v, n_points or 401)
        mus = [lo for lo, _ in d["boundary_mu"]]
        pis = [float(np.min(pi)) for pi in d["boundary_pi"]]
        tr = shoot.trajectory(v, n_points or 401)
        k, p = self.kernels, self.params
        H = []
        for ti, zi, aid in zip(tr["t"], tr["z"], tr["arc"]):
            H.append(k.hamiltonian(ti, zi, p) if aid < 0 else k.boundary_hamiltonian(aid, ti, zi, p))
        return {"violation": float(np.max(d["free_arc_violation"], initial=0.0)),
                "multiplier_min": float(min(mus, default=0.0)),
                "pi_min": float(min(pis, default=0.0)),
                "bound_violation": float(max(d["boundary_bound_violation"], default=0.0)),
                "hamiltonian": float(np.ptp(H)) if H else 0.0}

    def conjugate_points(self, ext, n_times=12):
        """Signs of ``det dx(t)/d(lam0, t)`` along the free arc after the last junction.

        Along an extremal family the map from the costates (on ``H = 0``,
        with ``t`` when the final time is free) to the state reached is
        nonsingular until a conjugate point, where its determinant changes
        sign: past it the extremal is no longer even locally optimal. The
        junction conditions are held (the family keeps the structure).
        Returns ``(times, signs)``; a valid sample has one sign throughout.
        """
        n = self.kernels.N_STATES
        if not self.free_time:
            raise NotImplementedError("conjugate points are checked for a free final time")
        if not self.regular:
            # With a bang-bang control the state reached does not move with lam0 until a
            # switch moves: the determinant is zero there, and the test needs another form.
            raise NotImplementedError("conjugate points are checked when H is regular")
        enters = [c for e, c in ext.arcs if e == "enter"]
        arcs = sorted([(c, a, b, pi) for c, (a, b, pi) in zip(enters, ext.values)],
                      key=lambda a: a[1])
        if any(np.isnan(a[2]) for a in arcs):
            return np.zeros(0), np.zeros(0)      # it ends on an arc: no free arc after it
        start = max([a[2] for a in arcs], default=self.t0)
        times = np.linspace(start + (ext.tf - start) / n_times, ext.tf, n_times)
        signs = []
        for t in times:
            shoot, order = self._shooting(ext.x0, arcs)
            R, Jrows, v, _ = self._rows(shoot, order, ext.x0, ext.lam0, arcs, t)
            _, J = shoot.residual_jacobian(v)
            # dx(t)/dv: the terminal state rows of the residual are x(tf) - x_f.
            cols = shoot.n_free + np.arange(n)
            Dx = np.zeros((n, n + 3 * len(arcs)))
            Dx[:, :n] = J[:n, cols]
            for pos, i in enumerate(order):
                Dx[:, self._arc_columns(n, i, arcs[i])] = J[:n, shoot._arc_slices[pos]]
            if Jrows.shape[0]:
                null = _null_space(Jrows)
            else:
                null = np.eye(Jrows.shape[1])
            M = Dx @ null
            if self.free_time:
                zt = shoot._evaluate(v, jac=False)[2][-1][-1]
                M = np.hstack([M, zt["dz_dt1"][:n, None]])
            sign, _ = np.linalg.slogdet(M) if M.shape[0] == M.shape[1] else (np.nan, 0)
            signs.append(sign)
        return times, np.array(signs)


def _null_space(A, rtol=1e-10):
    u, s, vt = np.linalg.svd(A)
    rank = int(np.sum(s > rtol * max(s[0], 1e-300))) if len(s) else 0
    return vt[rank:].T
