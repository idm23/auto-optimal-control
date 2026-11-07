"""From a costate distribution to the exact extremal, with no other input.

:func:`solve` (or :class:`Homotopy`, to keep the compiled kernels for several
solves) takes the problem, its parameter values and a distribution of
initial costates (and of ``t_f`` when it is free). Everything else is found:

0. **Check the statement.** :func:`~auto_optimal_control.policy.analyze`
   reads whether the optimal motion is unique (it is not when some
   controlled part of the problem is in no cost term, as in minimum time
   on several axes). If it is not, a policy chooses among the equally good
   motions: by default a small secondary cost, with a warning (see
   :mod:`~auto_optimal_control.policy` and ``Homotopy(policy=...)``).

1. **Regularize.** :func:`regularize` adds a smoothing term
   ``eps ((u - mid) / half)^2 / 2`` to the running cost for each bang-bang
   control (its law becomes clipped-linear), and for each path constraint
   an exterior penalty ``rho / 2 max(0, g)^2`` and an offset (the limit
   becomes ``g - off <= 0``), all runtime parameters.
2. **Sample.** Single shooting from random costates on the regularized
   problem, at ``eps = 1``, with every limit off.
   A free final time with separate components needs no draws: each follows its
   costates from a long time down to its own minimum time, and they share ``t_f``.
3. **The active-set route** (problems with path constraints). Exact arcs on
   the smoothed problem, built up as the limits are tightened:

   - each limit's offset starts just above the unconstrained solution and
     falls to zero in adaptive steps (a secant predictor starts each solve);
   - when a free arc starts to violate a limit, the structure changes
     there: a newborn touch point (order 2) or short boundary arc (order
     1) with zero jump; a touch point violated beside it grows into a
     boundary arc or splits into two; neighbouring arcs of one limit merge;
     an arc that shrinks to nothing, or a touch point whose jump turns
     negative, is retired;
   - then ``eps`` falls, in decades, with the structure following, and the
     *exact* problem is tried with the current structure at every step.

   Arcs are born where a limit first comes into play, small and with small
   jumps, which is where shooting with exact arcs is well conditioned. The
   route ends at the first exact solution accepted (step 6). It goes first
   unless the exact problem may have singular arcs (which only the penalty
   route reads); each route is the other's fallback, and the active-set
   route has a time budget, ``Homotopy.active_set_seconds``.
4. **The penalty route.** When a limit is first violated, ``eps`` is held
   while its penalty weight ramps up, in decades, from ``1e-6 rho_ref`` to
   ``1e4 rho_ref`` (``rho_ref`` makes the penalty of the violation it saw as
   large as the cost); then ``eps`` falls, in decades, with the penalties
   held. A ramp that stalls is held where it got to. The steps adapt: they
   grow after a success and halve after a failure, and a secant predictor
   starts each solve.
5. **Read the structure and jump** (penalty route). After each step the
   regularized solution is read: the intervals where each limit is violated
   become boundary arcs (or touch points, for order 2), and stretches where
   a bang-bang control stays strictly between its bounds become singular
   arcs (or, when short, the ramps of ordinary switches). Each reading
   gives candidate structures for
   :class:`~auto_optimal_control.shooting.MultiArcShooting` on the *exact*
   problem, seeded from the regularized costates, junction times and
   fitted jump multipliers, plus a few seeds with jittered times. This is
   the route that finds singular arcs.
6. **Accept.** A candidate is accepted when the exact solve converges and
   passes the necessary conditions it can check (no violation on free arcs,
   controls within bounds on special arcs, ``mu >= 0`` on boundary arcs,
   ``nu >= 0`` at touch points, Legendre-Clebsch, the held bounds agreeing
   with the switching function). Among the candidates accepted at one step,
   the cheapest wins.

Bang-bang switches need no reading at all: on free arcs they are events of
the integration. Without bang-bang controls or path constraints the
regularized problem is the exact one and step 2 is the whole solve.

When no candidate is accepted (a limit of order 3 or more, a structure outside
what :class:`MultiArcShooting` represents, a search out of time),
:func:`solve` returns the last regularized solution with ``exact = False``. A
fixed end state that already violates a limit is reported the same way, at once.
"""

import itertools
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from auto_optimal_control.build import build_kernels
from auto_optimal_control.policy import Lexicographic, SecondaryCost, analyze, resolve
from auto_optimal_control.pontryagin import derive
from auto_optimal_control.problem import Box, OptimalControlProblem, PathConstraint
from auto_optimal_control.shooting import MultiArcShooting, ShootingError, SingleShooting

# The path of the continuation is followed to 1e-6: a rough Jacobian steers Newton as well.
PATH_JACOBIAN_RTOL = 1e-7
SEED_ACCEPT = 100.0                # a seed is a guess: how far above tol a stalled Newton may stop

EPS = "eps_reg"
RHO = "rho_reg_"
OFF = "off_reg_"
_FAILURES = (ShootingError, RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError)


# =============================================================================
# The regularized and exact problems
# =============================================================================

def bang_bang_controls(problem):
    """The controls the Hamiltonian is linear in, on a :class:`Box` (the bang-bang ones)."""
    return tuple(sw.control for sw in derive(problem).switching_functions)


def _copy(problem, *, name, running_cost, params, path_constraints):
    return OptimalControlProblem(
        name=name, time=problem.time, states=problem.states, controls=problem.controls,
        dynamics=problem.dynamics, running_cost=running_cost,
        terminal_cost=problem.terminal_cost, initial_state=dict(problem.initial_state),
        terminal_constraints=problem.terminal_constraints, initial_time=problem.initial_time,
        final_time=problem.final_time, params=params, control_sets=problem.control_sets,
        path_constraints=path_constraints)


def regularize(problem, bang_bang=None):
    """The problem with a smoothing weight, and a penalty weight and an offset per path constraint.

    Each bang-bang control ``u`` in ``[lo, hi]`` gets
    ``eps ((u - mid) / half)^2 / 2`` in the running cost (``mid`` and
    ``half`` the middle and half-width of its range), so its law becomes
    ``clip(mid - half^2 S / eps)``. Each path constraint ``g <= 0`` gets the
    exterior penalty weight ``rho_reg_<name>`` and becomes
    ``g - off_reg_<name> <= 0``: an offset loosens the limit, and tightening
    it back to zero is how :class:`Homotopy` brings a limit into play. All
    are runtime parameters, appended to ``problem.params`` in that order
    (``eps_reg`` first, when there are bang-bang controls, then the weights,
    then the offsets). Penalty weights the problem gives itself are replaced.
    ``bang_bang`` defaults to :func:`bang_bang_controls`.

    Returns ``(problem, eps, rhos)``: the regularized problem, the ``eps``
    symbol (``None`` without bang-bang controls) and the ``rho`` symbols.
    """
    if bang_bang is None:
        bang_bang = bang_bang_controls(problem)
    params, cost, eps = list(problem.params), problem.running_cost, None
    if bang_bang:
        eps = sp.Symbol(EPS, positive=True)
        params.append(eps)
        boxes = {cs.control: cs for cs in problem.control_sets if isinstance(cs, Box)}
        for u in bang_bang:
            lo, hi = boxes[u].lower, boxes[u].upper
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            cost = cost + eps * ((u - mid) / half) ** 2 / 2
    rhos = tuple(sp.Symbol(RHO + c.name, nonnegative=True) for c in problem.path_constraints)
    offs = tuple(sp.Symbol(OFF + c.name, real=True) for c in problem.path_constraints)
    limits = [PathConstraint(c.expr - o, c.name, penalty=r)
              for c, r, o in zip(problem.path_constraints, rhos, offs)]
    reg = _copy(problem, name=(problem.name or "problem") + "_regularized", running_cost=cost,
                params=params + list(rhos) + list(offs), path_constraints=limits)
    return reg, eps, rhos


def exact(problem):
    """The problem with its path-constraint penalties removed."""
    return _copy(problem, name=problem.name or "problem", running_cost=problem.running_cost,
                 params=problem.params,
                 path_constraints=[PathConstraint(c.expr, c.name)
                                   for c in problem.path_constraints])


# =============================================================================
# Results
# =============================================================================

@dataclass
class Step:
    """One continuation step: the weights, the residual and what was tried.

    ``s`` is how many decades ``eps`` has fallen from its start (0 without
    bang-bang controls), ``rho`` the penalty weights, ``candidates`` how many
    exact solves were tried from this step's reading, ``accepted`` whether
    one passed. On the active-set route ``structure`` is the set of arcs
    the smoothed solution has at that step.
    """

    s: float
    eps: float
    rho: np.ndarray
    residual: float
    candidates: int = 0
    accepted: bool = False
    structure: tuple = ()


@dataclass
class HomotopyResult:
    """The result of :meth:`Homotopy.solve`.

    ``exact`` says whether ``solution`` solves the exact problem (with
    ``shooting``, a :class:`SingleShooting` or :class:`MultiArcShooting` on
    the exact kernels) or is the last regularized solution (``shooting`` on
    the regularized kernels, at ``params``). ``arcs`` is the structure found,
    as :class:`MultiArcShooting` entries (empty: free arcs only, bang-bang
    switches included). ``objective`` is the exact cost ``phi + int L`` of
    an exact solution. ``steps`` logs the continuation, ``tries`` the random
    draws of the first solve, and ``message`` says how it ended. ``policy``
    describes the policy applied to a degenerate problem (empty: solved as
    stated); with :class:`~auto_optimal_control.policy.Lexicographic`,
    ``stages`` holds the minimum-time result of each component.
    """

    solution: object
    shooting: object
    arcs: tuple
    exact: bool
    objective: float
    params: np.ndarray
    steps: list = field(default_factory=list)
    tries: int = 0
    message: str = ""
    policy: str = ""
    stages: list = field(default_factory=list)

    @property
    def success(self):
        return self.exact


# =============================================================================
# The solver
# =============================================================================

class Homotopy:
    """The regularized and exact kernels of one problem, and the automatic solve.

    The kernels do not depend on the parameter values, so one instance
    serves every :meth:`solve`.

    Parameters
    ----------
    problem : OptimalControlProblem
    cache_dir : path, optional
        Where the compiled kernels are cached.
    rtol, atol : float
        Integration tolerances of the shooting.
    verbose : bool or int
        Print each step; 2 also prints every solve of the active-set route.
    policy : str or policy object
        What to do when the problem's optimal motion is not unique as stated
        (:mod:`~auto_optimal_control.policy`): ``"auto"`` (a
        :class:`~auto_optimal_control.policy.SecondaryCost` when needed),
        ``"as_stated"``, ``"strict"`` (raise), or a ``SecondaryCost`` or
        ``Lexicographic``. ``analysis`` holds what was found and
        ``problem`` the problem actually solved.
    """

    def __init__(self, problem, *, cache_dir=None, rtol=1e-12, atol=1e-12, verbose=False,
                 policy="auto"):
        self.stated = problem
        self.analysis = analyze(problem)
        self.policy = resolve(policy, self.analysis)
        self.policy_text = "" if self.policy is None else self.policy.describe(self.analysis)
        self._stages = None
        if isinstance(self.policy, SecondaryCost):
            problem = self.policy.apply(problem, self.analysis)
        elif isinstance(self.policy, Lexicographic):
            parts, problem = self.policy.stages(problem, self.analysis)
            self._stages = [(Homotopy(sub, cache_dir=cache_dir, rtol=rtol, atol=atol,
                                      verbose=verbose, policy="as_stated"), idx)
                            for sub, idx in parts]
        if self.policy is not None and policy == "auto":
            name = self.stated.name or "problem"
            warnings.warn(f"{name}: {self.analysis}\nApplied the default policy, "
                          f"{self.policy_text}. Pass policy= to choose another.", stacklevel=2)
        self.problem = problem
        self.params = None
        self.cache_dir = cache_dir
        self.options = dict(rtol=rtol, atol=atol)
        self.verbose = verbose
        self.attempt_evaluations = 60      # MINPACK evaluations per exact attempt
        self.attempt_nfev = 50000          # right-hand-side evaluations per arc
        self.active_set_seconds = 60.0     # time budget of the active-set route (see below)
        self.penalty_seconds = 300.0       # time budget of the penalty route
        self.path_options = dict(rtol=rtol, atol=atol)    # along the active-set route
        self.bang_bang = bang_bang_controls(problem)
        self.regularized, self.eps_symbol, self.rho_symbols = regularize(problem, self.bang_bang)
        self.kr = build_kernels(derive(self.regularized), cache_dir=cache_dir)
        if self.kr.system.riding is not None:
            self.active_set_seconds = 300.0     # many arcs, and no penalty route to fall back on
        self.nc = len(problem.path_constraints)
        self._kx = None
        self._ends = None
        self._seed_key = self._seed = None
        t, x, p = problem.time, problem.states, problem.params
        self._terminal = sp.lambdify((t, x, p), problem.terminal_cost, "numpy")

    def _say(self, *args):
        if self.verbose:
            print(*args)

    def _detail(self, *args):
        if self.verbose and self.verbose > 1:
            print(*args)

    # -- kernels and parameters ---------------------------------------------

    @property
    def exact_kernels(self):
        """Kernels of the exact problem: singular arcs derived when there are
        bang-bang controls; otherwise the regularized kernels (at ``rho = 0``
        they are the exact ones)."""
        if self._kx is None:
            if self.bang_bang:
                self._kx = build_kernels(derive(exact(self.problem), singular=True),
                                         cache_dir=self.cache_dir)
            else:
                self._kx = self.kr
        return self._kx

    def exact_params(self):
        if self.bang_bang:
            return self.params.copy()
        return self.regularized_params(None, np.zeros(self.nc))

    def regularized_params(self, eps, rho, off=None):
        """The regularized problem's parameter vector at smoothing ``eps``, penalty
        weights ``rho`` and limit offsets ``off`` (default zero: the exact limits)."""
        p = list(self.params)
        if self.bang_bang:
            p.append(eps)
        off = np.zeros(self.nc) if off is None else off
        return np.array(p + list(rho) + list(off), dtype=float)

    def objective(self, shooting, v):
        """``phi(t_f, x(t_f)) + int L`` of the exact problem at ``v``."""
        if isinstance(shooting, MultiArcShooting):
            _, _, segs = shooting._evaluate(np.asarray(v, dtype=float), jac=False, record=True)
            running = float(sum(r["cost"] for *_, r in segs))
            tf, xf = segs[-1][3], segs[-1][-1]["z"][:shooting.kernels.N_STATES]
        else:
            x0, lam0, _, tf = shooting.unpack(v)
            r = shooting.kernels.propagate(shooting.t0, np.concatenate([x0, lam0]), tf,
                                           shooting.params, cost=True, **self.options)
            running, xf = float(r["cost"]), r["z"][:shooting.kernels.N_STATES]
        return running + float(self._terminal(tf, list(xf), list(self.params)))

    def _limits_broken_at_the_ends(self):
        """Path constraints the fixed initial or final state already violates.

        No motion can satisfy them, whatever the search does; finding that
        out by searching takes minutes on a large problem. Only ends with
        every state fixed are checked, and only time-invariant limits.
        """
        problem, kr = self.problem, self.kr
        if not self.nc:
            return []
        if self._ends is None:
            t, x, p = problem.time, problem.states, problem.params
            final = {}
            for psi in problem.terminal_constraints:
                involved = [s for s in x if s in psi.free_symbols]
                if len(involved) == 1 and sp.diff(psi, involved[0]) == 1:
                    final[involved[0]] = involved[0] - psi
            at_end = None
            if len(final) == len(x) and not any(
                    t in e.free_symbols for e in final.values()):
                at_end = sp.lambdify([p], [final[s] for s in x], "numpy")
            start = None
            if len(problem.initial_state) == len(x):
                start = sp.lambdify([p], [problem.initial_state[s] for s in x], "numpy")
            ok = not any(t in c.expr.free_symbols for c in problem.path_constraints)
            self._ends = (start, at_end) if ok else (None, None)
        start, at_end = self._ends
        params = self.regularized_params(1.0 if self.bang_bang else None, np.zeros(self.nc))
        broken = []
        for label, end in (("the initial state", start), ("the final state", at_end)):
            if end is None:
                continue
            x = np.asarray(end(list(self.params)), dtype=float)
            g = np.asarray(kr.constraint(0.0, np.concatenate([x, np.zeros(len(x))]), params))
            for c in np.flatnonzero(g > 1e-9 * np.maximum(1.0, np.abs(g))):
                broken.append(f"{label} violates {kr.CONSTRAINT_NAMES[c]} by {g[c]:.6g}")
        return broken

    # -- step 2: sampling -----------------------------------------------------

    def _free_arc_params(self):
        """Indices of the parameters that matter off the path constraints."""
        problem = self.problem
        exprs = [*problem.dynamics, problem.running_cost, problem.terminal_cost,
                 *problem.terminal_constraints, *problem.initial_state.values(),
                 problem.initial_time]
        for cs in problem.control_sets:
            exprs += [cs.lower, cs.upper] if isinstance(cs, Box) else []
        used = set().union(*(sp.sympify(e).free_symbols for e in exprs))
        return [i for i, s_ in enumerate(problem.params) if s_ in used]

    def _separate_blocks(self, shoot):
        """Per controlled component, its state indices and its terminal rows, or None.

        Only for a free final time with no terminal cost, each component fixing its
        own final state: then, with the penalties off, a component with the other
        costates at zero is its own minimum-time problem inside the whole one.
        """
        k, problem = shoot.kernels, self.problem
        parts = self.analysis.controlled
        if (len(parts) < 2 or not k.FREE_FINAL_TIME or problem.terminal_cost != 0
                or shoot.n_free or k.N_SWITCHING):
            return None
        index = {s: i for i, s in enumerate(problem.states)}
        terminal = [{index[s] for s in psi.free_symbols if s in index}
                    for psi in problem.terminal_constraints]
        blocks, covered = [], set()
        for part in parts:
            states = sorted(index[s] for s in part.states)
            rows = [i for i, involved in enumerate(terminal) if involved & set(states)]
            if len(rows) != len(states) or any(not terminal[i] <= set(states) for i in rows):
                return None
            covered |= set(states)
            blocks.append((np.array(states), np.array(rows)))
        return blocks if len(covered) == len(index) else None

    def _block_system(self, shoot, states, rows):
        """One component's equations ``(R, J)`` in its costates, and ``t_f`` when it is free."""
        k = shoot.kernels
        n = k.N_STATES
        nu = np.zeros(k.N_MULTIPLIERS)

        def system(lam, tf, free, with_jacobian=True):
            lam0 = np.zeros(n)
            lam0[states] = lam
            z0 = np.concatenate([shoot.x0, lam0])
            r = shoot._propagate(z0, tf, stm=with_jacobian)
            zf, phi = r["z"], r["stm"]
            psi = k.residual(z0, zf, nu, tf, shoot.params)[n:][rows]
            if not with_jacobian:
                H = np.atleast_1d(k.hamiltonian(tf, zf, shoot.params))
                return np.concatenate([psi, H]) if free else psi
            jac = k.residual_jacobian(z0, zf, nu, tf, shoot.params)
            Rzf = jac["zf"][n:][rows]
            columns = n + states
            J = Rzf @ phi[:, columns]
            if not free:
                return psi, J
            Ht, Hz = k.hamiltonian_gradient(tf, zf, shoot.params)
            H = np.atleast_1d(k.hamiltonian(tf, zf, shoot.params))
            top = np.hstack([J, (Rzf @ r["dz_dt1"])[:, None]])
            bottom = np.concatenate([Hz @ phi[:, columns], [float(Ht) + Hz @ r["dz_dt1"]]])
            return np.concatenate([psi, H]), np.vstack([top, bottom])
        return system

    def _separate_seed(self, shoot, final_time, tol):
        """A first guess from the components solved one at a time: ``v``, or None.

        Random draws must land every component at once, and a nearly bang-bang
        control gives Newton little to work with. But given plenty of time a
        component's cost is nearly quadratic and its bounds are idle, so Newton needs no
        guess there. From that end each component follows its costates as its time
        shrinks, until ``H`` reaches zero: its own minimum time. The slowest sets
        ``t_f``, and the others take their costates from their paths at that time.
        """
        blocks = self._separate_blocks(shoot)
        if blocks is None:
            return None
        # A guess, not the answer: the whole system's solve polishes it afterwards.
        whole = shoot
        shoot = SingleShooting(shoot.kernels, shoot.params, rtol=1e-8, atol=1e-10, n_interior=0)
        # It does not depend on the path constraints (_free_arc_params), so it ignores their guards.
        shoot.options["constraint_events"] = False
        shoot.max_nfev = 20000             # a bad guess can chatter on a saturation corner
        shoot.jacobian_rtol = PATH_JACOBIAN_RTOL
        tol = max(tol, 1e-6)
        k = shoot.kernels
        n = k.N_STATES
        rng = np.random.default_rng(0)
        start = max(float(_draw(final_time, rng)) for _ in range(8)) if callable(final_time) \
            else float(np.max(final_time[1]))
        paths = []
        for states, rows in blocks:
            system = self._block_system(shoot, states, rows)
            if np.max(np.abs(system(np.zeros(len(states)), start, False)[0])) < tol:
                paths.append(None)         # already there: it never moves
                continue
            # A component only has to reach the slowest time found so far.
            floor = max((p[-1][0] for p in paths if p is not None), default=0.0)
            path = self._shrink(shoot, system, states, start, tol, floor)
            if path is None:
                self._say("separate seed: a component's time did not shrink to its minimum")
                return None
            paths.append(path)
        moving = [p for p in paths if p is not None]
        if not moving:
            return None
        T = self._common_time(moving)
        self._say(f"separate seed: t_f = {T:.6f}")
        lam0 = np.zeros(n)
        for (states, rows), path in zip(blocks, paths):
            if path is None:
                continue
            system = self._block_system(shoot, states, rows)
            at = [point[1] for point in path if point[0] >= T][-1]
            lam = _newton(lambda w, jac, tf=T: system(w, tf, False, jac), at, tol,
                          accept=SEED_ACCEPT * tol)
            if lam is None:
                self._say("separate seed: a component did not reach the common time")
                return None
            lam0[states] = lam
        try:
            r = whole._propagate(np.concatenate([whole.x0, lam0]), T, stm=False)
        except _FAILURES:
            return None
        return whole.pack(lam0, r["z"][n:n + k.N_MULTIPLIERS], T)

    @staticmethod
    def _common_time(paths):
        """The ``t_f`` where the components' Hamiltonians add up to zero.

        Each path's ``H`` is one (the cost of time) plus that component's own part, so the
        sum is one plus the parts. It is positive with time to spare, and each part
        falls to minus one at the component's own minimum time. Several components that
        need nearly the same time share the one between them, so ``t_f`` sits above
        them all, and Newton does not have to find that from a long way off.
        """
        grid = np.unique(np.concatenate([[p[0] for p in path] for path in paths]))
        low = max(path[-1][0] for path in paths)
        grid = grid[grid >= low]

        def total(time_):
            value = 1.0
            for path in paths:
                times, values = np.array([p[0] for p in path])[::-1], np.array([p[2] for p in path])[::-1]
                value += float(np.interp(time_, times, values)) - 1.0
            return value

        values = [total(g) for g in grid]
        for k in range(len(grid) - 1):
            if values[k] <= 0.0 < values[k + 1]:
                share = -values[k] / (values[k + 1] - values[k])
                return float(grid[k] + share * (grid[k + 1] - grid[k]))
        return float(low)

    def _shrink(self, shoot, system, states, start, tol, floor=0.0):
        """``[(T, lam, H)]`` from a long time down to the component's minimum time, or to
        ``floor`` if that comes first; or None."""
        lam = _newton(lambda w, jac: system(w, start, False, jac), np.zeros(len(states)), tol)
        if lam is None:
            return None
        path, T, previous = [], start, None
        step = 0.15
        k = shoot.kernels

        def hamiltonian(lam_, tf):
            lam0 = np.zeros(k.N_STATES)
            lam0[states] = lam_
            zf = shoot._propagate(np.concatenate([shoot.x0, lam0]), tf, stm=False)["z"]
            return float(np.atleast_1d(k.hamiltonian(tf, zf, shoot.params))[0])

        H0 = hamiltonian(lam, T)
        if H0 <= 0.0:                      # already below its minimum time: nothing to follow
            return None
        path.append((T, lam, H0))
        while step > 1e-5:
            to = max(T * (1.0 - step), floor)
            guess = lam
            if previous is not None:
                # A secant in logarithms: costates grow like powers of 1 / T.
                ratio = np.log(to / T) / np.log(T / previous[0])
                same = lam * previous[1] > 0.0
                guess = np.where(same, lam * np.abs(lam / np.where(same, previous[1], 1.0)) ** ratio,
                                 lam + (lam - previous[1]) * (to - T) / (T - previous[0]))
            found = _newton(lambda w, jac, tf=to: system(w, tf, False, jac), guess, tol)
            if found is None and previous is not None:
                found = _newton(lambda w, jac, tf=to: system(w, tf, False, jac), lam, tol)
            if found is None:
                step *= 0.5
                continue
            try:
                H = hamiltonian(found, to)
            except _FAILURES:
                step *= 0.5
                continue
            if H < 0.0:
                # Between T and to: solve for the time as well.
                w = np.append(found, to)
                root = _newton(lambda v, jac: system(v[:-1], v[-1], True, jac), w, tol,
                               accept=SEED_ACCEPT * tol)
                if root is not None and root[-1] > 0.0:
                    path.append((float(root[-1]), root[:-1], 0.0))
                    return path
                step *= 0.5
                continue
            previous = (T, lam)
            T, lam = to, found
            path.append((T, lam, H))
            if T <= floor:
                return path
            step = min(step * 1.3, 0.15)
        return None

    def _sampler(self, shoot, costates, final_time):
        n = self.kr.N_STATES

        def sample(rng):
            lam0 = np.asarray(_draw(costates, rng), dtype=float).reshape(-1)
            if lam0.shape != (n,):
                raise ValueError(f"the costate sample must have {n} entries")
            tf = None
            if self.kr.FREE_FINAL_TIME:
                if final_time is None:
                    raise ValueError("the final time is free: give a final_time distribution")
                tf = float(_draw(final_time, rng))
            return shoot.guess_multipliers(lam0, tf=tf)

        return sample

    # -- step 4: reading the structure ----------------------------------------

    def _read(self, shoot, v, n_points=2001):
        """Labels of a regularized solution: time, states, violated intervals
        per constraint, and per bang-bang control its bound labels."""
        kr, p = self.kr, shoot.params
        x0, lam0, _, tf = shoot.unpack(v)
        z0 = np.concatenate([x0, lam0])
        tr = kr.trajectory(shoot.t0, z0, tf, p, n_points=n_points, **self.options)
        t, Z, U = tr["t"], tr["z"], tr["u"]
        T = tf - shoot.t0
        out = {"t": t, "z": Z, "u": U, "T": T, "t0": shoot.t0, "tf": tf, "z0": z0,
               "violated": [], "labels": {}}
        # Violated intervals, from the located crossings and g between them.
        for c in range(self.nc):
            cross = np.sort(tr["constraint_times"][tr["constraint_index"] == c])
            ts = np.concatenate([[shoot.t0], cross, [tf]])
            mids = 0.5 * (ts[:-1] + ts[1:])
            gm = np.interp(mids, t, tr["g"][:, c])
            spans = [(a, b) for a, b, g in zip(ts[:-1], ts[1:], gm) if g > 0 and b > a]
            merged = []
            for a, b in spans:
                if merged and a - merged[-1][1] < 0.02 * T:
                    merged[-1] = (merged[-1][0], b)
                else:
                    merged.append((a, b))
            out["violated"].append(merged)
        controls = list(kr.system.controls)
        for u in self.bang_bang:
            j = controls.index(u)
            bounds = np.array([kr.control_bounds(ti, zi, p) for ti, zi in zip(t, Z)])
            lo, hi = bounds[:, 0, j], bounds[:, 1, j]
            tol = 1e-9 * np.maximum(hi - lo, 1e-300)
            lab = np.zeros(len(t), dtype=int)
            lab[U[:, j] >= hi - tol] = 1
            lab[U[:, j] <= lo + tol] = -1
            out["labels"][u] = lab
        return out

    @staticmethod
    def _runs(t, mask):
        """``[(start, end)]`` of the runs where ``mask`` holds, in time."""
        edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
        out = []
        for a, b in zip(edges[0::2], edges[1::2]):
            ta = t[a] if a == 0 else 0.5 * (t[a - 1] + t[a])
            tb = t[b - 1] if b == len(t) else 0.5 * (t[b - 1] + t[b])
            out.append((ta, tb))
        return out

    def _candidates(self, read, max_candidates=24):
        """Candidate structures from a reading: ``(items, held)`` per candidate.

        ``items`` are ``(kind, index, a, b)`` in time order with kind
        ``boundary``, ``touch``, ``singular``; ``held`` gives, per control
        with a singular arc, the bound labels of its free arcs. The second
        return value is whether any candidate could be built (``False``
        while a limit without exact arcs is violated).
        """
        kx = self.exact_kernels
        t, T = read["t"], read["T"]
        options = []                       # per violated interval: kinds to try
        for c, spans in enumerate(read["violated"]):
            for a, b in spans:
                if not kx.EXACT_ARCS[c]:
                    return [], False
                kinds = ["boundary"]
                if kx.CONSTRAINT_ORDERS[c] == 2:
                    kinds = ["touch", "boundary"] if b - a < 0.05 * T else ["boundary", "touch"]
                options.append([(k, c, a, b) for k in kinds])
        limited = [(a, b) for spans in read["violated"] for a, b in spans]

        def on_limit(a, b):
            return sum(max(0.0, min(b, y) - max(a, x)) for x, y in limited)

        names = [sw.control for sw in kx.system.switching_functions]
        singular_choices = []              # per control: list of run subsets
        ramps = {}                         # per control: pieces that count against readiness
        near = 0.02 * T
        for u, lab in read["labels"].items():
            # Stretches strictly between the bounds, cut where a limit is
            # violated (the control there is the one riding the limit).
            pieces = []
            for a, b in self._runs(t, lab == 0):
                cuts = [(a, b)]
                for x, y in limited:
                    cuts = [(p, q) for (p0, q0) in cuts
                            for p, q in ((p0, min(q0, x)), (max(p0, y), q0)) if q > p]
                pieces += cuts
            # A piece that runs onto a limit is the smoothed approach to it;
            # one that leaves a limit may be a joined singular arc.
            onto = [any(abs(b - x) < near for x, _ in limited) for a, b in pieces]
            ramps[u] = [r for r, o in zip(pieces, onto)
                        if not o and not any(abs(r[0] - y) < near for _, y in limited)]
            sw = names.index(u)
            if not kx.SINGULAR_ARCS[sw]:
                singular_choices.append([(u, ())])
                continue
            long = sorted([r for r, o in zip(pieces, onto) if not o and r[1] - r[0] >= 0.02 * T],
                          key=lambda r: r[0] - r[1])[:3]
            subsets = [()]
            for k in range(1, len(long) + 1):
                subsets += [tuple(sorted(c)) for c in itertools.combinations(long, k)]
            singular_choices.append([(u, s) for s in subsets[::-1]])   # most singular first

        out = []
        for limits in itertools.product(*options) if options else [()]:
            for sing in itertools.product(*singular_choices) if singular_choices else [()]:
                # Ready: the ramps not read as singular are short.
                rest = sum(b - a for u, subset in sing for a, b in ramps[u]
                           if (a, b) not in subset)
                if rest > 0.05 * T:
                    continue
                items = sorted(list(limits) + [("singular", u, a, b) for u, subset in sing
                                               for a, b in subset], key=lambda it: it[2])
                out.append(items)
                if len(out) >= max_candidates:
                    return out, True
        return out, True

    def _structure(self, items, read):
        """``(arcs, values)`` for MultiArcShooting from candidate items, or None.

        Adds the joins, the bounds held next to singular arcs and the
        switches of controls with singular arcs on free arcs.
        """
        t, T = read["t"], read["T"]
        special = [(a, b) for kind, _, a, b in items]
        for (a1, b1), (a2, b2) in zip(special[:-1], special[1:]):
            if a2 < b1 - 0.02 * T:
                return None                # two special arcs at once
        frozen = {it[1] for it in items if it[0] == "singular"}
        # Per frozen control: (time, label) of its free-arc samples.
        free = np.ones(len(t), dtype=bool)
        for a, b in special:
            free &= ~((t >= a) & (t <= b))
        events = []                        # (time, entry)
        for i, (kind, c, a, b) in enumerate(items):
            if kind != "singular":
                events.append((a, (kind, c, a, b)))
                continue
            prev = items[i - 1] if i else None
            joined = prev is not None and prev[0] == "boundary" and a - prev[3] < 0.02 * T
            nxt = items[i + 1] if i + 1 < len(items) else None
            if nxt is not None and nxt[2] - b < 0.02 * T:
                return None                # singular -> boundary joins are not supported
            events.append((a, ("singular", c, a, b, joined)))
        # Labels on free arcs for frozen controls: segments and switches.
        held = {}
        for u in frozen:
            lab = read["labels"][u]
            segs = []                      # (start, end, label) of free labeled stretches
            for value in (1, -1):
                for a, b in self._runs(t, free & (lab == value)):
                    segs.append((a, b, value))
            segs.sort()
            held[u] = segs
        timeline = sorted(events, key=lambda e: e[0])
        # Switch events for frozen controls: label flips between consecutive
        # free segments with no singular arc of that control in between.
        for u in frozen:
            segs = held[u]
            sing = [(e[1][2], e[1][3]) for e in timeline if e[1][0] == "singular" and e[1][1] == u]
            for (a1, b1, l1), (a2, b2, l2) in zip(segs[:-1], segs[1:]):
                if l1 != l2 and not any(b1 - 1e-12 <= x <= a2 + 1e-12 or b1 <= y <= a2
                                        for x, y in sing):
                    ts = 0.5 * (b1 + a2)
                    inside = [(x, y) for kind, c, x, y in items if kind != "singular"
                              and x <= ts <= y]
                    if inside:
                        ts = inside[0][1] + 1e-3 * T
                    timeline.append((ts, ("switch", u, ts)))
        timeline.sort(key=lambda e: e[0])

        def label_near(u, time, before):
            segs = held[u]
            if before:
                cand = [s for s in segs if s[1] <= time + 1e-12]
                return cand[-1][2] if cand else None
            cand = [s for s in segs if s[0] >= time - 1e-12]
            return cand[0][2] if cand else None

        sides = []
        for _, ev in timeline:
            if ev[0] == "singular":
                _, u, a, b, joined = ev
                start = a
                if joined:
                    start = [x for kind, c, x, y in items if kind == "boundary" and y <= a + 0.02 * T][-1]
                before = label_near(u, start, True)
                after = label_near(u, b, False)
                sides.append((before, after))
        return timeline, sides

    def _assemble(self, timeline, sides_choice, read, base, jitter_rng=None):
        """Arc entries and their values; ``jitter_rng`` perturbs the times."""
        T, t0, tf = read["T"], read["t0"], read["tf"]
        times = []
        for _, ev in timeline:
            if ev[0] in ("boundary", "touch"):
                times += [ev[2], ev[3]] if ev[0] == "boundary" else [0.5 * (ev[2] + ev[3])]
            elif ev[0] == "singular":
                times += [ev[3]] if ev[4] else [ev[2], ev[3]]
            else:
                times.append(ev[2])
        times = np.array(times, dtype=float)
        lo, hi = t0 + 0.01 * T, tf - 0.01 * T
        times = np.clip(times, lo, hi)
        if jitter_rng is not None and len(times):
            grid = np.concatenate([[t0], times, [tf]])
            gaps = np.minimum(np.diff(grid)[:-1], np.diff(grid)[1:])
            times = times + jitter_rng.uniform(-0.5, 0.5, len(times)) * np.maximum(gaps, 0.02 * T)
            times = np.clip(times, lo, hi)
        if np.any(np.diff(times) <= 0):
            return None
        arcs, values, k = [], [], 0
        si = 0
        for _, ev in timeline:
            kind = ev[0]
            if kind == "boundary":
                a, b = times[k], times[k + 1]
                k += 2
                arcs.append(("boundary", ev[1]))
                values.append([a, b, None])
            elif kind == "touch":
                arcs.append(("touch", ev[1]))
                values.append([times[k], None])
                k += 1
            elif kind == "singular":
                before, after = sides_choice[si]
                si += 1
                name = ev[1].name
                if ev[4]:
                    arcs.append(("singular", name, before, after, "joined"))
                    values.append(times[k])
                    k += 1
                else:
                    arcs.append(("singular", name, before, after))
                    values.append((times[k], times[k + 1]))
                    k += 2
            else:
                arcs.append(("switch", ev[1].name))
                values.append(times[k])
                k += 1
        return arcs, values

    def _fit_jumps(self, shoot, values, read):
        """Fill in boundary ``pi`` and touch ``nu`` from the regularized costates."""
        kx, p, n = shoot.kernels, shoot.params, shoot.kernels.N_STATES
        t, Z = read["t"], read["z"]

        def z_at(ti):
            return np.array([np.interp(ti, t, Z[:, i]) for i in range(Z.shape[1])])

        out = []
        for (kind, c, _), val in zip(shoot.arcs, values):
            if kind == "boundary":
                a, b, _ = val
                q = kx.CONSTRAINT_ORDERS[c]
                pi = np.zeros(q)
                try:
                    za = z_at(a)
                    back = kx.propagate_boundary(c, b, z_at(b), a, p, **self.options)["z"]
                    _, Nx, _, _ = kx.tangency_gradient(c, a, za, p)
                    pi = np.linalg.lstsq(Nx.T, za[n:] - back[n:], rcond=None)[0]
                except _FAILURES:
                    pass
                out.append((a, b, pi))
            elif kind == "touch":
                tc = val[0]
                nu = 0.0
                try:
                    _, Nx, _, _ = kx.tangency_gradient(c, tc, z_at(tc), p)
                    w = 0.02 * read["T"]
                    jump = z_at(tc - w)[n:] - z_at(tc + w)[n:]
                    nu = float(np.linalg.lstsq(Nx[:1].T, jump, rcond=None)[0][0])
                except _FAILURES:
                    pass
                out.append((tc, nu))
            else:
                out.append(val)
        return out

    def _accept(self, shoot, sol, tol):
        """Whether an exact solution passes the checks it can make."""
        if not sol.success or not np.isfinite(sol.residual_norm):
            return False
        if sol.max_violation > 1e-7:
            return False
        if not isinstance(shoot, MultiArcShooting) or not shoot.arcs:
            return True
        try:
            d = shoot.diagnostics(sol.v)
        except _FAILURES:
            return False
        small = 1e-7
        if any(nu < -small for nu in d["touch_nu"]):
            return False
        if any(lo < -small * max(1.0, abs(hi)) for lo, hi in d["boundary_mu"]):
            return False
        if any(lo < -small * max(1.0, abs(hi)) for lo, hi in d["singular_legendre"]):
            return False
        if any(w > 1e-6 for w in d["held_sign_violation"].values()):
            return False
        return True

    def _attempt(self, read, base, n_jitter, rng, tol):
        """Exact solves of every candidate from one reading: ``[(J, shoot, sol)]``."""
        kx, px = self.exact_kernels, self.exact_params()
        candidates, possible = self._candidates(read)
        tried, accepted = 0, []
        seen = set()
        for items in candidates:
            built = self._structure(items, read)
            if built is None:
                continue
            timeline, sides = built
            if any(before is None and after is None for before, after in sides):
                continue                   # the control never reaches a bound: not ready
            side_options = []
            for before, after in sides:
                b_opts = ["upper", "lower"] if before is None else [
                    "upper" if before > 0 else "lower"]
                a_opts = ["lower", "upper"] if after is None else [
                    "upper" if after > 0 else "lower"]
                side_options.append([(b, a) for b in b_opts for a in a_opts])
            for sides_choice in itertools.product(*side_options) if side_options else [()]:
                for j in range(n_jitter + 1):
                    made = self._assemble(timeline, sides_choice, read, base,
                                          None if j == 0 else rng)
                    if made is None:
                        continue
                    arcs, values = made
                    try:
                        if arcs:
                            shoot = MultiArcShooting(kx, px, arcs, max_nfev=self.attempt_nfev,
                                                     **self.options)
                            vals = self._fit_jumps(shoot, values, read)
                            guess = shoot.pack_arcs(base, vals)
                            method = "lm" if kx.N_SWITCHING else "hybr"
                        else:
                            key = ("free",)
                            if key in seen:
                                continue
                            seen.add(key)
                            shoot = SingleShooting(kx, px, **self.options)
                            guess = np.asarray(base, dtype=float)
                            method = "hybr"
                        sol = shoot.solve(guess, method=method, tol=_scaled(tol, guess),
                                          options=_budget(method, self.attempt_evaluations))
                    except _FAILURES:
                        continue
                    tried += 1
                    if self._accept(shoot, sol, _scaled(tol, guess)):
                        try:
                            J = self.objective(shoot, sol.v)
                        except _FAILURES:
                            continue
                        accepted.append((J, shoot, sol))
                        break              # this structure is solved; next candidate
        return accepted, tried, possible

    # -- the active-set route ---------------------------------------------------

    def _smoothed(self, arcs, eps, off):
        """Shooting on the regularized kernels with exact arcs ``arcs`` and penalties off."""
        p = self.regularized_params(eps, np.zeros(self.nc), off)
        shoot = (MultiArcShooting(self.kr, p, arcs, max_nfev=self.attempt_nfev, **self.path_options)
                 if arcs else SingleShooting(self.kr, p, **self.path_options))
        shoot.jacobian_rtol = PATH_JACOBIAN_RTOL
        shoot.fast_residuals = True
        shoot.tangent_junctions = self.kr.system.riding is not None
        return shoot

    def _free_samples(self, shoot, v, n_points=801):
        """``[(t, g, active)]`` sampled on every segment of ``shoot`` at ``v``:
        free arcs (``active`` empty) and boundary arcs (the limits ridden)."""
        from fast_numerical_integration.integrators import solve_ivp

        k = shoot.kernels
        if not isinstance(shoot, MultiArcShooting):
            x0, lam0, _, tf = shoot.unpack(v)
            tr = k.trajectory(shoot.t0, np.concatenate([x0, lam0]), tf, shoot.params,
                              n_points=n_points, constraint_events=False, **self.options)
            return [(tr["t"], tr["g"], ())]
        _, _, segs = shoot._evaluate(np.asarray(v, dtype=float), jac=False)
        span = segs[-1][3] - shoot.t0
        out = []
        for kind, aid, ta, tb, za, flags, _ in segs:
            m = max(int(n_points * (tb - ta) / span), 3)
            if kind is None:
                tr = k.trajectory(ta, za, tb, shoot.params, n_points=m, constraint_events=False,
                                  arc=shoot._free_arc(ta, za, flags), frozen=sorted(flags),
                                  **self.options)
                out.append((tr["t"], tr["g"], ()))
            elif kind == "boundary":
                te = np.linspace(ta, tb, m)
                sol = solve_ivp(k.BoundaryDynamics(aid, shoot.params), (ta, tb), za,
                                method="DOP853", t_eval=te, rtol=self.options["rtol"],
                                atol=self.options["atol"])
                g = np.array([k.constraint(ti, zi, shoot.params)
                              for ti, zi in zip(sol.t, sol.y.T)]).reshape(len(sol.t), -1)
                out.append((sol.t, g, tuple(k.arc_constraints(aid))))
        return out

    def _free_violations(self, shoot, v, scale):
        """``[(c, start, end, peak)]``: where each limit is violated while it is not ridden."""
        out = []
        for t, g, active in self._free_samples(shoot, v):
            for c in range(self.nc):
                if c in active:
                    continue
                bad = g[:, c] > 1e-6 * scale[c]    # above rounding at tangential junctions
                if not bad.any():
                    continue
                idx = np.flatnonzero(bad)
                for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                    a, b = run[0], run[-1]
                    # Widen to the neighbouring samples, where g was still <= 0.
                    ta, tb = t[max(a - 1, 0)], t[min(b + 1, len(t) - 1)]
                    out.append((c, ta, tb, float(g[run, c].max())))
        # One violation can run across a junction of other limits: it is one, not two.
        merged = []
        for c, ta, tb, peak in sorted(out, key=lambda v: (v[0], v[1])):
            if merged and merged[-1][0] == c and ta <= merged[-1][2]:
                merged[-1] = (c, merged[-1][1], max(tb, merged[-1][2]), max(peak, merged[-1][3]))
            else:
                merged.append((c, ta, tb, peak))
        return merged

    def _mas_entries(self, arcs, values, names=False):
        """MultiArcShooting entries for arcs that may overlap: ``(entries, values)``.

        The events (entries, exits, touches) are ordered by their times, so a
        limit that comes into play while another is ridden becomes an arc
        inside or across that one; the values follow the order of entry.
        """
        events = []
        for i, ((kind, c), val) in enumerate(zip(arcs, values)):
            if kind == "touch":
                events.append((float(val[0]), 1, "touch", i))
            else:
                events += [(float(val[0]), 1, "enter", i), (float(val[1]), 0, "exit", i)]
        events.sort(key=lambda e: (e[0], e[1]))
        label = (lambda c: self.kr.CONSTRAINT_NAMES[c]) if names else (lambda c: c)
        depth, overlap = 0, False
        for _, _, ev, _ in events:
            if ev != "exit" and depth > 0:
                overlap = True             # an entry or touch while an arc is ridden
            depth += 1 if ev == "enter" else -1 if ev == "exit" else 0
        order = [i for _, _, ev, i in events if ev != "exit"]
        if not overlap:
            # Plain sequence: the compact entries.
            entries = [("boundary" if arcs[i][0] == "boundary" else "touch", label(arcs[i][1]))
                       for i in order]
        else:
            entries = [(ev, label(arcs[i][1])) for _, _, ev, i in events]
        return entries, [values[i] for i in order]

    @staticmethod
    def _arcs_of(shoot, v):
        """The structure and its values, as MultiArcShooting entries (``[]`` for none)."""
        if not isinstance(shoot, MultiArcShooting):
            return [], []
        return [(kind, c) for kind, c, _ in shoot.arcs], list(shoot.unpack_arcs(v))

    def _activate(self, shoot, v, violations):
        """Structures that bring the violated limits in, most likely first.

        A violation away from the limit's arcs becomes a newborn touch point
        (order 2) or a short boundary arc (order 1), with zero jump. A
        violation beside arcs of the same limit widens them into one
        boundary arc that covers it: a boundary arc widens, a touch point
        grows into a boundary arc (its jump kept as the first ``pi``), and
        neighbouring arcs with the violation between them merge. A touch
        point violated on both sides may instead split into two touch points
        that share its jump; that structure is offered second. Returns a list
        of ``(arcs, values)``.
        """
        kr = self.kr
        arcs, values = self._arcs_of(shoot, v)
        T = shoot.unpack(v)[3] - shoot.t0
        near = 0.05 * T

        def span(i):
            kind, val = arcs[i][0], values[i]
            return (val[0], val[0]) if kind == "touch" else (val[0], val[1])

        groups, born = [], []              # groups: [set of arc indices, [(a, b), ...]]
        for c, a, b, _ in violations:
            hits = {i for i, (_, cc) in enumerate(arcs)
                    if cc == c and a - near <= span(i)[1] and b + near >= span(i)[0]}
            if not hits:
                q = kr.CONSTRAINT_ORDERS[c]
                if q == 2:
                    born.append((("touch", c), (0.5 * (a + b), 0.0)))
                else:
                    w = max(b - a, 1e-3 * T) / 2
                    mid = 0.5 * (a + b)
                    born.append((("boundary", c), (mid - w, mid + w, np.zeros(q))))
                continue
            for g in groups:
                if g[0] & hits:
                    g[0] |= hits
                    g[1].append((a, b))
                    break
            else:
                groups.append([set(hits), [(a, b)]])

        def overlap(lo, hi):
            """The span of the arc this one overlaps most, or None."""
            spans = [span(i) for i, (kind, _) in enumerate(arcs) if kind == "boundary"]
            best = max(spans, key=lambda y: min(hi, y[1]) - max(lo, y[0]), default=None)
            return best if best and min(hi, best[1]) > max(lo, best[0]) else None

        def placed(entry_in, exit_in):
            """The newborn arcs with their ends moved inside the arc they overlap."""
            out = []
            for entry, val in born:
                other = overlap(val[0], val[1]) if entry[0] == "boundary" else None
                if other is None:
                    out.append((entry, val))
                    continue
                inset = 1e-3 * (other[1] - other[0])
                lo = max(val[0], other[0] + inset) if entry_in else val[0]
                hi = min(val[1], other[1] - inset) if exit_in else val[1]
                out.append((entry, (lo, hi, val[2])))
            return out

        def build(split, newborn=None):
            replaced = {}                  # arc index -> what stands in its place
            for idx, spans in groups:
                lo_i = min(idx)
                (kind, c), val = arcs[lo_i], values[lo_i]
                for i in idx:
                    replaced[i] = []
                if split and len(idx) == 1 and kind == "touch":
                    tc, nu = val
                    left = [s_ for s_ in spans if s_[1] <= tc + 1e-12 * T]
                    right = [s_ for s_ in spans if s_[0] >= tc - 1e-12 * T]
                    if left and right:
                        ta = 0.5 * (min(x for x, _ in left) + tc)
                        tb = 0.5 * (tc + max(y for _, y in right))
                        replaced[lo_i] = [(("touch", c), (ta, 0.5 * nu)),
                                          (("touch", c), (tb, 0.5 * nu))]
                        continue
                # The merged arc's entry jump carries the jumps of all it absorbs. Only
                # this limit's arcs go: another limit's arc may lie between them.
                pi = np.zeros(kr.CONSTRAINT_ORDERS[c])
                for i in idx:
                    if arcs[i][0] == "touch":
                        pi[0] += values[i][1]
                    else:
                        pi += np.asarray(values[i][2], dtype=float)
                lo = min([span(i)[0] for i in idx] + [x for x, _ in spans])
                hi = max([span(i)[1] for i in idx] + [y for _, y in spans])
                replaced[lo_i] = [(("boundary", c), (lo, hi, pi))]
            out = []
            for i, item in enumerate(zip(arcs, values)):
                out += replaced.get(i, [item])
            out += list(born if newborn is None else newborn)
            out.sort(key=lambda w: float(w[1][0]))
            return [a for a, _ in out], [w for _, w in out]

        # A limit that comes into play while another is ridden may enter and
        # leave during that arc, or cross either end. The events' order is
        # fixed in a solve, and a guess in the wrong order does not converge;
        # nested is the commonest, as two limits' peaks close in on each other.
        options = []
        if any(entry[0] == "boundary" and overlap(val[0], val[1]) for entry, val in born):
            options += [build(False, placed(True, True)), build(False),
                        build(False, placed(True, False)), build(False, placed(False, True))]
        else:
            options = [build(False)]
        if any(len(g[0]) == 1 and arcs[next(iter(g[0]))][0] == "touch" for g in groups):
            alternative = build(True)
            if alternative[0] != options[0][0]:
                options.append(alternative)
        return options

    def _retire(self, shoot, v):
        """The structure without arcs that have died: a boundary arc shrunk to
        nothing becomes a touch point (order 2) or goes (order 1), and a touch
        point whose jump has the wrong sign goes. ``None`` if nothing changes."""
        arcs, values = self._arcs_of(shoot, v)
        if not arcs:
            return None
        T = shoot.unpack(v)[3] - shoot.t0
        out_a, out_v, changed = [], [], False
        for (kind, c), val in zip(arcs, values):
            if kind == "boundary" and val[1] - val[0] < 1e-6 * T:
                changed = True
                if self.kr.CONSTRAINT_ORDERS[c] == 2:
                    out_a.append(("touch", c))
                    out_v.append((0.5 * (val[0] + val[1]), float(val[2][0])))
                continue
            if kind == "touch" and val[1] < -1e-9:
                changed = True
                continue
            out_a.append((kind, c))
            out_v.append(val)
        return (out_a, out_v) if changed else None

    def _exact_jump(self, arcs, v, tol):
        """The exact problem's solve of structure ``arcs`` from ``v``: ``(J, shoot, sol)`` or None."""
        kx, px = self.exact_kernels, self.exact_params()
        tol = _scaled(tol, v)
        # A large problem's search has run on tangential junctions; if a control really does
        # jump at one, the rows that allow it are the ones to try next.
        for tangent in ((True, False) if kx.system.riding is not None else (False,)):
            try:
                if arcs:
                    shoot = MultiArcShooting(kx, px, arcs, max_nfev=self.attempt_nfev, **self.options)
                    method = "lm" if kx.N_SWITCHING else "hybr"
                else:
                    shoot = SingleShooting(kx, px, **self.options)
                    method = "hybr"
                shoot.tangent_junctions = tangent
                sol = shoot.solve(v, method=method, tol=tol,
                                  options=_budget(method, 4 * self.attempt_evaluations))
                if self._accept(shoot, sol, tol):
                    return self.objective(shoot, sol.v), shoot, sol
                self._detail(f"  exact solve: residual {sol.residual_norm:.1e} (max |v| "
                             f"{np.abs(sol.v).max():.1e}), "
                             f"max violation {sol.max_violation:.1e}; {self._rejection(shoot, sol)}")
            except _FAILURES as exc:
                self._detail(f"  exact solve failed: {exc}")
        return None

    def _rejection(self, shoot, sol):
        """Why ``_accept`` turned a converged solution down (for the log)."""
        if not sol.success or not isinstance(shoot, MultiArcShooting) or not shoot.arcs:
            return "not converged" if not sol.success else "rejected"
        try:
            d = shoot.diagnostics(sol.v)
        except _FAILURES as exc:
            return f"diagnostics failed: {exc}"
        return (f"touch nu {np.round(d['touch_nu'], 6).tolist()}, mu ranges "
                f"{[(round(a, 6), round(b, 6)) for a, b in d['boundary_mu']]}")

    def _active_set(self, sol, eps0, s_max, steps, tol, max_solves=400):
        """The active-set route: from the unconstrained regularized solution ``sol``
        at ``eps0``, tighten every limit from just above the solution to its
        true value, bringing in arcs as limits come into play, then lower
        ``eps`` with the structure following, trying the exact problem at each
        step. Returns ``(J, shoot, sol)`` or None."""
        deadline = time.perf_counter() + self.active_set_seconds
        shoot = self._smoothed([], eps0, np.zeros(self.nc))
        (_, g, _), = self._free_samples(shoot, sol.v)
        scale = np.maximum(g.max(axis=0) - g.min(axis=0), 1e-12)
        off0 = np.maximum(g.max(axis=0), 0.0) + 1e-3 * scale
        state = {"shoot": shoot, "v": sol.v, "le": np.log10(eps0) if self.bang_bang else 0.0,
                 "off": off0.copy(), "solves": 0}
        log_eps_min = state["le"] - s_max

        # The path only has to be followed closely, not solved to the last digit:
        # at an order-1 entry with a smooth control H continuity has a double
        # root, and Newton converges only linearly there. The exact jumps use ``tol``.
        path_tol = max(tol, 1e-6)

        def solve_structure(arcs, values, base, le, off):
            entries, values = self._mas_entries(arcs, values)
            sh = self._smoothed(entries, 10.0 ** le if self.bang_bang else None, off)
            guess = sh.pack_arcs(base, values) if arcs else np.asarray(base, dtype=float)
            state["solves"] += 1
            r = sh.solve(guess, tol=path_tol,
                         options=_budget("hybr", 4 * self.attempt_evaluations))
            return sh, r

        def settle(sh, r, le, off):
            """Bring in violated limits and retire dead arcs until the solution at
            (le, off) is consistent; ``(shoot, sol)`` or None."""
            last = None
            for _ in range(2 * self.nc + 4):
                viol = self._free_violations(sh, r.v, scale)
                if viol:
                    worst = max(peak / scale[c] for c, _, _, peak in viol)
                    if worst > 1e-3:
                        self._detail("    violation too large: a shorter step")
                        return None        # too far in: take a shorter step
                    if last is not None and last[0] == self._arcs_of(sh, r.v)[0] \
                            and worst < 1e-4 and worst > 0.5 * last[1]:
                        # The same structure again, and the violation did not
                        # shrink: it is the path tolerance at a tangential
                        # junction, not a missing arc. The exact solve judges it.
                        return sh, r
                    last = (self._arcs_of(sh, r.v)[0], worst)
                    options = self._activate(sh, r.v, viol)
                else:
                    dead = self._retire(sh, r.v)
                    if dead is None:
                        return sh, r
                    options = [dead]
                base = r.v[:sh.n_base] if isinstance(sh, MultiArcShooting) else r.v
                for arcs, values in options:
                    try:
                        nsh, nr = solve_structure(arcs, values, base, le, off)
                    except _FAILURES as exc:
                        self._detail(f"    try {self._mas_entries(arcs, values, True)[0]}: {exc}")
                        continue
                    self._detail(f"    try {self._mas_entries(arcs, values, True)[0]}: "
                                 f"residual {nr.residual_norm:.1e}")
                    if nr.success:
                        break
                else:
                    return None
                sh, r = nsh, nr
                self._say(f"  structure: {self._names(arcs)}")
            return None

        def advance(target_le, target_off, on_step=None):
            le0, off_start = state["le"], state["off"].copy()
            frac, ds = 0.0, 0.25
            prev = None                    # (frac, v, structure) of the solution before
            while frac < 1.0:
                if (state["solves"] > max_solves or ds < 1e-6
                        or time.perf_counter() > deadline):
                    return False
                f = min(frac + ds, 1.0)
                le = le0 + f * (target_le - le0)
                off = off_start + f * (target_off - off_start)
                sh0, v0 = state["shoot"], state["v"]
                arcs, values = self._arcs_of(sh0, v0)
                guesses = [v0]
                if prev is not None and prev[2] == arcs and frac > prev[0]:
                    # Secant predictor along the path, while the structure holds.
                    guesses.insert(0, v0 + (v0 - prev[1]) * (f - frac) / (frac - prev[0]))
                done = None
                for g in guesses:
                    base = g[:sh0.n_base] if isinstance(sh0, MultiArcShooting) else g
                    vals = self._arcs_of(sh0, g)[1] if arcs else values
                    try:
                        sh, r = solve_structure(arcs, vals, base, le, off)
                        self._detail(f"  step to {f:.6f} (ds {ds:.1e}): residual "
                                     f"{r.residual_norm:.1e}; junctions "
                                     f"{self._junction_text(sh, r.v)}")
                        done = settle(sh, r, le, off) if r.success else None
                    except _FAILURES as exc:
                        self._detail(f"  step to {f:.6f} (ds {ds:.1e}): {exc}")
                        done = None
                    if done is not None:
                        break
                if done is None:
                    ds *= 0.5
                    continue
                prev = (frac, v0, arcs)
                state.update(shoot=done[0], v=done[1].v, le=le, off=off)
                if self._arcs_of(done[0], done[1].v)[0] != arcs:
                    prev = None            # the structure changed: no secant across it
                frac, ds = f, min(ds * 1.5, 0.25)
                if on_step is not None and on_step():
                    return True
            return True

        found = []

        def try_exact():
            sh, v = state["shoot"], state["v"]
            arcs, values = self._arcs_of(sh, v)
            entries, _ = self._mas_entries(arcs, values, names=True)
            step = Step(log_eps_min + s_max - state["le"] if self.bang_bang else 0.0,
                        10.0 ** state["le"] if self.bang_bang else np.nan, np.zeros(self.nc),
                        0.0, candidates=1, structure=tuple(entries))
            steps.append(step)
            hit = self._exact_jump(entries, v, tol)
            self._say(f"active set at eps = {step.eps:.3g}: {self._names(arcs)}, "
                      f"exact {'accepted' if hit else 'not accepted'}")
            if hit is not None:
                step.accepted = True
                found.append(hit)
                return True
            return False

        self._say(f"active set: tightening the limits from offsets {np.round(off0, 6)}")
        if not advance(state["le"], np.zeros(self.nc)):
            self._say("active set: tightening stalled")
            return None
        if try_exact():
            return found[0]
        if not self.bang_bang:
            return None
        if advance(log_eps_min, np.zeros(self.nc), on_step=try_exact) and found:
            return found[0]
        return found[0] if found else None

    def _junction_text(self, shoot, v):
        if isinstance(shoot, MultiArcShooting) and shoot._nested:
            return " ".join(f"{ev}:{self.kr.CONSTRAINT_NAMES[shoot.arcs[i][1]]}"
                            for ev, i in shoot.events) + " | " + self._junction_text_plain(shoot, v)
        return self._junction_text_plain(shoot, v)

    def _junction_text_plain(self, shoot, v):
        arcs, values = self._arcs_of(shoot, v)
        return " ".join(f"{self.kr.CONSTRAINT_NAMES[c]}@" + (
            f"{val[0]:.5f}" if kind == "touch" else f"[{val[0]:.5f},{val[1]:.5f}]")
            for (kind, c), val in zip(arcs, values)) or "-"

    def _names(self, arcs):
        return ", ".join(f"{kind} {self.kr.CONSTRAINT_NAMES[c]}" for kind, c in arcs) or "free"

    # -- the whole solve --------------------------------------------------------

    def solve(self, params, costates, final_time=None, **options):
        """From a costate distribution to the exact extremal.

        ``params`` are the values of the stated problem's parameters; the
        policy, if one applies, is handled here (see :class:`Homotopy`).
        The options are those of :meth:`_solve`. Returns a
        :class:`HomotopyResult`.
        """
        params = np.asarray(params, dtype=float).reshape(-1)
        if len(params) != len(self.stated.params):
            raise ValueError(f"expected {len(self.stated.params)} parameters "
                             f"{[str(p) for p in self.stated.params]}, got {len(params)}")
        if self._stages is None:
            r = self._solve(params, costates, final_time, **options)
        else:
            r = self._lexicographic(params, costates, final_time, **options)
        r.policy = self.policy_text
        return r

    def _lexicographic(self, params, costates, final_time, **options):
        """Each component's minimum time, then minimum effort at ``(1 + delta) T*``."""
        stages, times = [], []
        for h, idx in self._stages:
            if callable(costates):
                part = lambda rng, f=costates, idx=idx: np.asarray(f(rng))[idx]
            else:
                part = tuple(np.asarray(b, dtype=float)[idx] for b in costates)
            r = h.solve(params, part, final_time, **options)
            stages.append(r)
            if not r.exact:
                r0 = HomotopyResult(r.solution, r.shooting, (), False, np.nan, None,
                                    r.steps, r.tries,
                                    f"the minimum time of component {len(stages)} was not "
                                    f"found: {r.message}", stages=stages)
                return r0
            times.append(r.solution.tf)
        tf = (1.0 + self.policy.delta) * max(times)
        self._say(f"lexicographic: component minimum times {times}, t_f = {tf}")
        r = self._solve(np.append(params, tf), costates, None, **options)
        r.stages = stages
        r.message = (f"t_f = (1 + {self.policy.delta}) * {max(times):.12g}; " + r.message)
        return r

    def _solve(self, params, costates, final_time=None, *, n_tries=200, seed=0, eps0=1.0,
               s_max=10.0, max_steps=300, n_jitter=6, tol=1e-10):
        """The automatic solve of ``self.problem`` (the stated problem with its policy).

        ``params`` are the values of ``problem.params``; penalty weights the
        problem defines are not used (:func:`regularize` replaces them), so
        any value will do for them. ``costates`` is ``(low, high)`` (uniform
        per component) or a function ``rng -> lam0``; ``final_time``
        likewise for ``t_f`` (required when it is free). At most ``n_tries``
        random draws are made, from ``seed`` (an int or a
        ``numpy.random.Generator``). ``eps0`` is the first smoothing weight
        and ``s_max`` how many decades ``eps`` may fall; ``n_jitter`` is the
        number of extra seeds, with perturbed junction times, per candidate
        structure. Returns a :class:`HomotopyResult`.
        """
        params = np.asarray(params, dtype=float).reshape(-1)
        if len(params) != len(self.problem.params):
            raise ValueError(f"expected {len(self.problem.params)} parameters "
                             f"{[str(p) for p in self.problem.params]}, got {len(params)}")
        self.params = params
        if self.kr.FREE_FINAL_TIME and final_time is None:
            raise ValueError("the final time is free: give a final_time distribution")
        broken = self._limits_broken_at_the_ends()
        if broken:
            return HomotopyResult(None, None, (), False, np.nan, None, [], 0,
                                  f"infeasible: {'; '.join(broken)}")
        rng = np.random.default_rng(seed)
        kr = self.kr
        eps_start = [eps0, 10 * eps0, 0.1 * eps0] if self.bang_bang else [None]
        zero = np.zeros(self.nc)
        sol, tries = None, 0
        if kr.FREE_FINAL_TIME and not self.bang_bang:
            shoot = SingleShooting(kr, self.regularized_params(None, zero), **self.options)
            # With the penalties off the limits do not enter, so their values can change.
            key = tuple(params[self._free_arc_params()])
            if key != self._seed_key:
                self._seed_key = key
                self._seed = self._separate_seed(shoot, final_time, tol)
            seed_v = self._seed
            if seed_v is not None:
                # To the path's accuracy: the exact solve at the end polishes it.
                shoot = self._smoothed([], None, zero)
                shoot.max_nfev = self.attempt_nfev
                candidate = shoot.solve(seed_v, tol=max(tol, 1e-6), options=_budget("hybr", 80))
                if candidate.success:
                    sol = candidate
        for e0 in eps_start if sol is None else ():
            shoot = SingleShooting(kr, self.regularized_params(e0, zero), **self.options)
            sol, n = shoot.multistart(self._sampler(shoot, costates, final_time),
                                      n_tries=n_tries, seed=rng, tol=tol)
            tries += n
            if sol is not None and sol.success:
                eps0 = e0
                break
        if sol is None or not sol.success:
            return HomotopyResult(sol, None, (), False, np.nan, None, [], tries,
                                  "no random draw solved the regularized problem")
        self._say(f"regularized solve after {tries} draws, eps = {eps0}")

        # Step 3: continuation in phases. The penalty of a newly violated limit
        # ramps up first, with eps held (log10 rho_c from rho_ref_c 10^m_start
        # to rho_ref_c 10^m_max); then eps falls, with rho held.
        log_eps = np.log10(eps0) if self.bang_bang else 0.0
        log_eps_min = log_eps - s_max
        rho_ref = np.zeros(self.nc)
        m = np.full(self.nc, -np.inf)      # log10(rho / rho_ref); -inf: penalty off
        m_start, m_max = -6.0, 4.0

        def params_at(le, mm):
            eps = 10.0 ** le if self.bang_bang else None
            rho = np.where(np.isfinite(mm), rho_ref * 10.0 ** mm, 0.0)
            return self.regularized_params(eps, rho), eps, rho

        def activate(shoot, v):
            """Switch on the penalty of every newly violated limit (at m_start)."""
            if not self.nc:
                return False
            viol = shoot.constraint_violation(v)
            new = (viol > 1e-9) & ~np.isfinite(state["m"])
            if not new.any():
                return False
            x0, lam0, _, tf = shoot.unpack(v)
            tr = kr.trajectory(shoot.t0, np.concatenate([x0, lam0]), tf, shoot.params,
                               n_points=2001, **self.options)
            J = abs(self._regularized_cost(shoot, v)) or 1.0
            for c in np.flatnonzero(new):
                P = np.trapezoid(np.maximum(tr["g"][:, c], 0.0) ** 2 / 2, tr["t"])
                rho_ref[c] = J / max(P, 1e-300)
                self._say(f"limit {kr.CONSTRAINT_NAMES[c]} violated: rho_ref = {rho_ref[c]:.3g}")
            return new

        steps = []
        state = {"le": log_eps, "m": m.copy(), "v": sol.v, "sol": sol, "prev": None}
        best = None

        def record_and_attempt(le, mm, v, sol):
            p, eps, rho = params_at(le, mm)
            shoot = SingleShooting(kr, p, **self.options)
            step = Step(log_eps - le, eps if eps is not None else np.nan, rho.copy(),
                        sol.residual_norm)
            steps.append(step)
            read = self._read(shoot, v)
            accepted, tried, _ = self._attempt(read, v[:shoot.n_unknowns], n_jitter, rng, tol)
            step.candidates = tried
            self._say(f"eps = {eps}  rho = {rho}  candidates tried: {tried}, "
                      f"accepted: {len(accepted)}")
            if accepted:
                step.accepted = True
            return shoot, accepted

        def advance(move, size):
            """One adaptive step along ``move(state, size) -> (le, m)``: the new
            state, or None when the step falls below its floor."""
            ds = size
            while ds >= 1e-3 and time.perf_counter() < state["deadline"]:
                le, mm = move(ds)
                p, _, _ = params_at(le, mm)
                nxt = SingleShooting(kr, p, **self.options)
                v, prev = state["v"], state["prev"]
                guesses = [v]
                if prev is not None and prev[2] > 0:
                    guesses.insert(0, v + (v - prev[1]) * ds / prev[2])
                for g in guesses:
                    try:
                        r = nxt.solve(g, tol=tol)
                    except _FAILURES:
                        continue
                    if r.success:
                        state.update(le=le, m=mm, prev=(None, v, ds), v=r.v, sol=r)
                        return nxt, ds
                ds *= 0.5
            return None, ds

        if not self.bang_bang and not self.nc:
            # The regularized problem is the exact one.
            J = self.objective(shoot, sol.v)
            return HomotopyResult(sol, shoot, (), True, J, shoot.params,
                                  [Step(0.0, np.nan, np.zeros(0), sol.residual_norm)], tries,
                                  "exact: no bang-bang controls or path constraints")
        # The active-set route goes first unless the exact problem may have
        # singular arcs, which only the penalty route reads; each route gets
        # the other as its fallback.
        first = sol
        kx = self.exact_kernels
        # The active-set route needs every limit to have exact arcs (order 1 or 2).
        can_activate = self.nc and all(self.kr.EXACT_ARCS)
        active_first = can_activate and not any(kx.SINGULAR_ARCS)

        def active_route():
            hit = self._active_set(first, eps0, s_max, steps, tol)
            if hit is None:
                return None
            J, xs, xsol = hit
            return HomotopyResult(xsol, xs, self._entries(xs), True, J, xs.params, steps,
                                  tries, f"exact after {len(steps)} steps (active set)")

        if active_first:
            out = active_route()
            if out is not None:
                return out
            if self.kr.system.riding is not None:
                # A stiff penalty on limits this large takes minutes per step, and
                # reads back the structures the active-set route has already tried.
                return HomotopyResult(sol, shoot, (), False, np.nan, shoot.params, steps, tries,
                                      "no exact structure: the active-set route stalled")
            self._say("active set: no exact structure; trying the penalty route")
        state["deadline"] = time.perf_counter() + self.penalty_seconds
        new = activate(shoot, sol.v)
        if new is not False:
            state["m"][new] = m_start
        shoot, accepted = record_and_attempt(state["le"], state["m"], state["v"], state["sol"])
        ds_rho, ds_eps = 0.5, 0.25
        done = np.zeros(self.nc, dtype=bool)
        while not accepted and len(steps) < max_steps and (
                time.perf_counter() < state["deadline"]):
            best = (shoot, state["sol"])
            ramping = np.isfinite(state["m"]) & (state["m"] < m_max) & ~done
            if ramping.any():
                def move(ds):
                    mm = state["m"].copy()
                    mm[ramping] = np.minimum(mm[ramping] + ds, m_max)
                    return state["le"], mm
                nxt, ds = advance(move, ds_rho)
                if nxt is None:
                    # The penalty stalls: hold it where it is and go on lowering eps.
                    self._say("penalty continuation stalled")
                    done |= ramping
                    state["prev"] = None
                    continue
                ds_rho = min(ds * 1.5, 1.0)
            elif self.bang_bang and state["le"] > log_eps_min:
                def move(ds):
                    return max(state["le"] - ds, log_eps_min), state["m"]
                nxt, ds = advance(move, ds_eps)
                if nxt is None:
                    break
                ds_eps = min(ds * 1.5, 1.0)
            else:
                break
            new = activate(nxt, state["v"])
            if new is not False:
                state["m"][new] = m_start
                state["prev"] = None
            shoot, accepted = record_and_attempt(state["le"], state["m"], state["v"],
                                                 state["sol"])
        if accepted:
            J, xs, xsol = min(accepted, key=lambda a: a[0])
            return HomotopyResult(xsol, xs, self._entries(xs), True, J, xs.params, steps,
                                  tries, f"exact after {len(steps)} steps")
        if can_activate and not active_first:
            self._say("penalty route: no exact structure; trying the active-set route")
            out = active_route()
            if out is not None:
                return out
        shoot, sol = best if best is not None else (shoot, state["sol"])
        return HomotopyResult(sol, shoot, (), False, np.nan, shoot.params, steps, tries,
                              "no exact structure accepted before the continuation ended")

    def _regularized_cost(self, shoot, v):
        x0, lam0, _, tf = shoot.unpack(v)
        r = shoot.kernels.propagate(shoot.t0, np.concatenate([x0, lam0]), tf, shoot.params,
                                    cost=True, **self.options)
        xf = r["z"][:shoot.kernels.N_STATES]
        return float(r["cost"]) + float(self._terminal(tf, list(xf), list(self.params)))

    @staticmethod
    def _entries(shoot):
        """The structure as MultiArcShooting would be given it."""
        if not isinstance(shoot, MultiArcShooting):
            return ()
        k = shoot.kernels
        if not shoot._legacy and shoot._nested:
            out = []
            for ev, i in shoot.events:
                kind, c, _ = shoot.arcs[i]
                out.append(({"in": "enter", "out": "exit"}.get(ev, ev), k.CONSTRAINT_NAMES[c]))
            return tuple(out)
        out = []
        for kind, c, extra in shoot.arcs:
            if kind in ("boundary", "touch"):
                out.append((kind, k.CONSTRAINT_NAMES[c]))
                continue
            name = k.system.switching_functions[c].control.name
            sides = tuple("upper" if f else "lower" for f in extra)
            if kind == "switch":
                out.append(("switch", name))
            elif kind == "singular_joined":
                out.append(("singular", name, *sides, "joined"))
            else:
                out.append(("singular", name, *sides))
        return tuple(out)


def _newton(system, guess, tol, steps=10, accept=0.0):
    """Damped Newton on ``system(v, with_jacobian)``: the root, or None.

    The step length is tried on residuals alone, and the Jacobian found where it lands.
    A caller that only needs a guess can pass ``accept``: the residual at which a stalled
    iteration still counts, since the noise of a loose integration can sit above ``tol``.
    """
    v = np.asarray(guess, dtype=float)
    try:
        R = system(v, False)
    except _FAILURES:
        return None
    best = float(np.linalg.norm(R))
    for _ in range(steps):
        if best < tol:
            return v
        try:
            _, J = system(v, True)
            delta = np.linalg.lstsq(J, R, rcond=None)[0]
        except (*_FAILURES, np.linalg.LinAlgError):
            return None
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            w = v - scale * delta
            try:
                Rw = system(w, False)
            except _FAILURES:
                continue
            if np.linalg.norm(Rw) < best:
                v, R, best = w, Rw, float(np.linalg.norm(Rw))
                break
        else:
            return v if best < accept else None
    return v if best < max(tol, accept) else None


def _draw(spec, rng):
    """One sample of a distribution: a function of ``rng``, or ``(low, high)``."""
    if callable(spec):
        return spec(rng)
    lo, hi = (np.asarray(b, dtype=float) for b in spec)
    return rng.uniform(lo, hi)


def _scaled(tol, v):
    """``tol`` relative to the size of the unknowns (absolute while they are below 1):
    costates in the thousands cannot be resolved to 1e-10 absolutely."""
    return tol * max(1.0, float(np.abs(np.asarray(v, dtype=float)).max(initial=0.0)))


def _budget(method, n):
    """MINPACK options with an evaluation budget."""
    if method == "lm":
        return {"xtol": 1e-14, "ftol": 1e-14, "maxiter": int(n)}
    return {"xtol": 1e-13, "maxfev": int(n)}


def solve(problem, params, costates, final_time=None, *, cache_dir=None, verbose=False,
          policy="auto", **options):
    """:meth:`Homotopy.solve` in one call: see :class:`Homotopy` and the module docs."""
    return Homotopy(problem, cache_dir=cache_dir, verbose=verbose, policy=policy).solve(
        params, costates, final_time, **options)
