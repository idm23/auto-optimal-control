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
     there: a newborn touch point (order 2 or more) or short boundary arc
     (order 1) with zero jump; a touch point violated beside it grows into a
     boundary arc or splits into two (order 3 or more: another touch point,
     since such limits have no arcs with a smooth control law); neighbouring arcs of one limit merge;
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
   become boundary arcs (or touch points, for order 2 or more), and stretches where
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

When no candidate is accepted (a limit of order 4 or more, a structure outside
what :class:`MultiArcShooting` represents, a search out of time),
:func:`solve` returns the last regularized solution with ``exact = False``. A
fixed end state that already violates a limit is reported the same way, at once.
"""

import itertools
import time
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import sympy as sp

from auto_optimal_control.build import build_kernels
from auto_optimal_control.policy import Lexicographic, SecondaryCost, analyze, resolve
from auto_optimal_control.pontryagin import OUTLINE_ABOVE_OPS, bang_bang_controls, derive
from auto_optimal_control.problem import Box, OptimalControlProblem, PathConstraint
from auto_optimal_control.diagnostics import Diagnostics
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

def _copy(problem, *, name, running_cost, params, path_constraints):
    return OptimalControlProblem(
        name=name, time=problem.time, states=problem.states, controls=problem.controls,
        dynamics=problem.dynamics, running_cost=running_cost,
        terminal_cost=problem.terminal_cost, initial_state=dict(problem.initial_state),
        terminal_constraints=problem.terminal_constraints, initial_time=problem.initial_time,
        final_time=problem.final_time, params=params, control_sets=problem.control_sets,
        path_constraints=path_constraints, certified=problem.certified)


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
    #: What the solve ran into (:class:`~auto_optimal_control.diagnostics.Diagnostics`).
    diagnostics: dict = field(default_factory=dict)

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
    outline_above : int or None
        Path constraints with more operations than this go on a tape and
        their arcs are solved at run time (:func:`~auto_optimal_control.pontryagin.derive`).
    """

    def __init__(self, problem, *, cache_dir=None, rtol=1e-12, atol=1e-12, verbose=False,
                 policy="auto", outline_above=OUTLINE_ABOVE_OPS):
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
                                      verbose=verbose, policy="as_stated",
                                      outline_above=outline_above), idx)
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
        self.active_set_backtracks = 8     # structures the active-set route may back out of
        # Where the limit offsets start: "together" (just above every peak) or "staggered"
        # (rule R1: a random head start per limit, drawn with offset_seed).
        self.offset_start = "together"
        self.offset_seed = 0
        # Rule R4: when a solve fails and the problem is certified time-reversible, solve
        # the reversed problem and map its extremal back (Lemma F).
        self.reversal = True
        self.split_rising = True
        self.holdable_births = True
        self.width_steps = False
        self.merge_touching = True
        self.offset_order = "joint"
        #: Rule R15: routes to try after the first fails, as attribute settings (each
        #: gets an equal share of the time left); empty for one route.
        self.portfolio = []
        self.stall_seconds = 60.0
        #: Run the portfolio's routes at once in forked processes (one core each).
        self.parallel_routes = False
        self.corner_touches = False         # rule R12: refuted by Lemma L (see the note)
        #: Rule R16: after a step that overshoots a birth, place the next one at the
        #: birth by a secant on the peak excess (Lemma N), not by halving.
        self.locate_births = False
        #: Rule R17: when a step fails while an arc shrinks, extrapolate w^2 (linear
        #: in s by Lemma E) to the arc's death and step past it without the arc.
        self.locate_deaths = False
        #: Diagnostics of the last solve; snapshots of failed steps go to ``snapshot_dir``
        #: (None: none), named by ``snapshot_tag``; ``check_lemmas`` runs the lemma
        #: oracles on every path solution (slow: for development).
        self.diagnostics = Diagnostics()
        self.snapshot_dir = None
        self.snapshot_tag = None
        self.check_lemmas = False
        #: Path solves the active-set route may make (a budget that does not depend on
        #: the machine; ``active_set_seconds`` is the wall-clock one).
        self.max_path_solves = 1500        # rule R2: neutral on the census (see the note)
        self.edge_exits = False         # rule R8: tried, no gain (see the note)
        self._certificates = None
        self.penalty_seconds = 300.0       # time budget of the penalty route
        self.path_options = dict(rtol=rtol, atol=atol)    # along the active-set route
        self.path_evaluations = 240        # MINPACK evaluations per solve on that route
        self.path_fallback = 1e-2          # a chord step that ends below this earns MINPACK
        # Picking a new structure: with parallel > 1, each candidate and `variants` jittered
        # copies of it are solved at once in forked processes, each stopped after
        # candidate_seconds, and the first candidate (in order) that converged is taken.
        self.parallel = 0
        self.variants = 3
        self.candidate_seconds = 2.0
        self.bang_bang = bang_bang_controls(problem)
        self.regularized, self.eps_symbol, self.rho_symbols = regularize(problem, self.bang_bang)
        # Limits ridden at run time have no penalty route, and a penalty's tape in every
        # kernel takes most of the compile.
        self.outline_above = outline_above
        self.kr = build_kernels(derive(self.regularized, penalties_when_riding=False,
                                       outline_above=outline_above), cache_dir=cache_dir)
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
                self._kx = build_kernels(derive(exact(self.problem), singular=True,
                                                outline_above=self.outline_above),
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
                elif kx.CONSTRAINT_ORDERS[c] >= 3:
                    kinds = ["touch"]          # no arcs at order 3 or more
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
        """Whether an exact solution passes the checks it can make. A converged one
        that does not is counted in ``diagnostics.rejected`` by the check it failed."""
        reason = self._verdict(shoot, sol)
        if reason is not None and sol.success and np.isfinite(sol.residual_norm):
            self.diagnostics.reject(reason)
        return reason is None

    def _verdict(self, shoot, sol):
        """None if an exact solution passes every check, else the check it fails."""
        if not sol.success or not np.isfinite(sol.residual_norm):
            return "not converged"
        if sol.max_violation > 1e-7:
            return "limit violated off its arcs"
        if not isinstance(shoot, MultiArcShooting) or not shoot.arcs:
            return None
        if not shoot.in_order(sol.v):
            return "junctions out of order"
        try:
            d = shoot.diagnostics(sol.v)
        except _FAILURES:
            return "diagnostics failed"
        small = 1e-7
        if any(nu < -small for nu in d["touch_nu"]):
            return "touch jump negative"
        if any(lo < -small * max(1.0, abs(hi)) for lo, hi in d["boundary_mu"]):
            return "mu negative (Lemma H)"
        # Order 1: mu = int_t^{t_out} eta with eta >= 0, so mu must not rise along an arc
        # (docs/notes/structure_certificates.md, Lemma H).
        order1 = [q == 1 for kind, _, q in shoot.arcs if kind == "boundary"]
        if any(o and rise > 1e-6 * max(1.0, abs(hi))
               for o, rise, (_, hi) in zip(order1, d["boundary_mu_rise"], d["boundary_mu"])):
            return "mu rises (Lemma H)"
        if any(lo < -small * max(1.0, abs(hi)) for lo, hi in d["singular_legendre"]):
            return "Legendre condition"
        if any(w > 1e-6 for w in d["held_sign_violation"].values()):
            return "held control sign"
        return None

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
        shoot = (MultiArcShooting(self.kr, p, arcs, max_nfev=self.attempt_nfev, open_end=True,
                                  **self.path_options)
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

    def _birth_fraction(self, shoot, v, overshoot, scale, target=2e-4):
        """Rule R16: the fraction of an overshooting step at which the first limit's
        peak excess reaches ``target`` (relative to ``scale``), by a secant between the
        consistent solution ``v`` and the overshooting one, or None.

        Before a birth the structure is fixed, and the peak excess is a C^1 function of
        the route parameter with derivative the explicit one at the peak (Lemma N), so
        the secant puts the step within O(ds^2) of the birth. Only the step length
        changes; the step is solved and judged as any other."""
        if not overshoot:
            return None
        try:
            samples = self._free_samples(shoot, v)
        except _FAILURES:
            return None
        best = None
        for c, p1 in overshoot.items():
            vals = [g[:, c] for t, g, active in samples if c not in active]
            if not vals:
                continue
            p0 = float(max(np.max(x) for x in vals)) / scale[c]
            if not p1 > target > p0:
                continue
            theta = (target - p0) / (p1 - p0)
            best = theta if best is None else min(best, theta)
        return None if best is None else float(np.clip(best, 0.05, 0.9))

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

    def _merge_touching(self, arcs, values):
        """Rule R11: boundary arcs of one limit that touch or overlap are one arc.

        When the gap between two arcs of a limit closes (a split whose stretch off the
        limit dies, or a predictor carrying one arc into the next), their union is the
        set of times on the limit, so they are one arc. Its entry jump carries both,
        since mu = int d eta adds over the pieces (the convention of ``_activate``)."""
        arcs, values = list(arcs), list(values)
        merged = self.merge_touching
        while merged:
            merged = False
            order = sorted((i for i, (kind, _) in enumerate(arcs) if kind == "boundary"),
                           key=lambda i: float(values[i][0]))
            for i, j in zip(order, order[1:]):
                if arcs[i][1] != arcs[j][1]:
                    continue
                (a0, a1, pa), (b0, b1, pb) = values[i], values[j]
                if b0 <= a1:
                    keep = (a0, max(a1, b1), np.asarray(pa, dtype=float) + np.asarray(pb, dtype=float))
                    values[i] = keep
                    del arcs[j], values[j]
                    self.diagnostics.fired("R11")
                    merged = True
                    break
        return arcs, values

    def _mas_entries(self, arcs, values, names=False):
        """MultiArcShooting entries for arcs that may overlap: ``(entries, values)``.

        The events (entries, exits, touches) are ordered by their times, so a
        limit that comes into play while another is ridden becomes an arc
        inside or across that one; the values follow the order of entry.
        """
        arcs, values = self._merge_touching(arcs, values)
        # A predictor can carry a short arc past zero length: it stays an arc, just
        # barely, and the solve or the retirement rule decides what becomes of it.
        values = [(v[0], max(v[1], v[0] + 1e-10 * max(1.0, abs(v[0]))), *v[2:])
                  if kind == "boundary" and np.isfinite(v[0]) else v
                  for (kind, _), v in zip(arcs, values)]
        events = []
        for i, ((kind, c), val) in enumerate(zip(arcs, values)):
            if kind == "touch":
                events.append((float(val[0]), 1, "touch", i))
            else:
                events.append((float(val[0]), 1, "enter" if np.isfinite(val[0]) else "start", i))
                if np.isfinite(val[1]):
                    events.append((float(val[1]), 0, "exit", i))
        events.sort(key=lambda e: (e[0], e[1]))
        label = (lambda c: self.kr.CONSTRAINT_NAMES[c]) if names else (lambda c: c)
        depth, overlap = 0, False
        for _, _, ev, _ in events:
            if ev != "exit" and depth > 0:
                overlap = True             # an entry or touch while an arc is ridden
            depth += 1 if ev in ("enter", "start") else -1 if ev == "exit" else 0
        if any(ev == "start" for _, _, ev, _ in events) or depth > 0:
            overlap = True                 # only the event list can say "start" or leave one open
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
        # An arc ridden from the start has t_in = -inf, one ridden to the end t_out = inf.
        values = []
        for i, val in enumerate(shoot.unpack_arcs(v)):
            if shoot.arcs[i][0] == "boundary":
                val = (-np.inf if i in shoot._started else val[0],
                       np.inf if np.isnan(val[1]) else val[1], val[2])
            values.append(val)
        return [(kind, c) for kind, c, _ in shoot.arcs], values

    def _activate(self, shoot, v, violations):
        """Structures that bring the violated limits in, most likely first.

        A violation away from the limit's arcs becomes a newborn touch point
        (order 2; always for order 3 or more, which has no arcs) or a short
        boundary arc (order 1), with zero jump. A
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
        t_start = shoot.t0

        def on_limit(c, which):
            return self._on_limit(shoot, v, c, which)

        def span(i):
            kind, val = arcs[i][0], values[i]
            return (val[0], val[0]) if kind == "touch" else (val[0], val[1])

        groups, born = [], []              # groups: [set of arc indices, [(a, b), ...]]
        for c, a, b, _ in violations:
            hits = {i for i, (_, cc) in enumerate(arcs)
                    if cc == c and a - near <= span(i)[1] and b + near >= span(i)[0]}
            q = kr.CONSTRAINT_ORDERS[c]
            if q == 1 and not hits:
                # On the limit at an end of the motion: the arc reaches that end.
                if a <= t_start + near and on_limit(c, 0):
                    born.append((("boundary", c), (-np.inf, b, np.zeros(1))))
                    continue
                if b >= t_start + T - near and on_limit(c, -1):
                    born.append((("boundary", c), (a, np.inf, np.zeros(1))))
                    continue
            if q >= 3:
                # Order 3 or more: touch points only (no boundary arcs with a smooth
                # control law; Jacobson, Lele & Speyer 1971), so each violation is one more.
                born.append((("touch", c), (0.5 * (a + b), 0.0)))
                continue
            if not hits:
                if q == 2:
                    born.append((("touch", c), (0.5 * (a + b), 0.0)))
                else:
                    w = max(b - a, 1e-3 * T) / 2
                    mid = 0.5 * (a + b)
                    lo, hi = mid - w, mid + w
                    t_edge = self._corner_peak(shoot, v, c, lo, hi) if self.corner_touches else None
                    if t_edge is not None:
                        # Lemma L: a peak on the edge is born as a touch point, not an arc.
                        self.diagnostics.fired("R12")
                        born.append((("touch", c), (t_edge, 0.0)))
                        continue
                    if self.holdable_births:
                        lo, hi = self._holdable(shoot, v, c, lo, hi)
                    born.append((("boundary", c), (lo, hi, np.zeros(q))))
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
                inset = 1e-3 * min(other[1] - other[0], T)
                lo = max(val[0], other[0] + inset) if entry_in else val[0]
                hi = min(val[1], other[1] - inset) if exit_in else val[1]
                out.append((entry, (lo, hi, val[2])))
            return out

        def build(split, newborn=None, narrow=False):
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
                if narrow and len(idx) == 1 and kind == "touch":
                    # A touch point becomes a short arc about its time: at order 2 the
                    # arc born is much shorter than the stretch first violated.
                    tc = val[0]
                    w = 0.1 * max(min(tc - lo, hi - tc), 1e-6 * T)
                    lo, hi = tc - w, tc + w
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
            options.append(build(False, narrow=True))
        return options

    def _on_limit(self, shoot, v, c, which):
        """Whether the true limit ``c`` holds at the start (0) or the end (-1) of the motion."""
        p = np.array(shoot.params, dtype=float)
        p[len(p) - self.nc:] = 0.0                 # the offsets come last
        _, _, segs = shoot._evaluate(np.asarray(v, dtype=float), jac=False)
        t, z = (shoot.t0, segs[0][4]) if which == 0 else (segs[-1][3], segs[-1][-1]["z"])
        return abs(float(shoot.kernels.constraint(t, z, p)[c])) < 1e-5

    def _end_structure(self, shoot, v):
        """The structure with arcs next to an end of the motion running out to it.

        As the offsets reach zero with an end state on a limit, the free stretch
        between that arc and the end shrinks to nothing, which no step can
        cross: the arc must become one ridden from the start or to the end.
        Returns ``(arcs, values)``, or None if no arc qualifies."""
        if not isinstance(shoot, MultiArcShooting) or shoot.kernels.N_SWITCHING:
            return None
        arcs, values = self._arcs_of(shoot, v)
        tf = shoot.unpack(v)[3]
        near = 0.1 * (tf - shoot.t0)
        out, changed = [], False
        for (kind, c), val in zip(arcs, values):
            if kind == "boundary" and shoot.kernels.CONSTRAINT_ORDERS[c] == 1:
                if np.isfinite(val[1]) and tf - val[1] < near and self._on_limit(shoot, v, c, -1):
                    val, changed = (val[0], np.inf, val[2]), True
                if np.isfinite(val[0]) and val[0] - shoot.t0 < near and self._on_limit(shoot, v, c, 0):
                    val, changed = (-np.inf, val[1], val[2]), True
            out.append(val)
        return (arcs, out) if changed else None

    def _bound_options(self, shoot, v):
        """Structures that undo an overlap of arcs whose riding control leaves its bounds.

        Riding as many limits as controls fixes every control, so a stretch that
        needs more than the bound allows cannot be ridden. One arc must then end
        before that stretch, or start after it; the junction there has a control
        at its bound, so ``H`` continuity holds at it. Returns a list of
        ``(arcs, values)``, or ``None`` when every riding control is in bounds."""
        if not isinstance(shoot, MultiArcShooting) or len(shoot.arcs) < 2:
            return None
        arcs, values = self._arcs_of(shoot, v)
        tr = shoot.trajectory(v, 801)
        k, p = shoot.kernels, shoot.params
        lo, hi = (np.array(b, dtype=float) for b in zip(*[k.control_bounds(t, z, p)
                                                          for t, z in zip(tr["t"], tr["z"])]))
        over = np.maximum(tr["u"] - hi, lo - tr["u"]).max(axis=1)
        bad = (over > 1e-6 * np.maximum(hi - lo, 1e-12).max(axis=1)) & (tr["active"].sum(axis=1) >= 2)
        if not bad.any():
            return None
        t_bad = tr["t"][bad]
        ta, tb = float(t_bad[0]), float(t_bad[-1])
        margin = 1e-3 * (tr["t"][-1] - tr["t"][0])
        hit = [i for i, ((kind, _), val) in enumerate(zip(arcs, values))
               if kind == "boundary" and val[0] <= tb and val[1] >= ta]
        options = []
        for i in hit:
            a, b, pi = values[i]
            for new in ((a, ta - margin, pi), (tb + margin, b, pi)):
                if new[1] - new[0] > margin:
                    options.append((arcs, [new if j == i else w for j, w in enumerate(values)]))
        return options or None

    def _jittered(self, arcs, values, rng):
        """A copy of a structure's values with junction times and jumps perturbed."""
        out = []
        for (kind, _), val in zip(arcs, values):
            if kind == "boundary":
                a, b, pi = val
                width = (b - a) if np.isfinite(a) and np.isfinite(b) else 0.05
                a2 = a + rng.uniform(-0.25, 0.25) * width if np.isfinite(a) else a
                b2 = b + rng.uniform(-0.25, 0.25) * width if np.isfinite(b) else b
                out.append((a2, b2, np.asarray(pi, dtype=float) * rng.uniform(0.5, 1.5)))
            elif kind == "touch":
                t, nu = val
                out.append((t * (1 + rng.uniform(-0.01, 0.01)), nu * rng.uniform(0.5, 1.5)))
            else:
                out.append(val)
        return out

    def _routes_in_parallel(self, members, first, eps0, s_max, steps, tol):
        """Rule R15 in parallel: each route of the portfolio in a forked process with the
        whole time budget; the first exact answer is solved again here (and accepted or
        not by the same checks). ``(J, shoot, sol)`` or None."""
        import multiprocessing as mp
        ctx = mp.get_context("fork")

        def child(conn, member):
            try:
                for k_, val_ in member.items():
                    setattr(self, k_, val_)
                self.portfolio = []
                hit = self._active_set(first, eps0, s_max, steps, tol,
                                       max_solves=self.max_path_solves)
                out = None
                if hit is not None:
                    _, xs, xsol = hit
                    out = (self._entries(xs), np.asarray(xsol.v, dtype=float))
                conn.send((out, self.diagnostics.as_dict()))
            except Exception:
                conn.send((None, {}))
            conn.close()

        running = []
        for n_, member in enumerate(members):
            parent, kid = ctx.Pipe(duplex=False)
            p = ctx.Process(target=child, args=(kid, member), daemon=True)
            p.start()
            kid.close()
            running.append((n_, p, parent))
        deadline = time.perf_counter() + self.active_set_seconds + 30.0
        found = None
        while running and found is None and time.perf_counter() < deadline:
            time.sleep(0.05)
            for item in list(running):
                n_, p, conn = item
                if not conn.poll():
                    if not p.is_alive():
                        running.remove(item)
                    continue
                try:
                    out, diag = conn.recv()
                except EOFError:
                    out, diag = None, {}
                running.remove(item)
                p.join(timeout=1)
                for part in ("evaluations", "steps", "rejected", "rules", "oracles"):
                    for key_, cnt in diag.get(part, {}).items():
                        table = getattr(self.diagnostics, part)
                        table[key_] = table.get(key_, 0) + cnt
                if out is not None:
                    entries, v = out
                    hit = self._exact_jump(list(entries), v, tol)
                    if hit is not None:
                        if n_:
                            self.diagnostics.fired("R15")
                        self._say(f"active set: route {n_ + 1} of {len(members)} found it")
                        found = hit
                        break
        for _, p, conn in running:
            p.kill()
            p.join(timeout=1)
            conn.close()
        return found

    def _speculate(self, options, base, le, off, solve_structure, state):
        """Solve candidate structures at once in forked processes; the first in order
        that converges within ``candidate_seconds``, as ``((arcs, values), shoot, sol)``."""
        import multiprocessing as mp
        rng = np.random.default_rng(len(options) + int(1e6 * float(np.sum(np.abs(off)))) % 2 ** 31)
        tasks = []
        for arcs, values in options:
            tasks.append((arcs, values))
            tasks += [(arcs, self._jittered(arcs, values, rng)) for _ in range(self.variants)]
        ctx = mp.get_context("fork")

        def child(conn, arcs, values):
            try:
                _, r = solve_structure(arcs, values, base, le, off)
                conn.send((bool(r.success), float(r.residual_norm), r.v))
            except Exception:
                conn.send((False, float("inf"), None))
            conn.close()

        results = {}
        pending = list(enumerate(tasks))
        running = {}
        while pending or running:
            while pending and len(running) < self.parallel:
                i, (arcs, values) = pending.pop(0)
                parent, kid = ctx.Pipe(duplex=False)
                p = ctx.Process(target=child, args=(kid, arcs, values), daemon=True)
                p.start()
                kid.close()
                running[i] = (p, parent, time.perf_counter())
            time.sleep(0.01)
            for i, (p, conn, started) in list(running.items()):
                if conn.poll():
                    try:
                        results[i] = conn.recv()
                    except EOFError:
                        results[i] = (False, float("inf"), None)
                elif p.is_alive() and time.perf_counter() - started < self.candidate_seconds:
                    continue
                elif p.is_alive():
                    results[i] = (False, float("inf"), None)
                else:
                    results.setdefault(i, (False, float("inf"), None))
                if p.is_alive():
                    p.kill()
                p.join(timeout=1)
                conn.close()
                del running[i]
            # Stop early once the best-ranked candidate still possible has converged.
            done = [i for i in sorted(results) if results[i][0]]
            if done and all(j in results for j in range(done[0])):
                for p, conn, _ in running.values():
                    p.kill()
                    p.join(timeout=1)
                    conn.close()
                running.clear()
                pending.clear()
        state["solves"] += len(results)
        per = self.variants + 1
        converged = [i for i in sorted(results) if results[i][0]]
        for i in sorted(results):
            ok, _, v = results[i]
            if not ok:
                continue
            arcs, values = tasks[i]
            original = tasks[(i // per) * per][1]
            moved = [(round(float(w[0] - o[0]), 6), round(float(w[1] - o[1]), 6) if np.isfinite(w[1]) and np.isfinite(o[1]) else None,
                      round(float(np.atleast_1d(w[2])[0] / np.atleast_1d(o[2])[0]), 3) if len(w) > 2 and np.atleast_1d(o[2])[0] else None)
                     if kind == "boundary" else None for (kind, _), w, o in zip(arcs, values, original)]
            self._detail(f"    parallel pick: option {i // per} of {len(options)}, "
                         f"{'original' if i % per == 0 else f'jittered copy {i % per}'}; "
                         f"converged {[(j // per, j % per) for j in converged]} of {len(tasks)}; "
                         f"residuals of the originals {[round(results[j][1], 9) for j in range(0, len(tasks), per) if j in results]}; "
                         f"moved (d t_in, d t_out, pi ratio) {moved}")
            entries, _ = self._mas_entries(arcs, values)
            sh = self._smoothed(entries, 10.0 ** le if self.bang_bang else None, off)
            r = sh.solve_chord(v, tol=max(1e-6, 0.0)) if self.kr.system.riding is not None else \
                sh.solve(v, tol=1e-6, options=_budget("hybr", self.path_evaluations))
            if r.success:
                return (arcs, values), sh, r
        return None

    def _retire(self, shoot, v):
        """The structure without arcs that have died: a boundary arc shrunk to
        nothing becomes a touch point (order 2 or more) or goes (order 1), and a touch
        point whose jump has the wrong sign goes. ``None`` if nothing changes."""
        arcs, values = self._arcs_of(shoot, v)
        if not arcs:
            return None
        T = shoot.unpack(v)[3] - shoot.t0
        # Junctions that crossed (rule R10 lets the residual through them): the same arcs,
        # with their events sorted again, is a structure change.
        crossed = isinstance(shoot, MultiArcShooting) and not shoot.in_order(v)
        out_a, out_v, changed = [], [], crossed
        for (kind, c), val in zip(arcs, values):
            if kind == "boundary" and val[1] - val[0] < 1e-6 * T:
                changed = True
                if self.kr.CONSTRAINT_ORDERS[c] >= 2:
                    out_a.append(("touch", c))
                    out_v.append((0.5 * (val[0] + val[1]), float(val[2][0])))
                continue
            if kind == "touch" and val[1] < -1e-9:
                changed = True
                continue
            out_a.append((kind, c))
            out_v.append(val)
        return (out_a, out_v) if changed else None

    def _corner_peak(self, shoot, v, c, a, b, rel=1e-2):
        """The time of the peak of limit ``c`` on ``[a, b]`` if it lies on the edge of the
        holdable set (margin within ``rel`` of the jerks' reach there: the equality case
        of Lemma I'), else None."""
        from auto_optimal_control.lemmas import LimitObjects
        try:
            obj = LimitObjects(shoot.kernels, shoot.params)
            tr = shoot.trajectory(v, n_points=4001)
        except Exception:
            return None
        inside = np.flatnonzero((tr["t"] >= a) & (tr["t"] <= b))
        if len(inside) < 3:
            return None
        j0 = inside[int(np.argmax(tr["g"][inside, c]))]
        t0, z0 = float(tr["t"][j0]), tr["z"][j0]
        a_, d_ = obj.rate_parts(c, t0, z0)
        lo_, hi_ = (np.asarray(x, dtype=float) for x in shoot.kernels.control_bounds(t0, z0, shoot.params))
        reach = float(np.abs(a_) @ (0.5 * (hi_ - lo_)))
        m = obj.margin(c, t0, z0)
        return t0 if abs(m) <= rel * max(reach, 1e-300) else None

    def _holdable(self, shoot, v, c, a, b):
        """The stretch of ``[a, b]`` about the peak of limit ``c`` on which it can be held.

        Where a limit of order 1 is reached at an interior time, its one-sided rates
        have opposite signs, so (the control set being convex) some admissible control
        makes the rate zero: every contact lies in the set where the riding problem is
        feasible (Lemma I of ``docs/notes/structure_certificates.md``). In particular the
        peak of a violation does, up to the sampling. A newborn arc seeded where the
        limit cannot be held fails to integrate, so its ends are pulled in to the
        stretch about the peak where the riding control exists (the kernel's riding
        solve is exact: it fails only when no admissible control holds the limit)."""
        k, p = shoot.kernels, shoot.params
        try:
            tr = shoot.trajectory(v, n_points=4001)
        except Exception:
            return a, b
        ts, zs = tr["t"], tr["z"]
        inside = np.flatnonzero((ts >= a) & (ts <= b))
        if len(inside) < 3:
            return a, b
        g = tr["g"][inside, c]
        j0 = int(np.argmax(g))

        def ok(j):
            try:
                k.boundary_control(c, ts[inside[j]], zs[inside[j]][:2 * k.N_STATES], p)
                return True
            except Exception:
                return False

        if not ok(j0):
            # The peak is on the edge (equality in Lemma I's corollary: the free control
            # is at the corner there), so the arc is born one-sided, on the side where
            # the limit becomes holdable: start from the nearest holdable sample.
            near = [j for j in sorted(range(len(inside)), key=lambda j: abs(j - j0)) if ok(j)]
            if not near:
                return a, b
            j0 = near[0]
            self.diagnostics.fired("R6 one-sided")
        lo = j0
        while lo > 0 and ok(lo - 1):
            lo -= 1
        hi = j0
        while hi < len(inside) - 1 and ok(hi + 1):
            hi += 1
        if lo == 0 and hi == len(inside) - 1:
            return a, b
        T = shoot.unpack(v)[3] - shoot.t0
        lo_t, hi_t = float(ts[inside[lo]]), float(ts[inside[hi]])
        if hi_t - lo_t < 1e-4 * T:
            mid = float(ts[inside[j0]])
            lo_t, hi_t = mid - 5e-5 * T, mid + 5e-5 * T
        self.diagnostics.fired("R6")
        self._detail(f"    {self.kr.CONSTRAINT_NAMES[c]}: held only on [{lo_t:.5f}, {hi_t:.5f}] "
                     f"of [{a:.5f}, {b:.5f}]: the newborn arc is placed there")
        return lo_t, hi_t

    def _solve_structure(self, arcs, values, base, le, off, path_tol, state=None):
        """The path problem at ``(le, off)`` with structure ``(arcs, values)``, solved
        from ``base``: ``(shoot, sol)``. ``state["solves"]`` counts the solves."""
        if state is None:
            state = {"solves": 0}
        entries, values = self._mas_entries(arcs, values)
        sh = self._smoothed(entries, 10.0 ** le if self.bang_bang else None, off)
        guess = sh.pack_arcs(base, values) if arcs else np.asarray(base, dtype=float)
        state["solves"] += 1
        if isinstance(sh, MultiArcShooting) and (sh._open or sh._started):
            # On an arc that reaches a fixed end on the limit, the terminal row and the
            # riding say the same thing: rank-deficient, which needs the damped solver.
            return sh, sh.solve_damped(guess, tol=path_tol, max_steps=40, max_seconds=30.0)
        options = _budget("hybr", self.path_evaluations)
        if self.kr.system.riding is not None:
            # Limits ridden at run time make every evaluation dear: a cheap attempt
            # first, and only a step that gets close earns the slower solve.
            r = sh.solve_chord(guess, tol=path_tol)
            if (not r.success and r.residual_norm >= 1e5 and self.edge_exits and arcs
                    and isinstance(sh, MultiArcShooting)):
                # The seed did not propagate. If an arc rides past the edge of the
                # holdable set, its exit is pulled in to the edge (Lemma I).
                pulled = sh.pull_in_exits(guess)
                if pulled is not None:
                    self.diagnostics.fired("R8")
                    self._detail("    exits pulled in to the edge of the holdable set")
                    guess = pulled
                    state["solves"] += 1
                    r = sh.solve_chord(guess, tol=path_tol)
            if r.success or not r.residual_norm < self.path_fallback:
                return sh, r
        r = sh.solve(guess, tol=path_tol, options=options)
        if (not r.success and r.residual_norm < 1e-2 and arcs and sh.tangent_junctions):
            # A control at its bound at a junction jumps there: H continuity, not tangency.
            state["solves"] += 1
            sh.tangent_junctions = False
            alt = sh.solve(guess, tol=path_tol, options=options)
            if alt.success:
                return sh, alt
            sh.tangent_junctions = True
        return sh, r

    def _width_step(self, shoot, v, frac, point, factor, tol, iterations=10, guess=None,
                    arc=None):
        """Rule R2: a step in a young arc's width, with the route's parameter solved for.

        Near a birth (or death) of an order-1 arc its width moves like the square root
        of the route's parameter (Lemma E), so steps in the parameter stall while steps
        in the width are regular. The narrowest finite order-1 arc's width is set to
        ``factor`` times its current value and the system ``[Phi(v; s), w - w_target]``
        is solved for ``(v, s)``, with ``point(s) = (le, off)`` the route's line.
        ``guess`` is a predicted ``(v, s)`` to start from, ``arc`` the arc's index.
        Returns ``(v, s, arc, w, w_target)`` or None."""
        if not isinstance(shoot, MultiArcShooting):
            return None
        T = shoot.unpack(v)[3] - shoot.t0
        best = None
        for i, ((kind, c, q), sl) in enumerate(zip(shoot.arcs, shoot._arc_slices)):
            if kind != "boundary" or q != 1 or i in shoot._started or i in shoot._open:
                continue
            if arc is not None and i != arc:
                continue
            w = v[sl.start + 1] - v[sl.start]
            if 0 < w < 0.05 * T and (best is None or w < best[1]):
                best = (i, w, sl)
        if best is None:
            return None
        i, w, sl = best
        target = factor * w
        p_saved = shoot.params.copy()

        def params(sv):
            le, off = point(sv)
            return np.asarray(self.regularized_params(10.0 ** le if self.bang_bang else None,
                                                      np.zeros(self.nc), off), dtype=float)

        def resid(x, jac):
            shoot.params = params(x[-1])
            if jac:
                R, J = shoot.residual_jacobian(x[:-1])
            else:
                R, J = shoot.residual(x[:-1]), None
            row = x[sl.start + 1] - x[sl.start] - target
            return np.append(R, row), J

        if guess is not None:
            x = np.append(np.asarray(guess[0], dtype=float), float(guess[1]))
        else:
            x = np.append(np.asarray(v, dtype=float), frac)
        mid = 0.5 * (x[sl.start] + x[sl.start + 1])
        x[sl.start], x[sl.start + 1] = mid - 0.5 * target, mid + 0.5 * target
        try:
            for _ in range(iterations):
                R, J = resid(x, True)
                err = float(np.linalg.norm(R))
                if not np.isfinite(err):
                    return None
                if err < tol:
                    break
                h = 1e-7 * max(1.0, abs(x[-1]))
                xs = x.copy()
                xs[-1] += h
                Rs, _ = resid(xs, False)
                col = (Rs[:-1] - R[:-1]) / h
                n = len(x) - 1
                A = np.zeros((n + 1, n + 1))
                A[:n, :n] = J
                A[:n, n] = col
                A[n, sl.start + 1], A[n, sl.start] = 1.0, -1.0
                dx = np.linalg.lstsq(A, -R, rcond=None)[0]
                step = 1.0
                for _ in range(6):
                    try:
                        Rt, _ = resid(x + step * dx, False)
                        if np.linalg.norm(Rt) < (1 - 0.25 * step) * err:
                            break
                    except _FAILURES:
                        pass
                    step *= 0.5
                else:
                    return None
                x = x + step * dx
            else:
                return None
        except _FAILURES:
            return None
        finally:
            shoot.params = p_saved
        self.diagnostics.fired("R2")
        self._detail(f"    width step: arc {i} from {w:.3e} to {target:.3e}, route parameter "
                     f"{frac:.6f} -> {x[-1]:.6f}")
        return x[:-1], float(x[-1]), i, w, target

    def _death_points(self, shoot, v1, v0, s1, s0, s_fail):
        """Rule R17: ``[(arc index, s*)]`` for finite order-1 arcs that shrink from ``v1``
        (at ``s1``) to ``v0`` (at ``s0``) and, by a secant in ``w^2``, die before
        ``s_fail``, the step that failed; earliest death first.

        By Lemma E an arc's width obeys ``s - s* = kappa w^2 + O(w^3)`` near its birth
        or death, with the branch smooth in ``w``: so ``w^2`` is C^1 in ``s`` with a
        nonzero slope, and the secant through two points of the path finds ``s*`` to
        ``O(ds^2)``. Past ``s*`` the arc is gone; the step without it is solved and
        judged as any other."""
        if not isinstance(shoot, MultiArcShooting) or not s0 > s1:
            return []
        a0, val0 = self._arcs_of(shoot, v0)
        a1, val1 = self._arcs_of(shoot, v1)
        if a0 != a1:
            return []
        T = shoot.unpack(v0)[3] - shoot.t0
        out = []
        for i, ((kind, c), x0, x1) in enumerate(zip(a0, val0, val1)):
            if kind != "boundary" or self.kr.CONSTRAINT_ORDERS[c] != 1 or len(x0) < 2:
                continue
            if i in shoot._open or i in shoot._started:
                continue                   # an arc run out to an end has no death
            w0, w1 = float(x0[1] - x0[0]), float(x1[1] - x1[0])
            if not (w1 > w0 > 0 and w0 < 0.05 * T):
                continue
            s_star = s0 + (s0 - s1) * w0 ** 2 / (w1 ** 2 - w0 ** 2)
            if s0 < s_star < s_fail:
                out.append((i, s_star))
        return sorted(out, key=lambda e: e[1])

    def _width_moves(self, shoot, v, frac, point, tol):
        """Rule R2's moves at a stall, from Lemma E's normal form ``s = s* + kappa w^2``.

        A step to half the young arc's width tells which side of the event the route is
        on. If ``s`` moves forward the arc is dying: ``w^2`` is linear in ``s``, so the
        death is at ``s* = s_half + (s_half - s) / 3``; the moves are the structure
        without the arc just past ``s*``, then the half step. If ``s`` moves back the
        arc is being born: ``kappa`` and ``s*`` from the two points predict ``s`` at
        ``1.5 w``, and ``v`` is predicted linearly in ``w`` (the branch is smooth in
        ``w``). Yields ``(v, s, dropped arc or None)``, best first."""
        half = self._width_step(shoot, v, frac, point, 0.5, tol)
        if half is None:
            return
        v_h, s_h, i, w0, w_h = half
        if s_h > frac:
            s_star = s_h + (s_h - frac) * w_h ** 2 / (w0 ** 2 - w_h ** 2)
            beyond = s_star + 0.25 * (s_star - frac)
            if s_star < 1.0:
                self.diagnostics.fired("R2 death")
                self._detail(f"    width steps: arc {i} dies at s = {s_star:.6f}; past it without")
                yield v, min(beyond, 1.0), i
            yield v_h, s_h, None
            return
        kappa = (frac - s_h) / (w0 ** 2 - w_h ** 2)
        if not kappa > 0:
            return
        s_star = frac - kappa * w0 ** 2
        w_t = 1.5 * w0
        s_t = s_star + kappa * w_t ** 2
        v_t = v + (v - v_h) * (w_t - w0) / (w0 - w_h)
        grown = self._width_step(shoot, v, frac, point, 1.5, tol, guess=(v_t, s_t), arc=i)
        if grown is not None and frac < grown[1] <= 1.0:
            self.diagnostics.fired("R2 birth")
            yield grown[0], grown[1], None

    def _split_rising(self, shoot, v, n_points=801):
        """Structures with an order-1 boundary arc split where its multiplier rises.

        On an order-1 arc ``mu(t) = int_t^{t_out} d eta`` with ``eta >= 0`` (Lemma H of
        ``docs/notes/structure_certificates.md``, which holds for each limit separately
        while others are active too). Where ``mu`` rises, ``eta < 0``: the limit must
        leave there. The arc becomes two, with the gap over the stretch of the largest
        rise (running minimum to maximum); the second arc's entry jump is ``mu`` where it
        starts. A list of ``(arcs, values)``, or ``None`` if no arc rises."""
        if not isinstance(shoot, MultiArcShooting):
            return None
        k, p = shoot.kernels, shoot.params
        arcs, values = self._arcs_of(shoot, v)
        tr = shoot.trajectory(v, n_points)
        T = shoot.unpack(v)[3] - shoot.t0
        best = None
        for i, ((kind, c), val) in enumerate(zip(arcs, values)):
            if kind != "boundary" or self.kr.CONSTRAINT_ORDERS[c] != 1:
                continue
            on = np.flatnonzero(tr["active"][:, c] & (tr["t"] >= val[0]) & (tr["t"] <= val[1]))
            if len(on) < 3:
                continue
            mu = []
            for j in on:
                aid = tr["arc"][j]
                m_ = k.boundary_multiplier(aid, tr["t"][j], tr["z"][j], p)
                mu.append(float(np.atleast_1d(m_)[0]) if aid < k.N_CONSTRAINTS
                          else float(np.atleast_1d(m_)[list(k.arc_constraints(aid)).index(c)]))
            mu = np.asarray(mu)
            run_min = np.minimum.accumulate(mu)
            rise = mu - run_min
            j_hi = int(np.argmax(rise))
            if rise[j_hi] <= 1e-6 * max(1.0, float(np.abs(mu).max())):
                continue
            j_lo = int(np.flatnonzero(mu[:j_hi + 1] == run_min[j_hi])[-1])
            t_lo, t_hi = float(tr["t"][on[j_lo]]), float(tr["t"][on[j_hi]])
            if t_hi - t_lo < 1e-6 * T or (best is not None and rise[j_hi] <= best[0]):
                continue
            best = (rise[j_hi], i, t_lo, t_hi, mu[j_lo], mu[j_hi])
        if best is None:
            return None
        _, i, t_lo, t_hi, mu_lo, mu_hi = best
        a, b, pi = values[i]
        self.diagnostics.fired("R5")
        self._detail(f"    {self.kr.CONSTRAINT_NAMES[arcs[i][1]]}: multiplier rises on "
                     f"[{t_lo:.5f}, {t_hi:.5f}] (eta < 0): split the arc")
        out_a = arcs[:i] + [arcs[i], arcs[i]] + arcs[i + 1:]
        # The first piece must leave with mu = 0: mu_new = mu - mu(t_lo) on it, so its
        # entry jump falls by mu(t_lo). The second enters with mu(t_hi).
        pi1 = np.atleast_1d(np.asarray(pi, float)).copy()
        pi1[0] -= mu_lo
        pi2 = np.atleast_1d(np.asarray(mu_hi, float))
        # The rise is often exactly another limit's arc (its junctions bound it): a gap
        # on those junctions would make events coincide (Lemma D), so the gap is
        # offered a little wider, then a little narrower.
        d = 0.02 * (t_hi - t_lo)
        out = []
        for lo, hi in ((t_lo - d, t_hi + d), (t_lo + d, t_hi - d)):
            if not (lo > a and hi < b):
                continue
            out.append((out_a, values[:i] + [(a, lo, pi1), (hi, b, pi2)] + values[i + 1:]))
        return out or None

    def _exact_jump(self, arcs, v, tol):
        """The exact problem's solve of structure ``arcs`` from ``v``: ``(J, shoot, sol)`` or None."""
        kx, px = self.exact_kernels, self.exact_params()
        tol = _scaled(tol, v)
        # A large problem's search has run on tangential junctions; if a control really does
        # jump at one, the rows that allow it are the ones to try next.
        for tangent in ((True, False) if kx.system.riding is not None else (False,)):
            try:
                if arcs:
                    shoot = MultiArcShooting(kx, px, arcs, max_nfev=self.attempt_nfev, open_end=True,
                                             **self.options)
                    method = "lm" if kx.N_SWITCHING else "hybr"
                else:
                    shoot = SingleShooting(kx, px, **self.options)
                    method = "hybr"
                shoot.tangent_junctions = tangent
                if arcs and (shoot._open or shoot._started):
                    sol = shoot.solve_damped(v, tol=tol, max_seconds=60.0)
                else:
                    # A good path solution is one or two Newton steps away; MINPACK can
                    # throw it far off.
                    sol = shoot.solve(v, method=method, tol=tol,
                                      options=_budget(method, 4 * self.attempt_evaluations))
                    if not sol.success and method == "hybr" and kx.system.riding is not None:
                        other = shoot.solve_chord(v, tol=tol)
                        sol = other if other.residual_norm < sol.residual_norm else sol
                if not sol.success and sol.residual_norm < SEED_ACCEPT * tol:
                    polished = shoot.solve_damped(sol.v, tol=tol, max_steps=10, max_seconds=30.0)
                    if polished.residual_norm < sol.residual_norm:
                        sol = polished
                    if not sol.success:
                        sol = replace(sol, success=True)   # stalled at the integrator's rounding
                if sol.success:
                    sol = replace(sol, max_violation=shoot._max_violation(sol.v))
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
        names = shoot.kernels.CONSTRAINT_NAMES
        violated = {names[c]: round(float(g), 6) for c, g in enumerate(d["free_arc_violation"])
                    if g > 1e-8}
        return (f"violated off their arcs {violated}, controls past their bounds on arcs "
                f"{np.round(d['boundary_bound_violation'], 6).tolist()}, touch nu "
                f"{np.round(d['touch_nu'], 6).tolist()}, mu rises "
                f"{[float(f'{x:.3g}') for x in d.get('boundary_mu_rise', [])]}, mu ranges "
                f"{[(round(a, 6), round(b, 6)) for a, b in d['boundary_mu']]}")

    def _active_set(self, sol, eps0, s_max, steps, tol, max_solves=1500):
        """The active-set route: from the unconstrained regularized solution ``sol``
        at ``eps0``, tighten every limit from just above the solution to its
        true value, bringing in arcs as limits come into play, then lower
        ``eps`` with the structure following, trying the exact problem at each
        step. Returns ``(J, shoot, sol)`` or None.

        Each limit has its own offset, so the offsets form a vector annealed to
        zero. They first fall together; when that stalls, typically because
        several limits come into play at once, each moves on alone, which
        separates the events. When none can move, the route returns to the
        last change of structure and rules out the structure chosen there."""
        deadline = time.perf_counter() + self.active_set_seconds
        banned, checkpoints, ended = set(), [], set()
        tried_splits = {}                  # split structure -> times tried (at most 3)

        def key(arcs, values):
            return tuple(self._mas_entries(arcs, values, True)[0])

        shoot = self._smoothed([], eps0, np.zeros(self.nc))
        (_, g, _), = self._free_samples(shoot, sol.v)
        # Violations are measured against how much each limit's g moves along the
        # unconstrained solution, or, when it hardly moves there (a joint the motion
        # barely uses), how far that solution is from the limit.
        scale = np.maximum(np.maximum(g.max(axis=0) - g.min(axis=0), np.abs(g.max(axis=0))), 1e-12)
        off0 = np.maximum(g.max(axis=0), 0.0) + 1e-3 * scale
        if self.offset_start == "staggered":
            # Rule R1 (docs/notes/structure_certificates.md, Lemmas C and D): offsets that
            # all start just above their peaks bring every limit in at once; a random head
            # start per limit separates the births, with probability one.
            peaks = np.maximum(g.max(axis=0), 0.0)
            draw = np.random.default_rng(self.offset_seed).uniform(0.5, 1.5, self.nc)
            off0 = peaks * (1.0 + draw) + 1e-3 * scale
        peaks0 = g.max(axis=0)
        state = {"shoot": shoot, "v": sol.v, "le": np.log10(eps0) if self.bang_bang else 0.0,
                 "off": off0.copy(), "solves": 0}
        log_eps_min = state["le"] - s_max
        state["best_tight"], state["progress_at"] = np.inf, time.perf_counter()

        def stalled():
            """Rule R15: no new low of the offsets for ``stall_seconds`` (with a portfolio,
            the next route gets the time that is left)."""
            if not self.portfolio or state["best_tight"] <= 1e-12:
                return False               # one route, or offsets at zero: the last phase
            return time.perf_counter() - state["progress_at"] > self.stall_seconds

        # The path only has to be followed closely, not solved to the last digit:
        # at an order-1 entry with a smooth control H continuity has a double
        # root, and Newton converges only linearly there. The exact jumps use ``tol``.
        path_tol = max(tol, 1e-6)

        def solve_structure(arcs, values, base, le, off):
            return self._solve_structure(arcs, values, base, le, off, path_tol, state)

        def settle(sh, r, le, off):
            """Bring in violated limits and retire dead arcs until the solution at
            (le, off) is consistent; ``(shoot, sol)`` or None."""
            last = None
            changed = False
            for _ in range(2 * self.nc + 4):
                optional = False
                viol = self._free_violations(sh, r.v, scale)
                if viol:
                    worst = max(peak / scale[c] for c, _, _, peak in viol)
                    if worst > 1e-3 and not changed:
                        self._detail("    violation too large: a shorter step ("
                                     + ", ".join(f"{self.kr.CONSTRAINT_NAMES[c]} {peak / scale[c]:.1e}"
                                                 f" at [{a:.5f}, {b:.5f}]"
                                                 for c, a, b, peak in viol) + ")")
                        state["why"] = "violation too large"
                        over = {}
                        for c, _, _, peak in viol:
                            over[c] = max(over.get(c, -np.inf), peak / scale[c])
                        state["overshoot"] = over
                        return None        # too far in: take a shorter step
                    # After a structure change a large violation is that change's doing
                    # (new arcs move the motion), which no shorter step undoes: it is
                    # brought in as it is.
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
                    if dead is None and self.split_rising:
                        dead = self._split_rising(sh, r.v)
                        # A split that does not solve leaves the step as it was: the
                        # exact solve's acceptance still judges the structure.
                        optional = dead is not None
                        if optional and tried_splits.get(key(*dead[0]), 0) >= 3:
                            dead, optional = None, False
                        elif optional:
                            tried_splits[key(*dead[0])] = tried_splits.get(key(*dead[0]), 0) + 1
                    if dead is not None:
                        options = dead if optional else [dead]
                    else:
                        options = self._bound_options(sh, r.v)
                        if options is None:
                            return sh, r
                base = r.v[:sh.n_base] if isinstance(sh, MultiArcShooting) else r.v
                options = [o for o in options if key(*o) not in banned]
                if self.parallel > 1 and options and not optional:
                    hit = self._speculate(options, base, le, off, solve_structure, state)
                    if hit is None:
                        state["why"] = "no structure converged"
                        return None
                    (arcs, values), nsh, nr = hit
                    self._detail(f"    try in parallel: {self._mas_entries(arcs, values, True)[0]}")
                    sh, r = nsh, nr
                    changed = True
                    self._say(f"  structure: {self._names(arcs)}")
                    continue
                for arcs, values in options:
                    if key(arcs, values) in banned:
                        continue
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
                    if optional:
                        return sh, r
                    state["why"] = "no structure converged"
                    return None
                sh, r = nsh, nr
                changed = True
                self._say(f"  structure: {self._names(arcs)}")
            state["why"] = "structure changes did not settle"
            return None

        def advance(target_le, target_off, on_step=None, ds_min=1e-6):
            """Towards the target along a straight line; False, at the last solution
            reached, if the step falls below ``ds_min``."""
            le0, off_start = state["le"], state["off"].copy()
            frac, ds = 0.0, 0.25
            prev = None                    # (frac, v, structure) of the solution before
            while frac < 1.0:
                if (state["solves"] > max_solves or ds < ds_min
                        or time.perf_counter() > deadline or stalled()):
                    self.diagnostics.step("stalled" if ds < ds_min else "out of budget")
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
                outcome, attempt = None, None
                for g in guesses:
                    base = g[:sh0.n_base] if isinstance(sh0, MultiArcShooting) else g
                    vals = self._arcs_of(sh0, g)[1] if arcs else values
                    attempt = (arcs, vals, base)
                    before = dict(self.diagnostics.evaluations)
                    state["why"] = None
                    try:
                        sh, r = solve_structure(arcs, vals, base, le, off)
                        self._detail(f"  step to {f:.6f} (ds {ds:.1e}): residual "
                                     f"{r.residual_norm:.1e}; junctions "
                                     f"{self._junction_text(sh, r.v)}")
                        done = settle(sh, r, le, off) if r.success else None
                        if r.success:
                            outcome = ("converged" if done is not None
                                       else f"settle failed: {state.get('why') or 'other'}")
                        elif r.residual_norm >= 1e5:
                            new = {k_: n_ - before.get(k_, 0)
                                   for k_, n_ in self.diagnostics.evaluations.items()}
                            worst = max(new, key=new.get) if any(new.values()) else "other"
                            outcome = f"seed failed: {worst}"
                        else:
                            outcome = "not converged"
                    except _FAILURES as exc:
                        self._detail(f"  step to {f:.6f} (ds {ds:.1e}): {exc}")
                        done = None
                        outcome = f"raised: {getattr(exc, 'failure', type(exc).__name__)}"
                    if done is not None:
                        break
                self.diagnostics.step(outcome)
                located = None
                if (done is None and self.locate_births
                        and outcome == "settle failed: violation too large"):
                    located = self._birth_fraction(sh0, v0, state.get("overshoot"), scale)
                if done is None and attempt is not None and self.snapshot_dir is not None:
                    cur_arcs, cur_vals = self._arcs_of(sh0, v0)
                    self.diagnostics.snapshot(self.snapshot_dir, self.snapshot_tag, outcome, {
                        "params": np.array(self.params, dtype=float), "path_tol": path_tol,
                        "frac": frac, "f": f, "ds": ds,
                        "before": {"arcs": cur_arcs, "values": cur_vals,
                                   "base": v0[:sh0.n_base] if isinstance(sh0, MultiArcShooting)
                                   else v0, "le": state["le"], "off": state["off"].copy()},
                        "attempt": {"arcs": attempt[0], "values": attempt[1], "base": attempt[2],
                                    "le": le, "off": np.array(off, dtype=float)}})
                if done is None and f >= 1.0 and not np.any(target_off) and tuple(map(tuple, arcs)) not in ended:
                    ended.add(tuple(map(tuple, arcs)))
                    converted = self._end_structure(sh0, v0)
                    if converted is not None:
                        try:
                            sh, r = solve_structure(*converted, v0[:sh0.n_base], le, off)
                            self._detail(f"  arcs run out to the ends: residual {r.residual_norm:.1e}")
                            done = settle(sh, r, le, off) if r.success else None
                        except _FAILURES as exc:
                            self._detail(f"  arcs run out to the ends: {exc}")
                if done is None and self.locate_deaths and prev is not None and prev[2] == arcs:
                    for drop, s_star in self._death_points(sh0, prev[1], v0, prev[0], frac, f):
                        fd = min(s_star + 0.25 * (s_star - frac), f)
                        le_d = le0 + fd * (target_le - le0)
                        off_d = off_start + fd * (target_off - off_start)
                        arcs_d, vals_d = self._arcs_of(sh0, v0)
                        arcs_d = arcs_d[:drop] + arcs_d[drop + 1:]
                        vals_d = vals_d[:drop] + vals_d[drop + 1:]
                        try:
                            sh, r = solve_structure(arcs_d, vals_d, v0[:sh0.n_base], le_d, off_d)
                            done = settle(sh, r, le_d, off_d) if r.success else None
                        except _FAILURES:
                            done = None
                        self._detail(f"    arc {drop} dies at {s_star:.6f}: "
                                     + ("stepped past it" if done is not None else "no"))
                        if done is not None:
                            self.diagnostics.fired("R17")
                            f, le, off = fd, le_d, off_d
                            break
                if done is None and self.width_steps and ds < 1e-3:
                    # Rule R2: steps in s stall at a birth or death; a step in the young
                    # arc's width is regular there (Lemma E).
                    def point(sv, le0=le0, off_start=off_start):
                        return (le0 + sv * (target_le - le0),
                                off_start + sv * (target_off - off_start))
                    for vw, fw, drop in self._width_moves(sh0, v0, frac, point, path_tol):
                        le_w, off_w = point(fw)
                        try:
                            arcs_w, vals_w = self._arcs_of(sh0, vw)
                            if drop is not None:
                                arcs_w = arcs_w[:drop] + arcs_w[drop + 1:]
                                vals_w = vals_w[:drop] + vals_w[drop + 1:]
                            sh, r = solve_structure(arcs_w, vals_w, vw[:sh0.n_base], le_w, off_w)
                            done = settle(sh, r, le_w, off_w) if r.success else None
                        except _FAILURES:
                            done = None
                        if done is not None:
                            f, le, off = fw, le_w, off_w
                            break
                if done is None:
                    if located is not None:
                        self.diagnostics.fired("R16")
                        self._detail(f"    birth located at {located:.2f} of the step")
                        ds = (f - frac) * located
                    else:
                        ds *= 0.5
                    continue
                prev = (frac, v0, arcs)
                changed = self._arcs_of(done[0], done[1].v)[0] != arcs
                if changed:
                    # What to return to if this new structure leads nowhere.
                    new_arcs, new_values = self._arcs_of(done[0], done[1].v)
                    checkpoints.append((dict(state), key(new_arcs, new_values)))
                state.update(shoot=done[0], v=done[1].v, le=le, off=off)
                tight = float(np.sum(np.asarray(off, dtype=float) / scale))
                if tight < state["best_tight"] - 1e-9:
                    state["best_tight"], state["progress_at"] = tight, time.perf_counter()
                if self.check_lemmas:
                    from auto_optimal_control.lemmas import check
                    try:
                        for lemma, n_bad in check(done[0], done[1].v).items():
                            if n_bad:
                                self.diagnostics.oracle(lemma)
                    except _FAILURES:
                        self.diagnostics.oracle("check failed")
                if changed:
                    prev = None            # the structure changed: no secant across it
                    state["progress_at"] = time.perf_counter()     # progress, for R15
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

        def out_of_budget():
            return state["solves"] > max_solves or time.perf_counter() > deadline or stalled()

        def anneal():
            """The vector of offsets to zero: together, then one by one, then back up."""
            backtracks = 0
            if self.offset_order == "sequential":
                # Rule R13: the limits come in one at a time, the one the free motion
                # breaks most first. A limit that slows the motion is then tight before
                # the ones whose holdable set shrinks with speed are born (Lemma I'), so
                # their births happen off the edge, as on the true path.
                excess = np.maximum(peaks0, 0.0) / scale
                for c in np.argsort(-excess):
                    if excess[c] <= 0 or out_of_budget():
                        continue
                    target = state["off"].copy()
                    target[c] = 0.0
                    self.diagnostics.fired("R13")
                    advance(state["le"], target, ds_min=1e-4)
            while True:
                if advance(state["le"], np.zeros(self.nc)):
                    return True
                if out_of_budget():
                    return False
                before = state["off"].copy()
                order = np.argsort(-before / scale)
                for c in order:
                    if before[c] <= 0 or out_of_budget():
                        continue
                    target = state["off"].copy()
                    target[c] = 0.0
                    advance(state["le"], target, ds_min=1e-3)
                moved = np.max(np.abs(state["off"] - before) / scale)
                if moved > 1e-3:
                    self._detail(f"    offsets moved one by one, by up to {moved:.1e} of their scale")
                    continue
                if not checkpoints or backtracks >= self.active_set_backtracks or out_of_budget():
                    return False
                saved, structure = checkpoints.pop()
                banned.add(structure)
                backtracks += 1
                solves = state["solves"]
                state.clear()
                state.update(saved, solves=solves)
                self._detail(f"    backing up: ruling out {structure}")

        self._say(f"active set: tightening the limits from offsets {np.round(off0, 6)}")
        if not anneal():
            self._say("active set: tightening stalled")
            return None
        if try_exact():
            return found[0]
        converted = self._end_structure(state["shoot"], state["v"])
        if converted is not None:
            try:
                sh, r = solve_structure(*converted, state["v"][:state["shoot"].n_base], state["le"],
                                        state["off"])
            except _FAILURES as exc:
                sh, r = None, None
                self._detail(f"  arcs run out to the ends: {exc}")
            if r is not None:
                self._detail(f"  arcs run out to the ends: residual {r.residual_norm:.1e}")
                if r.success:
                    state.update(shoot=sh, v=r.v)
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
        self.diagnostics = Diagnostics()
        sink, SingleShooting.sink = SingleShooting.sink, self.diagnostics.evaluations
        try:
            if self._stages is None:
                r = self._solve(params, costates, final_time, **options)
                if not r.exact and self.reversal:
                    back = self._through_reversal(params, costates, final_time, **options)
                    if back is not None:
                        self.diagnostics.fired("R4")
                        r = back
            else:
                r = self._lexicographic(params, costates, final_time, **options)
        finally:
            SingleShooting.sink = sink
        r.policy = self.policy_text
        r.diagnostics = self.diagnostics.as_dict()
        return r

    @property
    def certificates(self):
        """The certificates of the problem solved (:func:`~auto_optimal_control.certificates.certify`)."""
        if self._certificates is None:
            from auto_optimal_control.certificates import certify
            self._certificates = certify(self.problem)
        return self._certificates

    def _reversed_params(self, params):
        """The parameters of the reversed problem (rule R4), or None if the ends do not
        map to parameters: each end value must be a number or a parameter symbol."""
        c = self.certificates["time_reversible"]
        if not c:
            return None
        p = self.problem
        signs = dict(zip(p.states, c.detail["state_signs"]))
        index = {s: i for i, s in enumerate(p.params)}
        final = {}
        for e in p.terminal_constraints:
            for s in p.states:
                rest = sp.expand(s - e)
                if not (rest.free_symbols & set(p.states)):
                    final[s] = rest
        out = np.array(params, dtype=float)
        for s in p.states:
            if s not in p.initial_state or s not in final:
                return None
            e0, ef = sp.sympify(p.initial_state[s]), sp.sympify(final[s])
            r = signs[s]
            if e0.is_number and ef.is_number:
                if not (float(r * ef) == float(e0) and float(r * e0) == float(ef)):
                    return None
            elif e0 in index and ef in index:
                out[index[e0]], out[index[ef]] = r * params[index[ef]], r * params[index[e0]]
            else:
                return None
        return out

    def _through_reversal(self, params, costates, final_time, **options):
        """Rule R4 (Lemma F): solve the reversed problem and map its extremal back exactly.

        The reversed extremal's junction times reflect (t -> T - t), each arc keeps
        its jump and moves to the limit reversal maps it to, and the costates map
        as lambda(0) = -R lambda~(T); the exact problem is then solved from there."""
        if self.bang_bang:
            return None
        rev = self._reversed_params(params)
        if rev is None:
            return None
        self._say("reversal: solving the reversed problem (rule R4)")
        rr = self._solve(rev, costates, final_time, **options)
        if not rr.exact or not isinstance(rr.shooting, (MultiArcShooting, SingleShooting)):
            self._say("reversal: the reversed problem was not solved either")
            return None
        c = self.certificates["time_reversible"]
        R = np.array(c.detail["state_signs"], dtype=float)
        limit_map = c.detail["limit_map"]
        sh, v = rr.shooting, rr.solution.v
        n = sh.kernels.N_STATES
        segs = sh._evaluate(np.asarray(v, dtype=float), jac=False)[2] if isinstance(sh, MultiArcShooting) \
            else None
        if segs is not None:
            zT = segs[-1][-1]["z"]
        else:
            x0, lam0, _, tf_ = sh.unpack(v)
            zT = sh.kernels.propagate(sh.t0, np.concatenate([x0, lam0]), tf_, sh.params,
                                      **self.options)["z"]
        T = rr.solution.tf
        lam0 = -R * zT[n:]
        arcs, values = self._arcs_of(sh, v)
        f_arcs, f_vals = [], []
        for (kind, cc), val in zip(arcs, values):
            f_arcs.append((kind, limit_map[cc]))
            if kind == "boundary":
                a, b, pi = val
                f_vals.append((T - b, T - a, pi))
            else:
                f_vals.append((T - val[0], *val[1:]))
        self.params = np.asarray(params, dtype=float)
        kx, px = self.exact_kernels, self.exact_params()
        entries, f_vals = self._mas_entries(f_arcs, f_vals)
        if entries:
            shoot = MultiArcShooting(kx, px, entries, max_nfev=self.attempt_nfev, open_end=True,
                                     **self.options)
            base = shoot.pack(lam0, None, T if kx.FREE_FINAL_TIME else None)
            guess = shoot.pack_arcs(base, f_vals)
        else:
            shoot = SingleShooting(kx, px, **self.options)
            guess = shoot.pack(lam0, None, T if kx.FREE_FINAL_TIME else None)
        if kx.N_MULTIPLIERS:
            # The terminal multipliers enter linearly: fit them.
            Rv, Jv = shoot.residual_jacobian(guess)
            cols = slice(shoot.n_free + n, shoot.n_free + n + kx.N_MULTIPLIERS)
            guess[cols] += np.linalg.lstsq(Jv[:, cols], -Rv, rcond=None)[0]
        hit = self._exact_jump(entries, guess, _scaled(options.get("tol", 1e-10), guess))
        if hit is None:
            self._say("reversal: the mapped extremal was not accepted")
            return None
        J, xs, xsol = hit
        return HomotopyResult(xsol, xs, self._entries(xs), True, J, xs.params, rr.steps, rr.tries,
                              f"exact through the reversed problem (rule R4); {rr.message}")

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
            # Rule R15: a portfolio of routes. Each is a valid route (Lemma D: almost every
            # draw of the offsets separates the births), and which cases a route solves
            # depends on its path; on failure the next one gets the time that is left.
            members = [{}] + [dict(m_) for m_ in self.portfolio]
            if self.portfolio and self.parallel_routes:
                hit = self._routes_in_parallel(members, first, eps0, s_max, steps, tol)
                if hit is None:
                    return None
                J, xs, xsol = hit
                return HomotopyResult(xsol, xs, self._entries(xs), True, J, xs.params, steps,
                                      tries, f"exact after {len(steps)} steps (active set)")
            total, start = self.active_set_seconds, time.perf_counter()
            saved = {k_: getattr(self, k_) for m_ in members for k_ in m_}
            hit = None
            try:
                for n_, member in enumerate(members):
                    left = total - (time.perf_counter() - start)
                    if left <= 1.0:
                        break
                    for k_, val_ in saved.items():
                        setattr(self, k_, val_)
                    for k_, val_ in member.items():
                        setattr(self, k_, val_)
                    # Every route may use all the time left: a route that stops making
                    # progress (stall_seconds without a new low of its offsets) yields.
                    share = left
                    self.active_set_seconds = share
                    if n_:
                        self.diagnostics.fired("R15")
                        self._say(f"active set: route {n_ + 1} of {len(members)} ({member}), {share:.0f} s")
                    hit = self._active_set(first, eps0, s_max, steps, tol,
                                           max_solves=self.max_path_solves)
                    if hit is not None:
                        break
            finally:
                self.active_set_seconds = total
                for k_, val_ in saved.items():
                    setattr(self, k_, val_)
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
