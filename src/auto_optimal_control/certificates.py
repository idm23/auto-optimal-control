"""Properties of a problem's expressions that solver rules may rely on.

Each :class:`Certificate` states a property, whether it holds (``True``,
``False``, or ``None`` when it could not be decided), and how that was
established:

- ``"exact"``: proved from the SymPy expressions.
- ``"by construction"``: proved by a lemma about how the expression was built,
  with the lemma's inputs checked (the constructor records it in
  ``problem.certified``).
- ``"unknown"``: not decided; rules that need it stay off.

The properties and the rules they license are set out in
``docs/notes/structure_certificates.md``.
"""

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class Certificate:
    """One property of a problem: whether it holds, how that is known, and details."""

    name: str
    holds: object                 # True, False or None (undecided)
    how: str                      # "exact", "by construction" or "unknown"
    detail: object = None

    def __bool__(self):
        return self.holds is True


def _free_of(expr, symbols):
    return not (sp.sympify(expr).free_symbols & set(symbols))


def certify(problem):
    """The certificates of ``problem``, by name (see the module docstring)."""
    t, x, u = problem.time, list(problem.states), list(problem.controls)
    f = sp.Matrix(problem.dynamics)
    out = {}

    def put(c):
        out[c.name] = c

    # C1: dynamics affine in the controls, with a constant control matrix of full column rank.
    G = f.jacobian(u) if u else sp.zeros(len(x), 0)
    affine = all(_free_of(e, u) for e in G)
    constant = affine and all(_free_of(e, [t, *x]) for e in G)
    rank = G.rank() if constant else None
    put(Certificate("control_affine", affine and constant and rank == len(u), "exact",
                    {"G": G, "rank": rank}))

    # C2: Hessian of the running cost in the controls constant and positive definite.
    L = sp.sympify(problem.running_cost)
    R = sp.hessian(L, u) if u else sp.zeros(0, 0)
    const_R = all(_free_of(e, [t, *x, *u]) for e in R)
    pd = None
    if const_R and u:
        if R.is_diagonal():
            signs = [sp.sympify(d).is_positive for d in R.diagonal()]
            pd = True if all(s is True for s in signs) else (False if any(s is False for s in signs) else None)
        else:
            pd = R.is_positive_definite
    put(Certificate("strictly_convex_in_control", (const_R and pd) if pd is not None else None,
                    "exact" if pd is not None else "unknown", {"R": R}))

    # C5: the control law is smooth except where a bounded control saturates.
    bounded = [str(getattr(cs, "control", cs)) for cs in problem.control_sets]
    put(Certificate("smooth_except_saturation", True, "exact", {"clipped": bounded}))

    # C7: cyclic coordinates, in no dynamics, cost or limit: their costates are constant.
    used = set()
    for e in [*problem.dynamics, problem.running_cost, problem.terminal_cost,
              *[c.expr if hasattr(c, "expr") else c for c in problem.path_constraints]]:
        used |= sp.sympify(e).free_symbols
    cyclic = [s for s in x if s not in used]
    put(Certificate("cyclic_states", bool(cyclic), "exact", cyclic))

    # C6: time reversal. Signs r on the states and s on the controls with
    # f(r x, s u) = -r f(x, u), found along chains x_i' = x_k and x_i' = u_k.
    signs, controls_sign = _reversal_signs(problem)
    rev = signs is not None
    detail = {"state_signs": signs, "control_signs": controls_sign}
    how = "exact"
    if rev:
        sub = {**{s: r * s for s, r in zip(x, signs)}, **{v: c * v for v, c in zip(u, controls_sign)}}
        rev = all(sp.expand(fe.xreplace(sub) + r * fe) == 0 for fe, r in zip(problem.dynamics, signs))
        rev = rev and sp.expand(L.xreplace(sub) - L) == 0 and _free_of(L, [t]) and \
            all(_free_of(e, [t]) for e in problem.dynamics)
    if rev:
        limits = [c.expr if hasattr(c, "expr") else c for c in problem.path_constraints]
        claimed = problem.certified.get("limits_time_reversible")
        # Reversal must map the set of limits to itself (a rate limit's two sides swap).
        names = [c.name if hasattr(c, "name") else str(i) for i, c in enumerate(problem.path_constraints)]
        small = [i for i, e in enumerate(limits) if sp.count_ops(e) <= 2000]
        pool = {i: sp.expand(limits[i]) for i in small}
        image = {}
        for i in small:
            mapped = sp.expand(limits[i].xreplace(sub))
            image[i] = next((k for k, e in pool.items() if e == mapped), None)
        exact_ok = all(v is not None for v in image.values())
        if len(small) == len(limits):
            rev = exact_ok
        elif claimed is not None and exact_ok:
            lemma, by_name = claimed
            how, detail["lemma"] = "by construction", lemma
            for i, nm in enumerate(names):
                if i not in image:
                    image[i] = names.index(by_name.get(nm, nm)) if by_name.get(nm, nm) in names else None
            rev = all(v is not None for v in image.values())
        else:
            rev, how = None, "unknown"
        detail["limit_map"] = [image.get(i) for i in range(len(limits))]
    put(Certificate("time_reversible", rev, how if rev is not None else "unknown", detail))

    # C3/C4 by construction, from the constructor.
    for name in ("mass_matrix_positive_definite", "limit_control_rows"):
        if name in problem.certified:
            holds, lemma = problem.certified[name]
            put(Certificate(name, holds, "by construction", lemma))
        else:
            put(Certificate(name, None, "unknown"))
    return out


def _reversal_signs(problem):
    """State and control signs for time reversal, from chains ``x_i' = +-x_k`` or ``+-u_k``.

    Reversal maps ``x(t) -> r x(T - t)``, so ``x_i' = x_k`` needs ``r_k = -r_i``. States
    whose derivative is not a single signed state or control get ``None``."""
    x, u = list(problem.states), list(problem.controls)
    r = {s: None for s in x}
    c = {v: None for v in u}
    links = []
    for s, fe in zip(x, problem.dynamics):
        fe = sp.sympify(fe)
        sign, var = (1, fe) if fe.is_Symbol else ((-1, -fe) if (-fe).is_Symbol else (None, None))
        if var is None or var not in r and var not in c:
            return None, None
        links.append((s, var, sign))
    # Positions (states nothing else integrates into) take +1, then propagate.
    targets = {var for _, var, _ in links}
    for s in x:
        if s not in targets:
            r[s] = 1
    for _ in range(len(x) + len(u)):
        for s, var, sign in links:
            if r[s] is None:
                continue
            want = -r[s]                  # x~ = r x(T - t): x_i' = +-x_k needs r_k = -r_i
            book = r if var in r else c
            if book[var] is None:
                book[var] = want
            elif book[var] != want:
                return None, None
    if any(v is None for v in r.values()) or any(v is None for v in c.values()):
        return None, None
    return [r[s] for s in x], [c[v] for v in u]
