"""Print a :class:`~auto_optimal_control.pontryagin.PontryaginSystem` as plain C99.

:func:`generate_c` returns a header and a source file that depend only on
``<math.h>``. Every function returns ``0`` on success. Arrays are row-major;
``z = [x, lambda]`` (``NZ = 2 n`` entries), ``p`` is the parameter vector.

Functions, for a prefix ``P`` (``aoc_<name>_``)::

    int P_rhs(t, z, p, dz)                   z' = F(t, z, p), bang-bang controls switch inside
    int P_rhs_arc(t, z, p, arc, dz)          F on a fixed arc: arc[k] = 1 puts bang-bang
                                             control k at its upper bound, 0 at its lower one;
                                             arc[NSW + c] = 1 turns path constraint c's
                                             penalty on (NM = NSW + NC flags in all)
    int P_jac_arc(t, z, p, arc, A)           A = dF/dz        (NZ x NZ)
    int P_dfdp_arc(t, z, p, arc, B)          B = dF/dp        (NZ x NP)
    int P_stm_rhs(t, y, p, arc, with_p, dy)  y = [z, Phi (NZ x NZ), Psi (NZ x NP) if with_p]:
                                             Phi' = A Phi, Psi' = A Psi + B
    int P_stm_jump(k, t, y, p, arc, with_p)  apply the jump of Phi (and Psi) when mode k
                                             changes sign; arc is the arc before. A path
                                             constraint (k >= NSW) has no jump.
    int P_switching(t, z, p, S)              switching functions (NSW)
    int P_switching_grad(t, z, p, St, Sz, Sp)  dS/dt (NSW), dS/dz (NSW x NZ), dS/dp (NSW x NP)
    int P_constraint(t, z, p, g)             path constraints g (NC); g <= 0 is feasible
    int P_modes(t, z, p, M)                  the NM mode functions: S, then -g. Flag k of
                                             the arc is 1 exactly where M[k] < 0
    int P_kinks_arc(t, z, p, arc, K)         saturation guards (NK): zero where a clipped
                                             control meets its bound, so F has a kink there.
                                             Integration restarts at each
    int P_constraint_rate_arc(t, z, p, arc, gd)  dg/dt along the flow on a fixed arc (NC).
                                             Its zeros are the peaks and troughs of g:
                                             between them g is monotone and crosses zero
                                             at most once
    int P_control(t, z, p, u)                optimal control u* (NU)
    int P_control_bounds(t, z, p, lo, hi)    each control's bounds (NU each; +-HUGE_VAL
                                             where unbounded)
    int P_hamiltonian(t, z, p, H)            H at u* (one value)
    int P_cost_rate(t, z, p, L)              running cost at u* (one value)
    int P_cost_rate_arc(t, z, p, arc, L)     running cost on a fixed arc
    int P_residual(z0, zf, nu, tf, p, R)     boundary residuals (NR); tf is ignored when the
                                             final time is fixed
    int P_residual_jac(z0, zf, nu, tf, p, Rz0, Rzf, Rnu, Rtf, Rp)
                                             dR/dz0, dR/dzf (NR x NZ), dR/dnu (NR x NNU),
                                             dR/dtf (NR), dR/dp (NR x NP)

Exact boundary arcs of path constraint ``c`` get ``P_b<c>_*`` functions
(dispatched by ``P_boundary_*(c, ...)``), and singular arcs of switching
function ``k`` get ``P_s<k>_*`` (dispatched by ``P_singular_*(k, ...)``):
the dynamics at the singular control, its Jacobian, the control, ``H``, the
running cost, the Legendre-Clebsch coefficient ``-dS''/du``, and the
junction conditions ``(S, S')`` with their gradients in ``t`` and ``z``.

The jump uses the saltation matrix. Let switching function ``S`` cross zero
at ``t_s``, with ``F-`` and ``F+`` the dynamics on the arcs before and after
it and ``Sdot = dS/dt + dS/dz . F-``. Then
``Phi+ = Phi- + (F+ - F-) (dS/dz Phi-) / Sdot`` and
``Psi+ = Psi- + (F+ - F-) (dS/dz Psi- + dS/dp) / Sdot``.

A path constraint's penalty ``w/2 max(0, g)^2`` and its gradient vanish at
``g = 0``, so ``F`` is continuous across a crossing and the STM has no jump.
"""

import re

import numpy as np
import sympy as sp
from sympy.printing.c import C99CodePrinter

from auto_optimal_control.functions import greater, lesser
from auto_optimal_control.riding import riding_kernels
from auto_optimal_control.tape import OUTLINED_PREFIX, Outlined


class _Printer(C99CodePrinter):
    """C99 printer that expands small integer and half-integer powers."""

    def _print_Pow(self, expr):
        base, exp = expr.args
        if exp.is_Integer and 2 <= exp <= 4:
            b = self.parenthesize(base, 100)
            return "(" + "*".join([b] * int(exp)) + ")"
        if exp == -1:
            return f"(1.0/{self.parenthesize(base, 100)})"
        if exp.is_Integer and -4 <= exp <= -2:
            b = self.parenthesize(base, 100)
            return "(1.0/(" + "*".join([b] * int(-exp)) + "))"
        if exp == sp.Rational(1, 2):
            return f"sqrt({self._print(base)})"
        if exp == -sp.Rational(1, 2):
            return f"(1.0/sqrt({self._print(base)}))"
        if exp == sp.Rational(3, 2):
            b = self._print(base)
            return f"(({b})*sqrt({b}))"
        if exp == -sp.Rational(3, 2):
            b = self._print(base)
            return f"(1.0/(({b})*sqrt({b})))"
        return super()._print_Pow(expr)

    def _print_BooleanTrue(self, expr):
        return "1"

    def _print_BooleanFalse(self, expr):
        return "0"


_PRINTER = _Printer()


def _c(expr):
    return _PRINTER.doprint(expr)


def c_identifier(name):
    """A C identifier derived from ``name``."""
    ident = re.sub(r"\W", "_", name or "problem")
    if not ident or ident[0].isdigit():
        ident = "p_" + ident
    return ident


def _abs(a):
    return sp.Piecewise((a, a >= 0), (-a, True))


def _arc_smooth(expr):
    """:func:`_smooth` for boundary-arc kernels: Abs as Piecewise, and Min/Max as the
    unevaluated :mod:`~auto_optimal_control.functions` (clipped riding controls
    are built with those already; see there for why).
    """
    expr = sp.sympify(expr)
    if expr.has(sp.Abs):
        expr = expr.replace(sp.Abs, _abs)
    if expr.has(sp.Min, sp.Max):
        expr = expr.replace(sp.Min, lesser).replace(sp.Max, greater)
    return expr


def _smooth(expr):
    """Min/Max (saturated controls) and Abs as Piecewise, so they differentiate and print.

    SymPy differentiates ``Abs`` through ``re``/``im`` unless it can prove the
    argument real; as a Piecewise its derivative is the plain ``sign(a) a'``.
    """
    expr = sp.sympify(expr)
    if expr.has(sp.Abs):
        expr = expr.replace(sp.Abs, _abs)
    if expr.has(sp.Min, sp.Max):
        expr = expr.rewrite(sp.Piecewise)
    return expr


class _Function:
    """One generated C function: outputs computed through common subexpressions."""

    def __init__(self, signature, args, mapping=None):
        self.signature = signature
        self.args = args
        self.pre = []
        self.zero = []
        self.outputs = []
        self.mapping = mapping
        self.helpers = []

    def zero_fill(self, array, size):
        if size:
            self.zero.append(f"    for (int i_ = 0; i_ < {size}; ++i_) {array}[i_] = 0.0;")

    def out(self, lhs, expr):
        self.outputs.append((lhs, expr))

    def render(self, mapping):
        mapping = self.mapping or mapping
        lines = [self.signature + " {"]
        lines += [f"    (void){a};" for a in self.args]
        lines += self.pre
        lines += self.zero
        keep = [(lhs, e.xreplace(mapping)) for lhs, e in self.outputs]
        keep = [(lhs, e) for lhs, e in keep if not (self.zero and e == 0)]
        if keep:
            repl, reduced = sp.cse([e for _, e in keep], symbols=sp.numbered_symbols("cse"))
            lines += self._outlined_lines([e for _, e in repl] + list(reduced))
            for s, e in repl:
                lines.append(f"    const double {s} = {_c(e)};")
            for (lhs, _), e in zip(keep, reduced):
                lines.append(f"    {lhs} = {_c(e)};")
        lines.append("    return 0;")
        lines.append("}")
        return "\n".join(self.helpers + ["\n".join(lines)])

    def _outlined_lines(self, exprs):
        """Statements computing the :class:`~auto_optimal_control.tape.Outlined` steps used.

        What each input of the tape is called here comes from the functions' own
        arguments. The boundary residuals work on ``tf`` and ``zf``, which go by
        ``t`` and ``z`` in these statements, as they do everywhere else.
        """
        nodes = [node for e in exprs for node in sp.sympify(e).atoms(Outlined)]
        if not nodes:
            return []
        tape = nodes[0].tape
        names = {}
        for node in nodes:
            names.update(zip(tape.input_names(int(node.args[0])), map(str, node.args[1:])))
        boundary = any(text == "tf" or text.startswith("zf[") for text in names.values())
        if boundary:
            names = {name: "t" if text == "tf" else text.replace("zf[", "z[")
                     for name, text in names.items()}
        steps = sorted({int(node.args[0]) for node in nodes})
        prefix = "h_" + re.sub(r"\W", "_", self.signature.split("(")[0].split()[-1])
        helpers, body, operand = tape.c_block(steps, names, prefix, args="t, z, p, 0, 0")
        self.helpers = helpers
        head = ["    const double t = tf; const double* z = zf; (void)t; (void)z;"] if boundary else []
        return head + body + [f"    const double {OUTLINED_PREFIX}{k} = {operand(k)}; "
                              f"(void){OUTLINED_PREFIX}{k};" for k in steps]


def _arc_dynamics(pmp, flags):
    """Dynamics and running cost with each mode fixed by a 0/1 flag.

    On an arc, bang-bang control k equals ``flag_k * upper + (1 - flag_k) *
    lower``, a smooth expression, so derivatives with respect to parameters in
    the bounds come out right. Path constraint c's penalty is
    ``flag_{NSW + c} * w / 2 * g**2``.
    """
    nsw = len(pmp.switching_functions)
    law = {u: _smooth(e) for u, e in zip(pmp.control_law, _arc_laws(pmp, flags))}
    H = pmp.hamiltonian_full - pmp.penalty      # the penalty is added per arc below
    xdot = [f.subs(law) for f in pmp.problem.dynamics]
    lamdot = [-sp.diff(H, x).subs(law) for x in pmp.states]
    cost = pmp.problem.running_cost.subs(law)
    for c, flag in zip(pmp.path_constraints, flags[nsw:]):
        if c.penalty is None:
            continue
        cost += flag * c.penalty / 2 * c.expr ** 2
        lamdot = [e - flag * c.penalty * c.expr * sp.diff(c.expr, x)
                  for e, x in zip(lamdot, pmp.states)]
    return [_smooth(e) for e in xdot + lamdot], _smooth(cost)


def _kinks(laws):
    """Where a clipped law switches branch: ``a - b`` for each ``Min(a, b)`` or ``Max(a, b)``."""
    out = []
    for law in laws:
        for node in sorted(sp.sympify(law).atoms(sp.Min, sp.Max), key=sp.default_sort_key):
            args = node.args
            for i in range(len(args)):
                for j in range(i + 1, len(args)):
                    guard = _smooth(args[i] - args[j])
                    if guard.free_symbols and not any(sp.simplify(guard - o) == 0 or
                                                      sp.simplify(guard + o) == 0 for o in out):
                        out.append(guard)
    return out


def _stm_pattern(F, S, flags, z):
    """What ``Phi`` can hold on the fixed arcs of ``propagate``.

    ``(depends, reach)``, both ``NZ x NZ``: ``depends[i, k]`` when ``F[i]`` involves ``z[k]``,
    and ``reach[i, j]`` when ``z[i]`` can respond to its start ``z[j]``: by the flow, or by the
    jump at switch ``k``, which adds ``dF @ grad S_k`` to ``Phi`` wherever ``F`` holds flag ``k``.
    ``F`` has the path constraints' penalties off: on, they would tie everything together.
    """
    index = {s: i for i, s in enumerate(z)}
    n = len(z)
    depends = np.zeros((n, n), dtype=bool)
    for i, e in enumerate(F):
        depends[i, [index[s] for s in e.free_symbols if s in index]] = True
    edges = depends | np.eye(n, dtype=bool)
    for flag, e in zip(flags, S):
        rows = [i for i, f in enumerate(F) if flag in f.free_symbols]
        edges[np.ix_(rows, [index[s] for s in e.free_symbols if s in index])] = True
    reach = edges
    while True:
        wider = (reach.astype(int) @ reach.astype(int)) > 0
        if (wider == reach).all():
            return depends, reach
        reach = wider


def _sparse_stm(P, NZ, depends, reach):
    """``Phi`` as a list of the entries it can hold. FNI steps a flat vector, so it need not know.

    All of a step's work, its stages and its error, scales with the length of that vector, and
    on a free arc of a large problem most of ``Phi`` is zero for good. A jump goes through the
    dense code: jumps are rare.
    """
    NPHI = int(reach.sum())
    sig = (f"int {P}stm_sparse_rhs(double t, const double* y, const double* p, const int* arc, "
           "double* dy)",
           f"int {P}stm_sparse_jump(int k, double t, double* y, const double* p, const int* arc)",
           f"void {P}stm_indices(int* out)")
    if NPHI > NZ * NZ // 2:                    # not worth an indirection per entry
        return NZ * NZ, 0, sig, f"""
{sig[0]} {{ return {P}stm_rhs(t, y, p, arc, 0, dy); }}
{sig[1]} {{ return {P}stm_jump(k, t, y, p, arc, 0); }}
{sig[2]} {{ for (int i = 0; i < {NZ * NZ}; ++i) out[i] = i; }}
"""
    position = {(int(i), int(j)): s for s, (i, j) in enumerate(zip(*np.nonzero(reach)))}
    index, ptr, a, b = [], [0], [], []
    for (i, j), _ in position.items():
        index.append(i * NZ + j)
        for k in np.nonzero(depends[i])[0]:
            if (int(k), j) in position:
                a.append(i * NZ + int(k))
                b.append(position[int(k), j])
        ptr.append(len(a))
    table = lambda name, values: (f"static const int {P}stm_{name}[{max(len(values), 1)}] = "
                                  f"{{{', '.join(map(str, values or [0]))}}};")
    return NPHI, 1, sig, f"""
{table("index", index)}
{table("ptr", ptr)}
{table("a", a)}
{table("b", b)}

{sig[0]} {{
    double A[{NZ * NZ}];
    const double* phi = y + {NZ};
    double* dphi = dy + {NZ};
    int rc = {P}rhs_arc(t, y, p, arc, dy);
    if (rc) return rc;
    rc = {P}jac_arc(t, y, p, arc, A);
    if (rc) return rc;
    for (int s = 0; s < {NPHI}; ++s) {{
        double v = 0.0;
        for (int e = {P}stm_ptr[s]; e < {P}stm_ptr[s + 1]; ++e) v += A[{P}stm_a[e]] * phi[{P}stm_b[e]];
        dphi[s] = v;
    }}
    return 0;
}}

{sig[1]} {{
    double full[{NZ + NZ * NZ}] = {{0.0}};
    for (int i = 0; i < {NZ}; ++i) full[i] = y[i];
    for (int s = 0; s < {NPHI}; ++s) full[{NZ} + {P}stm_index[s]] = y[{NZ} + s];
    int rc = {P}stm_jump(k, t, full, p, arc, 0);
    for (int s = 0; s < {NPHI}; ++s) y[{NZ} + s] = full[{NZ} + {P}stm_index[s]];
    return rc;
}}

{sig[2]} {{ for (int s = 0; s < {NPHI}; ++s) out[s] = {P}stm_index[s]; }}
"""


def _arc_laws(pmp, flags):
    """The control laws on an arc, bang-bang controls fixed by their flags (Min/Max kept).

    Other laws may contain a bang-bang law; it is replaced there too."""
    nsw = len(pmp.switching_functions)
    bang = {sw.control: flag * sw.upper + (1 - flag) * sw.lower
            for sw, flag in zip(pmp.switching_functions, flags[:nsw])}
    on_arc = {pmp.control_law[u]: value for u, value in bang.items()}
    return [bang[u] if u in bang else e.xreplace(on_arc) for u, e in pmp.control_law.items()]


def generate_c(pmp, name=None):
    """Return ``(prefix, header, source)`` C99 text for ``pmp``."""
    ident = c_identifier(name or pmp.problem.name)
    P = f"aoc_{ident}_"
    n = len(pmp.states)
    NZ, NP = 2 * n, len(pmp.params)
    NSW, NU = len(pmp.switching_functions), len(pmp.controls)
    NC = len(pmp.path_constraints)
    NM = NSW + NC
    QMAX = 2                       # the highest constraint order with exact arcs
    bc = pmp.boundary
    NNU, NR = len(bc.nu), len(bc.residuals)
    free_tf = pmp.problem.free_final_time

    t_sym = sp.Symbol("t")
    mapping = {pmp.time: t_sym}
    mapping.update({s: sp.Symbol(f"z[{i}]") for i, s in enumerate(pmp.z)})
    mapping.update({s: sp.Symbol(f"p[{j}]") for j, s in enumerate(pmp.params)})
    bc_map = {s: sp.Symbol(f"p[{j}]") for j, s in enumerate(pmp.params)}
    bc_map.update({s: sp.Symbol(f"z0[{i}]") for i, s in enumerate(bc.x0 + bc.lam0)})
    bc_map.update({s: sp.Symbol(f"zf[{i}]") for i, s in enumerate(bc.xf + bc.lamf)})
    bc_map.update({s: sp.Symbol(f"nu[{j}]") for j, s in enumerate(bc.nu)})
    if free_tf:
        bc_map[bc.tf] = sp.Symbol("tf")

    flags = [sp.Symbol(f"on{k}") for k in range(NM)]
    flag_lines = [line for k in range(NM) for line in
                  (f"    const double on{k} = arc[{k}] ? 1.0 : 0.0;", f"    (void)on{k};")]

    F = [_smooth(e) for e in pmp.dynamics]
    F_arc, L_arc = _arc_dynamics(pmp, flags)
    K_arc = _kinks(_arc_laws(pmp, flags))
    NK = len(K_arc)
    z_c = [mapping[s] for s in pmp.z]
    p_c = [mapping[s] for s in pmp.params]
    S = [_smooth(sw.expr) for sw in pmp.switching_functions]
    G = [_smooth(c.expr) for c in pmp.path_constraints]

    tzp = "double t, const double* z, const double* p"
    functions = []

    f = _Function(f"int {P}rhs({tzp}, double* dz)", ["t", "z", "p"])
    for i, e in enumerate(F):
        f.out(f"dz[{i}]", e)
    functions.append(f)

    # A path constraint's penalty is ``flag * w * g * dg``: its whole tape, Hessian and all, for
    # a flag that is usually zero. So each arc kernel comes in two, chosen by the flags.
    no_penalty = {flag: sp.Integer(0) for flag in flags[NSW:]}
    penalty_flags = any(flag in e.free_symbols for flag in flags[NSW:] for e in F_arc)
    penalty_guard = " || ".join(f"arc[{NSW + c}]" for c in range(NC))
    arc_kernels = []

    def per_arc(name, out, write):
        """``int {P}{name}(t, z, p, arc, out)``, without a penalty's work when its flag is off."""
        def emit(function, off):
            f = _Function(f"int {P}{function}({tzp}, const int* arc, double* {out})",
                          ["t", "z", "p", "arc"])
            f.pre = flag_lines
            write(f, off)
            functions.append(f)
        if not penalty_flags:
            emit(name, {})
            return
        emit(name + "_penalized", {})
        emit(name + "_free", no_penalty)
        sig = f"int {P}{name}({tzp}, const int* arc, double* {out})"
        arc_kernels.append((sig, f"{sig} {{\n    return ({penalty_guard}) ? "
                                 f"{P}{name}_penalized(t, z, p, arc, {out}) : "
                                 f"{P}{name}_free(t, z, p, arc, {out});\n}}\n"))

    def write_rhs(f, off):
        for i, e in enumerate(F_arc):
            f.out(f"dz[{i}]", e.xreplace(off))

    def write_jac(f, off):
        f.zero_fill("A", NZ * NZ)
        for i, e in enumerate(F_arc_c):
            for j, zj in enumerate(z_c):
                f.out(f"A[{i * NZ + j}]", sp.diff(e.xreplace(off), zj))

    def write_dfdp(f, off):
        f.zero_fill("B", NZ * NP)
        for i, e in enumerate(F_arc_c):
            for j, pj in enumerate(p_c):
                f.out(f"B[{i * NP + j}]", sp.diff(e.xreplace(off), pj))

    F_arc_c = [e.xreplace(mapping) for e in F_arc]
    per_arc("rhs_arc", "dz", write_rhs)
    per_arc("jac_arc", "A", write_jac)
    per_arc("dfdp_arc", "B", write_dfdp)

    f = _Function(f"int {P}switching({tzp}, double* S)", ["t", "z", "p", "S"])
    for k, e in enumerate(S):
        f.out(f"S[{k}]", e)
    functions.append(f)

    S_c = [e.xreplace(mapping) for e in S]
    f = _Function(f"int {P}switching_grad({tzp}, double* St, double* Sz, double* Sp)",
                  ["t", "z", "p", "St", "Sz", "Sp"])
    f.zero_fill("St", NSW)
    f.zero_fill("Sz", NSW * NZ)
    f.zero_fill("Sp", NSW * NP)
    for k, e in enumerate(S_c):
        f.out(f"St[{k}]", sp.diff(e, t_sym))
        for j, zj in enumerate(z_c):
            f.out(f"Sz[{k * NZ + j}]", sp.diff(e, zj))
        for j, pj in enumerate(p_c):
            f.out(f"Sp[{k * NP + j}]", sp.diff(e, pj))
    functions.append(f)

    f = _Function(f"int {P}constraint({tzp}, double* g)", ["t", "z", "p", "g"])
    for c, e in enumerate(G):
        f.out(f"g[{c}]", e)
    functions.append(f)

    f = _Function(f"int {P}kinks_arc({tzp}, const int* arc, double* K)",
                  ["t", "z", "p", "arc", "K"])
    f.pre = flag_lines
    for i, e in enumerate(K_arc):
        f.out(f"K[{i}]", e)
    functions.append(f)

    xdot_arc = F_arc[:n]
    Gdot = [_smooth(sp.diff(c.expr, pmp.time)
                    + sum(sp.diff(c.expr, x) * xd for x, xd in zip(pmp.states, xdot_arc)))
            for c in pmp.path_constraints]
    f = _Function(f"int {P}constraint_rate_arc({tzp}, const int* arc, double* gd)",
                  ["t", "z", "p", "arc", "gd"])
    f.pre = flag_lines
    for c, e in enumerate(Gdot):
        f.out(f"gd[{c}]", e)
    functions.append(f)

    f = _Function(f"int {P}modes({tzp}, double* M)", ["t", "z", "p", "M"])
    for k, e in enumerate(S + [-e for e in G]):
        f.out(f"M[{k}]", e)
    functions.append(f)

    f = _Function(f"int {P}control({tzp}, double* u)", ["t", "z", "p", "u"])
    for k, u in enumerate(pmp.controls):
        f.out(f"u[{k}]", _smooth(pmp.control_law[u]))
    functions.append(f)

    bounds = {cs.control: (cs.lower, cs.upper) for cs in pmp.problem.control_sets
              if hasattr(cs, "lower")}
    f = _Function(f"int {P}control_bounds({tzp}, double* lo, double* hi)",
                  ["t", "z", "p", "lo", "hi"])
    for k, u in enumerate(pmp.controls):
        lo, hi = bounds.get(u, (-sp.oo, sp.oo))
        f.out(f"lo[{k}]", sp.sympify(lo))
        f.out(f"hi[{k}]", sp.sympify(hi))
    functions.append(f)

    f = _Function(f"int {P}hamiltonian({tzp}, double* H)", ["t", "z", "p"])
    f.out("H[0]", _smooth(pmp.hamiltonian))
    functions.append(f)

    f = _Function(f"int {P}cost_rate({tzp}, double* L)", ["t", "z", "p"])
    f.out("L[0]", _smooth(pmp.running_cost))
    functions.append(f)

    per_arc("cost_rate_arc", "L", lambda f, off: f.out("L[0]", L_arc.xreplace(off)))

    bc_sig = "const double* z0, const double* zf, const double* nu, double tf, const double* p"
    bc_args = ["z0", "zf", "nu", "tf", "p"]
    R = [_smooth(r) for r in bc.residuals]
    f = _Function(f"int {P}residual({bc_sig}, double* R)", bc_args)
    for i, r in enumerate(R):
        f.out(f"R[{i}]", r)
    functions.append(f)

    R_c = [r.xreplace(bc_map) for r in R]
    z0_c = [bc_map[s] for s in bc.x0 + bc.lam0]
    zf_c = [bc_map[s] for s in bc.xf + bc.lamf]
    nu_c = [bc_map[s] for s in bc.nu]
    f = _Function(f"int {P}residual_jac({bc_sig}, double* Rz0, double* Rzf, double* Rnu, "
                  "double* Rtf, double* Rp)", bc_args + ["Rz0", "Rzf", "Rnu", "Rtf", "Rp"])
    f.zero_fill("Rz0", NR * NZ)
    f.zero_fill("Rzf", NR * NZ)
    f.zero_fill("Rnu", NR * NNU)
    f.zero_fill("Rtf", NR)
    f.zero_fill("Rp", NR * NP)
    for i, r in enumerate(R_c):
        for j, s in enumerate(z0_c):
            f.out(f"Rz0[{i * NZ + j}]", sp.diff(r, s))
        for j, s in enumerate(zf_c):
            f.out(f"Rzf[{i * NZ + j}]", sp.diff(r, s))
        for j, s in enumerate(nu_c):
            f.out(f"Rnu[{i * NNU + j}]", sp.diff(r, s))
        if free_tf:
            f.out(f"Rtf[{i}]", sp.diff(r, sp.Symbol("tf")))
        for j, s in enumerate(p_c):
            f.out(f"Rp[{i * NP + j}]", sp.diff(r, s))
    functions.append(f)

    # Exact boundary arcs: per supported constraint c, the dynamics riding
    # g = 0 and the tangency conditions N = (g, ..., g^(q-1)) with their first
    # and second derivatives (for the costate jump and its Jacobian).
    x_c = z_c[:n]
    H_c = _smooth(pmp.hamiltonian).xreplace(mapping)
    f = _Function(f"int {P}hamiltonian_grad({tzp}, double* Ht, double* Hz)",
                  ["t", "z", "p", "Ht", "Hz"])
    f.zero_fill("Hz", NZ)
    f.out("Ht[0]", sp.diff(H_c, t_sym))
    for j, zj in enumerate(z_c):
        f.out(f"Hz[{j}]", sp.diff(H_c, zj))
    functions.append(f)

    MSET = max([1] + [len(idx) for idx, _ in pmp.boundary_sets])
    exact = []
    # Arc kernels: one per constraint with an exact arc (id c), then one per
    # set of limits ridden at once (id NC + k). Tangency is per constraint.
    riding = [(c, sc.boundary, sc) for c, sc in enumerate(pmp.path_constraints)
              if sc.boundary is not None]
    riding += [(NC + k, arc, None) for k, (_, arc) in enumerate(pmp.boundary_sets)]
    arc_ids = [c for c, _, _ in riding]
    numeric = pmp.riding is not None
    riding_tape, riding_text = "", ""
    if numeric:
        small, riding_tape, riding_text = riding_kernels(pmp, P, _Function, mapping)
        functions += small
    for c, arc, sc in riding:
        tag = f"{P}b{c}_"
        if not numeric:
            Fb = [_arc_smooth(e) for e in arc.dynamics]
            Fb_c = [e.xreplace(mapping) for e in Fb]
            f = _Function(f"int {tag}rhs({tzp}, double* dz)", ["t", "z", "p"])
            for i, e in enumerate(Fb):
                f.out(f"dz[{i}]", e)
            functions.append(f)
            f = _Function(f"int {tag}jac({tzp}, double* A)", ["t", "z", "p", "A"])
            f.zero_fill("A", NZ * NZ)
            for i, e in enumerate(Fb_c):
                for j, zj in enumerate(z_c):
                    f.out(f"A[{i * NZ + j}]", sp.diff(e, zj))
            functions.append(f)
            Hb_c = _arc_smooth(arc.hamiltonian).xreplace(mapping)
            f = _Function(f"int {tag}hamiltonian({tzp}, double* H)", ["t", "z", "p"])
            f.out("H[0]", _arc_smooth(arc.hamiltonian))
            functions.append(f)
            f = _Function(f"int {tag}hamiltonian_grad({tzp}, double* Ht, double* Hz)",
                          ["t", "z", "p", "Ht", "Hz"])
            f.zero_fill("Hz", NZ)
            f.out("Ht[0]", sp.diff(Hb_c, t_sym))
            for j, zj in enumerate(z_c):
                f.out(f"Hz[{j}]", sp.diff(Hb_c, zj))
            functions.append(f)
            f = _Function(f"int {tag}cost_rate({tzp}, double* L)", ["t", "z", "p"])
            f.out("L[0]", _arc_smooth(arc.running_cost))
            functions.append(f)
            f = _Function(f"int {tag}control({tzp}, double* u)", ["t", "z", "p", "u"])
            for k, u in enumerate(pmp.controls):
                f.out(f"u[{k}]", _arc_smooth(arc.control[u]))
            functions.append(f)
            mults = arc.multipliers or (arc.multiplier,)
            f = _Function(f"int {tag}multiplier({tzp}, double* mu)", ["t", "z", "p"])
            for i, m in enumerate(mults):
                f.out(f"mu[{i}]", _arc_smooth(m))
            functions.append(f)
            f = _Function(f"int {tag}multiplier_grad({tzp}, double* Mt, double* Mz)",
                          ["t", "z", "p", "Mt", "Mz"])
            f.zero_fill("Mt", MSET)
            f.zero_fill("Mz", MSET * NZ)
            for i, m in enumerate(mults):
                m_c = _arc_smooth(m).xreplace(mapping)
                f.out(f"Mt[{i}]", sp.diff(m_c, t_sym))
                for j, zj in enumerate(z_c):
                    f.out(f"Mz[{i * NZ + j}]", sp.diff(m_c, zj))
            functions.append(f)
        if sc is None:
            continue
        exact.append(c)
        N_c = [_smooth(e).xreplace(mapping) for e in sc.tangency]
        f = _Function(f"int {tag}tangency({tzp}, double* N)", ["t", "z", "p", "N"])
        f.zero_fill("N", QMAX)
        for i, e in enumerate(N_c):
            f.out(f"N[{i}]", e)
        functions.append(f)
        f = _Function(f"int {tag}tangency_grad({tzp}, double* Nt, double* Nx)",
                      ["t", "z", "p", "Nt", "Nx"])
        f.zero_fill("Nt", QMAX)
        f.zero_fill("Nx", QMAX * n)
        for i, e in enumerate(N_c):
            f.out(f"Nt[{i}]", sp.diff(e, t_sym))
            for j, xj in enumerate(x_c):
                f.out(f"Nx[{i * n + j}]", sp.diff(e, xj))
        functions.append(f)
        f = _Function(f"int {tag}tangency_hess({tzp}, double* Nxx, double* Nxt)",
                      ["t", "z", "p", "Nxx", "Nxt"])
        f.zero_fill("Nxx", QMAX * n * n)
        f.zero_fill("Nxt", QMAX * n)
        for i, e in enumerate(N_c):
            for j, xj in enumerate(x_c):
                d = sp.diff(e, xj)
                f.out(f"Nxt[{i * n + j}]", sp.diff(d, t_sym))
                for k2, xk in enumerate(x_c):
                    f.out(f"Nxx[{(i * n + j) * n + k2}]", sp.diff(d, xk))
        functions.append(f)

    # Singular arcs: per bang-bang control k with a derived SingularArc, the
    # dynamics at the singular control and the junction conditions (S, S').
    singular = []
    for k_sw, arc in enumerate(pmp.singular_arcs):
        if arc is None:
            continue
        singular.append(k_sw)
        Fs = [_smooth(e) for e in arc.dynamics]
        Fs_c = [e.xreplace(mapping) for e in Fs]
        tag = f"{P}s{k_sw}_"
        f = _Function(f"int {tag}rhs({tzp}, double* dz)", ["t", "z", "p"])
        for i, e in enumerate(Fs):
            f.out(f"dz[{i}]", e)
        functions.append(f)
        f = _Function(f"int {tag}jac({tzp}, double* A)", ["t", "z", "p", "A"])
        f.zero_fill("A", NZ * NZ)
        for i, e in enumerate(Fs_c):
            for j, zj in enumerate(z_c):
                f.out(f"A[{i * NZ + j}]", sp.diff(e, zj))
        functions.append(f)
        f = _Function(f"int {tag}hamiltonian({tzp}, double* H)", ["t", "z", "p"])
        f.out("H[0]", _smooth(arc.hamiltonian))
        functions.append(f)
        f = _Function(f"int {tag}cost_rate({tzp}, double* L)", ["t", "z", "p"])
        f.out("L[0]", _smooth(arc.running_cost))
        functions.append(f)
        f = _Function(f"int {tag}control({tzp}, double* u)", ["t", "z", "p", "u"])
        for kk, u in enumerate(pmp.controls):
            f.out(f"u[{kk}]", _smooth(arc.law[u]))
        functions.append(f)
        f = _Function(f"int {tag}legendre({tzp}, double* LC)", ["t", "z", "p"])
        f.out("LC[0]", _smooth(arc.legendre_clebsch))
        functions.append(f)
        J_c = [_smooth(e).xreplace(mapping) for e in arc.junction]
        f = _Function(f"int {tag}junction({tzp}, double* N)", ["t", "z", "p", "N"])
        for i, e in enumerate(J_c):
            f.out(f"N[{i}]", e)
        functions.append(f)
        f = _Function(f"int {tag}junction_grad({tzp}, double* Nt, double* Nz)",
                      ["t", "z", "p", "Nt", "Nz"])
        f.zero_fill("Nt", 2)
        f.zero_fill("Nz", 2 * NZ)
        for i, e in enumerate(J_c):
            f.out(f"Nt[{i}]", sp.diff(e, t_sym))
            for j, zj in enumerate(z_c):
                f.out(f"Nz[{i * NZ + j}]", sp.diff(e, zj))
        functions.append(f)

    def dispatchers(group, letter, ids, specs):
        out, sigs = [], []
        for name, outs, out_args in specs:
            sig = f"int {P}{group}_{name}(int c, {tzp}, {outs})"
            sigs.append(sig)
            if numeric and group == "boundary" and not name.startswith("tangency"):
                out.append(f"{sig} {{\n    return {P}riding_{name}(c, t, z, p, {out_args});\n}}\n")
                continue
            cases = "".join(f"    case {c}: return {P}{letter}{c}_{name}(t, z, p, {out_args});\n"
                            for c in ids)
            voids = "".join(f" (void){a.strip()};" for a in out_args.split(","))
            out.append(f"{sig} {{\n    (void)t; (void)z; (void)p;{voids}\n"
                       f"    switch (c) {{\n{cases}    default: return 3;\n    }}\n}}\n")
        stm_sig = (f"int {P}{group}_stm_rhs(int c, double t, const double* y, const double* p, "
                   "double* dy)")
        sigs.append(stm_sig)
        out.append(f"""{stm_sig} {{
    double A[{NZ * NZ}];
    const double* Phi = y + {NZ};
    double* dPhi = dy + {NZ};
    int rc = {P}{group}_rhs(c, t, y, p, dy);
    if (rc) return rc;
    rc = {P}{group}_jac(c, t, y, p, A);
    if (rc) return rc;
    for (int i = 0; i < {NZ}; ++i) {{  /* A is mostly zeros on the free arcs of a large problem */
        double* row = dPhi + i * {NZ};
        for (int j = 0; j < {NZ}; ++j) row[j] = 0.0;
        for (int k = 0; k < {NZ}; ++k) {{
            const double a = A[i * {NZ} + k];
            if (a == 0.0) continue;
            const double* from = Phi + k * {NZ};
            for (int j = 0; j < {NZ}; ++j) row[j] += a * from[j];
        }}
    }}
    return 0;
}}
""")
        return out, sigs

    common_specs = [
        ("rhs", "double* dz", "dz"), ("jac", "double* A", "A"), ("hamiltonian", "double* H", "H"),
        ("cost_rate", "double* L", "L"), ("control", "double* u", "u"),
    ]
    dispatch, dispatch_sigs = dispatchers("boundary", "b", arc_ids, common_specs + [
        ("hamiltonian_grad", "double* Ht, double* Hz", "Ht, Hz"),
        ("multiplier", "double* mu", "mu"),
        ("multiplier_grad", "double* Mt, double* Mz", "Mt, Mz"),
    ])
    tang, tang_sigs = dispatchers("boundary", "b", exact, [
        ("tangency", "double* N", "N"),
        ("tangency_grad", "double* Nt, double* Nx", "Nt, Nx"),
        ("tangency_hess", "double* Nxx, double* Nxt", "Nxx, Nxt"),
    ])
    # The tangency group needs no STM right-hand side of its own.
    dispatch += tang[:-1]
    dispatch_sigs += tang_sigs[:-1]
    more, more_sigs = dispatchers("singular", "s", singular, common_specs + [
        ("legendre", "double* LC", "LC"), ("junction", "double* N", "N"),
        ("junction_grad", "double* Nt, double* Nz", "Nt, Nz"),
    ])
    dispatch += more
    dispatch_sigs += more_sigs

    body = "\n\n".join(fn.render(mapping if not fn.signature.startswith(f"int {P}residual")
                                 else bc_map) for fn in functions)
    body = "\n\n".join([riding_tape, body, riding_text]) if numeric else body

    NSET = len(pmp.boundary_sets)
    NPHI, SPARSE_STM, sparse_sigs, sparse_stm = _sparse_stm(
        P, NZ, *_stm_pattern([e.xreplace(no_penalty) for e in F_arc_c], S_c, flags, z_c))
    PENALTY_FLAGS = int(penalty_flags)
    macros = dict(NZ=NZ, NP=NP, NSW=NSW, NC=NC, NM=NM, NK=NK, NU=NU, NNU=NNU, NR=NR, QMAX=QMAX,
                  NSET=NSET, MSET=MSET, FREE_TF=int(free_tf), NPHI=NPHI, SPARSE_STM=SPARSE_STM,
                  PENALTY_FLAGS=PENALTY_FLAGS)
    guard = f"AOC_{ident.upper()}_H"
    header_lines = [
        f"/* Generated by auto-optimal-control for problem {pmp.problem.name!r}. Do not edit. */",
        f"#ifndef {guard}", f"#define {guard}", "",
    ]
    header_lines += [f"#define {P.upper()}{k} {v}" for k, v in macros.items()]
    header_lines.append("")
    for fn in functions:
        header_lines.append(fn.signature + ";")
    stm_sigs = [
        f"int {P}stm_rhs(double t, const double* y, const double* p, const int* arc, int with_p, double* dy)",
        f"int {P}stm_jump(int k, double t, double* y, const double* p, const int* arc, int with_p)",
    ]
    header_lines += [s + ";" for s in (*stm_sigs, *sparse_sigs)]
    header_lines += [s + ";" for s in dispatch_sigs]
    header_lines += [sig + ";" for sig, _ in arc_kernels]
    header_lines += ["", f"#endif /* {guard} */", ""]
    header = "\n".join(header_lines)

    NP1, NSW1, NM1 = max(NP, 1), max(NSW, 1), max(NM, 1)
    stm = f"""
int {P}stm_rhs(double t, const double* y, const double* p, const int* arc, int with_p, double* dy) {{
    double A[{NZ * NZ}];
    double B[{NZ * NP1}];
    const double* Phi = y + {NZ};
    double* dPhi = dy + {NZ};
    int rc = {P}rhs_arc(t, y, p, arc, dy);
    if (rc) return rc;
    rc = {P}jac_arc(t, y, p, arc, A);
    if (rc) return rc;
    for (int i = 0; i < {NZ}; ++i) {{  /* A is mostly zeros on the free arcs of a large problem */
        double* row = dPhi + i * {NZ};
        for (int j = 0; j < {NZ}; ++j) row[j] = 0.0;
        for (int k = 0; k < {NZ}; ++k) {{
            const double a = A[i * {NZ} + k];
            if (a == 0.0) continue;
            const double* from = Phi + k * {NZ};
            for (int j = 0; j < {NZ}; ++j) row[j] += a * from[j];
        }}
    }}
    if (with_p && {NP} > 0) {{
        const double* Psi = Phi + {NZ * NZ};
        double* dPsi = dPhi + {NZ * NZ};
        rc = {P}dfdp_arc(t, y, p, arc, B);
        if (rc) return rc;
        for (int i = 0; i < {NZ}; ++i) {{
            for (int j = 0; j < {NP}; ++j) {{
                double s = B[i * {NP1} + j];
                for (int k = 0; k < {NZ}; ++k) s += A[i * {NZ} + k] * Psi[k * {NP1} + j];
                dPsi[i * {NP1} + j] = s;
            }}
        }}
    }}
    return 0;
}}

int {P}stm_jump(int k, double t, double* y, const double* p, const int* arc, int with_p) {{
    int arc_after[{NM1}];
    double Fm[{NZ}], Fp[{NZ}], dF[{NZ}];
    double St[{NSW1}], Sz[{NSW1 * NZ}], Sp[{NSW1 * NP1}];
    double* Phi = y + {NZ};
    double Sdot;
    int rc;
    if (k >= {NSW} && k < {NM}) return 0;  /* path constraint: F is continuous */
    if (k < 0 || k >= {NSW}) return 1;
    for (int i = 0; i < {NM}; ++i) arc_after[i] = arc[i];
    arc_after[k] = !arc[k];
    rc = {P}rhs_arc(t, y, p, arc, Fm);
    if (rc) return rc;
    rc = {P}rhs_arc(t, y, p, arc_after, Fp);
    if (rc) return rc;
    rc = {P}switching_grad(t, y, p, St, Sz, Sp);
    if (rc) return rc;
    const double* g = Sz + k * {NZ};
    Sdot = St[k];
    for (int i = 0; i < {NZ}; ++i) {{
        Sdot += g[i] * Fm[i];
        dF[i] = Fp[i] - Fm[i];
    }}
    if (Sdot == 0.0) return 2;            /* tangential switch: no jump defined */
    for (int j = 0; j < {NZ}; ++j) {{
        double c = 0.0;
        for (int i = 0; i < {NZ}; ++i) c += g[i] * Phi[i * {NZ} + j];
        c /= Sdot;
        for (int i = 0; i < {NZ}; ++i) Phi[i * {NZ} + j] += dF[i] * c;
    }}
    if (with_p && {NP} > 0) {{
        double* Psi = Phi + {NZ * NZ};
        for (int j = 0; j < {NP}; ++j) {{
            double c = Sp[k * {NP1} + j];
            for (int i = 0; i < {NZ}; ++i) c += g[i] * Psi[i * {NP1} + j];
            c /= Sdot;
            for (int i = 0; i < {NZ}; ++i) Psi[i * {NP1} + j] += dF[i] * c;
        }}
    }}
    return 0;
}}
"""
    if NSW == 0:
        # No switches: keep the symbol so callers link, but nothing can jump.
        stm = stm.split(f"int {P}stm_jump")[0] + (
            f"int {P}stm_jump(int k, double t, double* y, const double* p, const int* arc, int with_p) {{\n"
            "    (void)t; (void)y; (void)p; (void)arc; (void)with_p;\n"
            f"    return (k >= 0 && k < {NM}) ? 0 : 1;\n}}\n")
    source = "\n".join([
        f"/* Generated by auto-optimal-control for problem {pmp.problem.name!r}. Do not edit. */",
        "#include <math.h>",
        f'#include "{{header}}"',
        "",
        body,
        stm,
        sparse_stm,
        "\n".join(dispatch),
        "\n".join(text for _, text in arc_kernels),
    ])
    return P, header, source, macros
