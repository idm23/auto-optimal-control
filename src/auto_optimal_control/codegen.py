"""Print a :class:`~auto_optimal_control.pontryagin.PontryaginSystem` as plain C99.

:func:`generate_c` returns a header and a source file that depend only on
``<math.h>``. Every function returns ``0`` on success. Arrays are row-major;
``z = [x, lambda]`` (``NZ = 2 n`` entries), ``p`` is the parameter vector.

Functions, for a prefix ``P`` (``aoc_<name>_``)::

    int P_rhs(t, z, p, dz)                   z' = F(t, z, p), bang-bang controls switch inside
    int P_rhs_arc(t, z, p, arc, dz)          F on a fixed arc: arc[k] = 1 puts bang-bang
                                             control k at its upper bound, 0 at its lower one
    int P_jac_arc(t, z, p, arc, A)           A = dF/dz        (NZ x NZ)
    int P_dfdp_arc(t, z, p, arc, B)          B = dF/dp        (NZ x NP)
    int P_stm_rhs(t, y, p, arc, with_p, dy)  y = [z, Phi (NZ x NZ), Psi (NZ x NP) if with_p]:
                                             Phi' = A Phi, Psi' = A Psi + B
    int P_stm_jump(k, t, y, p, arc, with_p)  apply the jump of Phi (and Psi) when switching
                                             function k changes sign; arc is the arc before
    int P_switching(t, z, p, S)              switching functions (NSW)
    int P_switching_grad(t, z, p, St, Sz, Sp)  dS/dt (NSW), dS/dz (NSW x NZ), dS/dp (NSW x NP)
    int P_control(t, z, p, u)                optimal control u* (NU)
    int P_hamiltonian(t, z, p, H)            H at u* (one value)
    int P_cost_rate(t, z, p, L)              running cost at u* (one value)
    int P_cost_rate_arc(t, z, p, arc, L)     running cost on a fixed arc
    int P_residual(z0, zf, nu, tf, p, R)     boundary residuals (NR); tf is ignored when the
                                             final time is fixed
    int P_residual_jac(z0, zf, nu, tf, p, Rz0, Rzf, Rnu, Rtf, Rp)
                                             dR/dz0, dR/dzf (NR x NZ), dR/dnu (NR x NNU),
                                             dR/dtf (NR), dR/dp (NR x NP)

The jump uses the saltation matrix. Let switching function ``S`` cross zero
at ``t_s``, with ``F-`` and ``F+`` the dynamics on the arcs before and after
it and ``Sdot = dS/dt + dS/dz . F-``. Then
``Phi+ = Phi- + (F+ - F-) (dS/dz Phi-) / Sdot`` and
``Psi+ = Psi- + (F+ - F-) (dS/dz Psi- + dS/dp) / Sdot``.
"""

import re

import sympy as sp
from sympy.printing.c import C99CodePrinter


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


def _smooth(expr):
    """Min/Max (saturated controls) as Piecewise, so they differentiate and print."""
    expr = sp.sympify(expr)
    if expr.has(sp.Min, sp.Max):
        expr = expr.rewrite(sp.Piecewise)
    return expr


class _Function:
    """One generated C function: outputs computed through common subexpressions."""

    def __init__(self, signature, args):
        self.signature = signature
        self.args = args
        self.pre = []
        self.zero = []
        self.outputs = []

    def zero_fill(self, array, size):
        if size:
            self.zero.append(f"    for (int i_ = 0; i_ < {size}; ++i_) {array}[i_] = 0.0;")

    def out(self, lhs, expr):
        self.outputs.append((lhs, expr))

    def render(self, mapping):
        lines = [self.signature + " {"]
        lines += [f"    (void){a};" for a in self.args]
        lines += self.pre
        lines += self.zero
        keep = [(lhs, e.xreplace(mapping)) for lhs, e in self.outputs]
        keep = [(lhs, e) for lhs, e in keep if not (self.zero and e == 0)]
        if keep:
            repl, reduced = sp.cse([e for _, e in keep], symbols=sp.numbered_symbols("cse"))
            for s, e in repl:
                lines.append(f"    const double {s} = {_c(e)};")
            for (lhs, _), e in zip(keep, reduced):
                lines.append(f"    {lhs} = {_c(e)};")
        lines.append("    return 0;")
        lines.append("}")
        return "\n".join(lines)


def _arc_dynamics(pmp, flags):
    """Dynamics and running cost with each bang-bang control fixed by a 0/1 flag.

    On an arc, control k equals ``flag_k * upper + (1 - flag_k) * lower``, a
    smooth expression, so derivatives with respect to parameters in the bounds
    come out right.
    """
    bang = {sw.control: flag * sw.upper + (1 - flag) * sw.lower
            for sw, flag in zip(pmp.switching_functions, flags)}
    # Other laws may contain a bang-bang law; replace it there too.
    on_arc = {pmp.control_law[u]: value for u, value in bang.items()}
    law = {u: bang[u] if u in bang else _smooth(e.xreplace(on_arc))
           for u, e in pmp.control_law.items()}
    H = pmp.hamiltonian_full
    xdot = [f.subs(law) for f in pmp.problem.dynamics]
    lamdot = [-sp.diff(H, x).subs(law) for x in pmp.states]
    cost = _smooth(pmp.problem.running_cost.subs(law))
    return [_smooth(e) for e in xdot + lamdot], cost


def generate_c(pmp, name=None):
    """Return ``(prefix, header, source)`` C99 text for ``pmp``."""
    ident = c_identifier(name or pmp.problem.name)
    P = f"aoc_{ident}_"
    n = len(pmp.states)
    NZ, NP = 2 * n, len(pmp.params)
    NSW, NU = len(pmp.switching_functions), len(pmp.controls)
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

    flags = [sp.Symbol(f"on{k}") for k in range(NSW)]
    flag_lines = [line for k in range(NSW) for line in
                  (f"    const double on{k} = arc[{k}] ? 1.0 : 0.0;", f"    (void)on{k};")]

    F = [_smooth(e) for e in pmp.dynamics]
    F_arc, L_arc = _arc_dynamics(pmp, flags)
    z_c = [mapping[s] for s in pmp.z]
    p_c = [mapping[s] for s in pmp.params]
    S = [_smooth(sw.expr) for sw in pmp.switching_functions]

    tzp = "double t, const double* z, const double* p"
    functions = []

    f = _Function(f"int {P}rhs({tzp}, double* dz)", ["t", "z", "p"])
    for i, e in enumerate(F):
        f.out(f"dz[{i}]", e)
    functions.append(f)

    f = _Function(f"int {P}rhs_arc({tzp}, const int* arc, double* dz)", ["t", "z", "p", "arc"])
    f.pre = flag_lines
    for i, e in enumerate(F_arc):
        f.out(f"dz[{i}]", e)
    functions.append(f)

    F_arc_c = [e.xreplace(mapping) for e in F_arc]
    f = _Function(f"int {P}jac_arc({tzp}, const int* arc, double* A)", ["t", "z", "p", "arc"])
    f.pre = flag_lines
    f.zero_fill("A", NZ * NZ)
    for i, e in enumerate(F_arc_c):
        for j, zj in enumerate(z_c):
            f.out(f"A[{i * NZ + j}]", sp.diff(e, zj))
    functions.append(f)

    f = _Function(f"int {P}dfdp_arc({tzp}, const int* arc, double* B)", ["t", "z", "p", "arc"])
    f.pre = flag_lines
    f.zero_fill("B", NZ * NP)
    for i, e in enumerate(F_arc_c):
        for j, pj in enumerate(p_c):
            f.out(f"B[{i * NP + j}]", sp.diff(e, pj))
    functions.append(f)

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

    f = _Function(f"int {P}control({tzp}, double* u)", ["t", "z", "p", "u"])
    for k, u in enumerate(pmp.controls):
        f.out(f"u[{k}]", _smooth(pmp.control_law[u]))
    functions.append(f)

    f = _Function(f"int {P}hamiltonian({tzp}, double* H)", ["t", "z", "p"])
    f.out("H[0]", _smooth(pmp.hamiltonian))
    functions.append(f)

    f = _Function(f"int {P}cost_rate({tzp}, double* L)", ["t", "z", "p"])
    f.out("L[0]", _smooth(pmp.running_cost))
    functions.append(f)

    f = _Function(f"int {P}cost_rate_arc({tzp}, const int* arc, double* L)", ["t", "z", "p", "arc"])
    f.pre = flag_lines
    f.out("L[0]", L_arc)
    functions.append(f)

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

    body = "\n\n".join(fn.render(mapping if not fn.signature.startswith(f"int {P}residual")
                                 else bc_map) for fn in functions)

    macros = dict(NZ=NZ, NP=NP, NSW=NSW, NU=NU, NNU=NNU, NR=NR, FREE_TF=int(free_tf))
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
    header_lines += [s + ";" for s in stm_sigs]
    header_lines += ["", f"#endif /* {guard} */", ""]
    header = "\n".join(header_lines)

    NP1, NSW1 = max(NP, 1), max(NSW, 1)
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
    for (int i = 0; i < {NZ}; ++i) {{
        for (int j = 0; j < {NZ}; ++j) {{
            double s = 0.0;
            for (int k = 0; k < {NZ}; ++k) s += A[i * {NZ} + k] * Phi[k * {NZ} + j];
            dPhi[i * {NZ} + j] = s;
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
    int arc_after[{NSW1}];
    double Fm[{NZ}], Fp[{NZ}], dF[{NZ}];
    double St[{NSW1}], Sz[{NSW1 * NZ}], Sp[{NSW1 * NP1}];
    double* Phi = y + {NZ};
    double Sdot;
    int rc;
    if (k < 0 || k >= {NSW}) return 1;
    for (int i = 0; i < {NSW}; ++i) arc_after[i] = arc[i];
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
            "    (void)k; (void)t; (void)y; (void)p; (void)arc; (void)with_p;\n"
            "    return 1;\n}\n")
    source = "\n".join([
        f"/* Generated by auto-optimal-control for problem {pmp.problem.name!r}. Do not edit. */",
        "#include <math.h>",
        f'#include "{{header}}"',
        "",
        body,
        stm,
    ])
    return P, header, source, macros
