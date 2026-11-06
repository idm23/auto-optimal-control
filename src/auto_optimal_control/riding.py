"""One kernel for every set of order-1 limits ridden at once, solved at run time.

Generating an arc's kernels symbolically needs its controls and multipliers in
closed form, once per set of limits. A six-joint arm has 728 sets, each with
the inverse of a matrix of mass-matrix entries in it. But the equations are
linear in the controls and multipliers, so C can solve them where they are
needed. For limits ``g_i^(1) = A_i(x) u + b_i(x)`` and a Hamiltonian
quadratic in ``u`` with ``d2H/du2 = R``::

    [ R  A^T ] [ u  ]     [ -dH/du at u = 0 ]
    [ A  0   ] [ mu ]  =  [ -b              ]

and the arc's Jacobian differentiates that system: with ``y = (u, mu)`` and
``r(z, y) = 0`` the equations above, ``dy/dz = -K^-1 dr/dz``.

Bounds on the controls make this a small quadratic program: strictly convex, so
its solution is unique and a function of ``z``. It is found by holding controls at
their bounds and solving the same system with those rows added, until the controls are
in bounds and the bounds' multipliers have the right signs. A held control does not
move with ``z``, so the rule for ``dy/dz`` is unchanged.

Everything about the limits comes from :class:`~auto_optimal_control.tape.Tape`
steps of their state parts, the *roots* (``tau`` for both ``tau - tau_max`` and
``-tau - tau_max``): the rate along the dynamics in one forward pass, its
gradient in one reverse pass, and the second derivatives of
``sum_k mu_k h_k^(1)``, one scalar however many limits are active.
"""

import sympy as sp


def _names(pmp, mapping, extra=()):
    names = {s.name: str(c) for s, c in mapping.items()}
    names.update(extra)
    return names


def _tape_function(tape, name, args, outputs, names, zero=()):
    """A C function of ``outputs``, pairs of a left-hand side and a tape step."""
    helpers, body, operand = tape.c_block([s for _, s in outputs], names, f"h_{name}")
    lines = [f"static void {name}({args}) {{",
             "    (void)t; (void)z; (void)p; (void)u; (void)mu;"]
    lines += [f"    for (int i_ = 0; i_ < {size}; ++i_) {array}[i_] = 0.0;" for array, size in zero]
    lines += body
    lines += [f"    {lhs} = {operand(s)};" for lhs, s in outputs
              if not (zero and s == tape.zero)]
    lines.append("}")
    return "\n".join(helpers + ["\n".join(lines)])


def riding_kernels(pmp, P, make_function, mapping):
    """The C of ``P_riding_*`` and the small symbolic functions it calls.

    ``make_function`` is :class:`~auto_optimal_control.codegen._Function`.
    Returns ``(functions, text)``: ``functions`` to render with the others,
    ``text`` to place after them.
    """
    system = pmp.riding
    tape = system.tape
    x, lam, u = pmp.states, pmp.costates, pmp.controls
    n, m, nr, nc = len(x), len(u), len(system.roots), len(pmp.path_constraints)
    nz = 2 * n
    problem = pmp.problem
    f = list(problem.dynamics)
    H = problem.running_cost + sp.Add(*[l * e for l, e in zip(lam, f)])

    # -- the small symbolic parts: only the problem's own H and f ---------------
    u_c = {s: sp.Symbol(f"u[{k}]") for k, s in enumerate(u)}
    mapped = {**mapping, **u_c}
    tzpu = "double t, const double* z, const double* p, const double* u"
    functions = []

    def small(name, args, outs, zero=()):
        fn = make_function(f"int {P}r_{name}({args})", ["t", "z", "p", *name_args(args)],
                           mapping=mapped)
        for array, size in zero:
            fn.zero_fill(array, size)
        for lhs, e in outs:
            fn.out(lhs, e)
        functions.append(fn)

    def name_args(args):
        return [a.split("*")[-1].split()[-1] for a in args.split(",")][3:]

    at0 = {s: 0 for s in u}
    small("quadratic", "double t, const double* z, const double* p, double* R, double* c",
          [(f"R[{i * m + j}]", sp.diff(H, a, b)) for i, a in enumerate(u) for j, b in enumerate(u)]
          + [(f"c[{i}]", sp.diff(H, a).subs(at0)) for i, a in enumerate(u)])
    small("flow", f"{tzpu}, double* dx, double* Hx",
          [(f"dx[{i}]", e) for i, e in enumerate(f)]
          + [(f"Hx[{i}]", sp.diff(H, s)) for i, s in enumerate(x)])
    small("value", f"{tzpu}, double* H, double* L",
          [("H[0]", H), ("L[0]", problem.running_cost)])
    small("derivatives", f"{tzpu}, double* Fx, double* Fu, double* Hxx, double* Hxu",
          [(f"Fx[{i * n + j}]", sp.diff(e, s)) for i, e in enumerate(f) for j, s in enumerate(x)]
          + [(f"Fu[{i * m + j}]", sp.diff(e, a)) for i, e in enumerate(f) for j, a in enumerate(u)]
          + [(f"Hxx[{i * n + j}]", sp.diff(H, a, b))
             for i, a in enumerate(x) for j, b in enumerate(x)]
          + [(f"Hxu[{i * m + j}]", sp.diff(H, a, b))
             for i, a in enumerate(x) for j, b in enumerate(u)],
          zero=[("Fx", n * n), ("Fu", n * m), ("Hxx", n * n), ("Hxu", n * m)])

    # -- the tape parts ----------------------------------------------------------
    names = _names(pmp, mapping, {s.name: f"u[{k}]" for k, s in enumerate(u)})
    names.update({f"mu_root_{k}": f"mu[{k}]" for k in range(nr)})
    xn = [s.name for s in x]
    un = [s.name for s in u]
    seeds = {s.name: tape.from_sympy(e) for s, e in zip(x, f)}
    seeds0 = {s.name: tape.from_sympy(e.subs(at0)) for s, e in zip(x, f)}
    rates = [tape.directional(r, seeds) for r in system.roots]
    b = [tape.directional(r, seeds0) for r in system.roots]
    A = []
    for a in u:
        slope = {s.name: tape.from_sympy(sp.diff(e, a)) for s, e in zip(x, f)}
        A.append([tape.directional(r, slope) for r in system.roots])

    # sum_k mu_k h_k^(1): one scalar, so its gradient is one reverse pass.
    total = tape.zero
    for k, rate in enumerate(rates):
        total = tape.add(total, tape.mul(tape.input(f"mu_root_{k}"), rate))
    Lx = tape.gradient(total, xn)
    Lxx = {}
    for i in range(n):
        for j in range(i, n):
            Lxx[i, j] = tape.diff(Lx[i], xn[j])
    Lxu = [[tape.diff(Lx[i], a) for a in un] for i in range(n)]
    Gx = [tape.gradient(rate, xn) for rate in rates]

    args = ("double t, const double* z, const double* p, const double* u, const double* mu, "
            "double* restrict o")
    parts = [
        _tape_function(tape, f"{P}r_limits", args,
                       [(f"o[{k * m + j}]", A[j][k]) for k in range(nr) for j in range(m)]
                       + [(f"o[{nr * m + k}]", b[k]) for k in range(nr)],
                       names, zero=[("o", nr * m + nr)]),
        _tape_function(tape, f"{P}r_lagrangian_x", args,
                       [(f"o[{i}]", Lx[i]) for i in range(n)], names, zero=[("o", n)]),
        _tape_function(tape, f"{P}r_rate_grad", args,
                       [(f"o[{k * n + i}]", Gx[k][i]) for k in range(nr) for i in range(n)],
                       names, zero=[("o", nr * n)]),
        _tape_function(tape, f"{P}r_lagrangian_hess", args,
                       [(f"o[{i * n + j}]", s) for (i, j), s in Lxx.items()]
                       + [(f"o[{j * n + i}]", s) for (i, j), s in Lxx.items() if i != j]
                       + [(f"o[{n * n + i * m + j}]", Lxu[i][j]) for i in range(n)
                          for j in range(m)],
                       names, zero=[("o", n * n + n * m)]),
    ]

    # -- the sets ---------------------------------------------------------------
    sets = [(c,) for c in range(nc)] + [idx for idx, _ in pmp.boundary_sets]
    mset = max(len(s) for s in sets)
    bounded = bool(problem.control_sets)
    table = ",\n    ".join("{" + ", ".join(str(i) for i in (*s, *([-1] * (mset - len(s))))) + "}"
                           for s in sets)
    sizes = ", ".join(str(len(s)) for s in sets)
    roots = ", ".join(str(r) for r in system.root_of)
    signs = ", ".join(str(float(s)) for s in system.sign_of)
    dim = m + mset + (m if bounded else 0)

    text = f"""
static const signed char {P}r_set[{len(sets)}][{mset}] = {{
    {table}
}};
static const signed char {P}r_size[{len(sets)}] = {{{sizes}}};
static const signed char {P}r_root[{nc}] = {{{roots}}};
static const double {P}r_sign[{nc}] = {{{signs}}};

/* Dense LU with partial pivoting; K has d rows of stride {dim}. */
static int {P}r_lu(double* K, int d, int* piv) {{
    double largest = 0.0;
    for (int i = 0; i < d * {dim}; ++i) if (fabs(K[i]) > largest) largest = fabs(K[i]);
    for (int c = 0; c < d; ++c) {{
        int best = c;
        for (int r = c + 1; r < d; ++r)
            if (fabs(K[r * {dim} + c]) > fabs(K[best * {dim} + c])) best = r;
        if (!(fabs(K[best * {dim} + c]) > 1e-13 * largest)) return 4;  /* limits not independent */
        piv[c] = best;
        if (best != c)
            for (int j = c; j < d; ++j) {{  /* the multipliers stay: the solve swaps as it goes */
                double s = K[c * {dim} + j];
                K[c * {dim} + j] = K[best * {dim} + j];
                K[best * {dim} + j] = s;
            }}
        for (int r = c + 1; r < d; ++r) {{
            double w = K[r * {dim} + c] / K[c * {dim} + c];
            K[r * {dim} + c] = w;
            if (w != 0.0) for (int j = c + 1; j < d; ++j) K[r * {dim} + j] -= w * K[c * {dim} + j];
        }}
    }}
    return 0;
}}

static void {P}r_lu_solve(const double* K, int d, const int* piv, double* v) {{
    for (int c = 0; c < d; ++c) {{
        double s = v[piv[c]];
        v[piv[c]] = v[c];
        v[c] = s;
        for (int r = c + 1; r < d; ++r) v[r] -= K[r * {dim} + c] * v[c];
    }}
    for (int c = d - 1; c >= 0; --c) {{
        for (int j = c + 1; j < d; ++j) v[c] -= K[c * {dim} + j] * v[j];
        v[c] /= K[c * {dim} + c];
    }}
}}

/* One system: the limits of the set, and the controls held at bounds (held[i] = -1, +1). */
static int {P}r_system(int id, const double* R, const double* limits, const double* c0,
                       const double* lo, const double* hi, const int* held, double* K, int* piv,
                       double* v, int* dout) {{
    const int s = {P}r_size[id];
    int d = {m} + s;
    for (int i = 0; i < {m}; ++i) if (held[i]) ++d;
    if (d > {dim}) return 4;
    for (int i = 0; i < d * {dim}; ++i) K[i] = 0.0;
    for (int i = 0; i < {m}; ++i) {{
        for (int j = 0; j < {m}; ++j) K[i * {dim} + j] = R[i * {m} + j];
        v[i] = -c0[i];
    }}
    for (int i = 0; i < s; ++i) {{
        const int c = {P}r_set[id][i];
        const int k = {P}r_root[c];
        const double sign = {P}r_sign[c];
        for (int j = 0; j < {m}; ++j) {{
            K[({m} + i) * {dim} + j] = sign * limits[k * {m} + j];
            K[j * {dim} + {m} + i] = sign * limits[k * {m} + j];
        }}
        v[{m} + i] = -sign * limits[{nr * m} + k];
    }}
    int row = {m} + s;
    for (int i = 0; i < {m}; ++i) {{
        if (!held[i]) continue;
        K[row * {dim} + i] = 1.0;
        K[i * {dim} + row] = 1.0;
        v[row] = held[i] > 0 ? hi[i] : lo[i];
        ++row;
    }}
    *dout = d;
    int rc = {P}r_lu(K, d, piv);
    if (rc) return rc;
    {P}r_lu_solve(K, d, piv, v);
    return 0;
}}

/* Whether the solution of a system is the one: controls within bounds, and multipliers
   of the bounds with the right signs. If not, which and to say what to change next. */
static int {P}r_check(int id, const double* v, const double* lo, const double* hi,
                      const int* held, int* which, int* to) {{
    const int s = {P}r_size[id];
    double worst = 0.0;
    *which = -1;
    int row = {m} + s;
    for (int i = 0; i < {m}; ++i) {{
        const double margin = 1e-13 * (1.0 + fabs(v[i]));
        if (!held[i]) {{
            double over = v[i] - hi[i], under = lo[i] - v[i];
            if (over > margin && over > worst) {{ worst = over; *which = i; *to = 1; }}
            if (under > margin && under > worst) {{ worst = under; *which = i; *to = -1; }}
        }}
    }}
    if (*which >= 0) return 0;
    for (int i = 0; i < {m}; ++i) {{
        if (!held[i]) continue;
        /* R u + c + A^T mu + nu = 0, with nu >= 0 at an upper bound and <= 0 at a lower. */
        const double wrong = held[i] > 0 ? -v[row] : v[row];
        if (wrong > 1e-12 * (1.0 + fabs(v[row])) && wrong > worst) {{ worst = wrong; *which = i; *to = 0; }}
        ++row;
    }}
    return *which < 0;
}}

/* The controls u and the multipliers: per limit of the set (ms) and summed per root (mr).
   With bounds this is a small strictly convex quadratic program, whose solution is
   unique: an active set from nothing held, and every combination if that cycles. */
static int {P}r_solve(int id, double t, const double* z, const double* p, double* u, double* ms,
                      double* mr, double* K, int* piv, double* c0, int* dout) {{
    if (id < 0 || id >= {len(sets)}) return 3;
    const int s = {P}r_size[id];
    double R[{m * m}], limits[{nr * m + nr}], v[{dim}], lo[{m}], hi[{m}];
    int held[{m}], rc = 0, solved = 0;
    {P}r_quadratic(t, z, p, R, c0);
    {P}r_limits(t, z, p, 0, 0, limits);
    {P}control_bounds(t, z, p, lo, hi);
    for (int i = 0; i < {m}; ++i) held[i] = 0;
    for (int iter = 0; iter < {4 * m + 8} && !solved; ++iter) {{
        int which, to;
        rc = {P}r_system(id, R, limits, c0, lo, hi, held, K, piv, v, dout);
        if (rc) break;
        if ({P}r_check(id, v, lo, hi, held, &which, &to)) solved = 1;
        else held[which] = to;
    }}
    if (!solved) {{
        int pow3 = 1;
        for (int i = 0; i < {m}; ++i) pow3 *= 3;
        for (int code = 0; code < pow3 && !solved; ++code) {{
            int c_ = code, which, to;
            for (int i = 0; i < {m}; ++i) {{ held[i] = c_ % 3 - 1; c_ /= 3; }}
            if ({P}r_system(id, R, limits, c0, lo, hi, held, K, piv, v, dout)) continue;
            if ({P}r_check(id, v, lo, hi, held, &which, &to)) solved = 1;
        }}
        if (!solved) return 4;     /* these limits cannot be ridden within the bounds */
    }}
    for (int i = 0; i < {m}; ++i) u[i] = v[i];
    for (int k = 0; k < {nr}; ++k) mr[k] = 0.0;
    for (int i = 0; i < s; ++i) {{
        const int c = {P}r_set[id][i];
        ms[i] = v[{m} + i];
        mr[{P}r_root[c]] += {P}r_sign[c] * v[{m} + i];
    }}
    return 0;
}}

#define {P}R_LOCALS \\
    double u[{m}], ms[{mset}], mr[{max(nr, 1)}], K[{dim * dim}], c0[{m}]; int piv[{dim}], d; \\
    int rc = {P}r_solve(c, t, z, p, u, ms, mr, K, piv, c0, &d); \\
    if (rc) return rc

int {P}riding_rhs(int c, double t, const double* z, const double* p, double* dz) {{
    {P}R_LOCALS;
    double Hx[{n}], Lx[{n}];
    {P}r_flow(t, z, p, u, dz, Hx);
    {P}r_lagrangian_x(t, z, p, u, mr, Lx);
    for (int i = 0; i < {n}; ++i) dz[{n} + i] = -Hx[i] - Lx[i];
    return 0;
}}

/* dy/dz, y = (u, ms), as a {dim} x {nz} array, from the system already factored. */
static int {P}r_sensitivity(int c, double t, const double* z, const double* p, const double* u,
                            const double* mr, const double* K, const int* piv, int d, double* D,
                            const double* Fu, const double* Hxu, const double* Lxu, double* Gx) {{
    const int s = {P}r_size[c];
    double v[{dim}];
    {P}r_rate_grad(t, z, p, u, mr, Gx);
    for (int j = 0; j < {nz}; ++j) {{
        for (int i = 0; i < {m}; ++i) {{
            v[i] = j < {n} ? -(Hxu[j * {m} + i] + Lxu[j * {m} + i]) : -Fu[(j - {n}) * {m} + i];
        }}
        for (int i = 0; i < s; ++i) {{
            const int l = {P}r_set[c][i];
            v[{m} + i] = j < {n} ? -{P}r_sign[l] * Gx[{P}r_root[l] * {n} + j] : 0.0;
        }}
        for (int i = {m} + s; i < d; ++i) v[i] = 0.0;   /* a held control does not move */
        {P}r_lu_solve(K, d, piv, v);
        for (int i = 0; i < d; ++i) D[i * {nz} + j] = v[i];
    }}
    return 0;
}}

int {P}riding_jac(int c, double t, const double* z, const double* p, double* A) {{
    {P}R_LOCALS;
    double Fx[{n * n}], Fu[{n * m}], Hxx[{n * n}], Hxu[{n * m}];
    double L2[{n * n + n * m}], Gx[{nr * n}], D[{dim * nz}];
    const double* Lxx = L2;
    const double* Lxu = L2 + {n * n};
    const int s = {P}r_size[c];
    {P}r_derivatives(t, z, p, u, Fx, Fu, Hxx, Hxu);
    {P}r_lagrangian_hess(t, z, p, u, mr, L2);
    {P}r_sensitivity(c, t, z, p, u, mr, K, piv, d, D, Fu, Hxu, Lxu, Gx);
    for (int j = 0; j < {nz}; ++j) {{
        for (int i = 0; i < {n}; ++i) {{
            double a = j < {n} ? Fx[i * {n} + j] : 0.0;
            double b = j < {n} ? -(Hxx[i * {n} + j] + Lxx[i * {n} + j]) : -Fx[(j - {n}) * {n} + i];
            for (int q = 0; q < {m}; ++q) {{
                a += Fu[i * {m} + q] * D[q * {nz} + j];
                b -= (Hxu[i * {m} + q] + Lxu[i * {m} + q]) * D[q * {nz} + j];
            }}
            for (int q = 0; q < s; ++q) {{
                const int l = {P}r_set[c][q];
                b -= {P}r_sign[l] * Gx[{P}r_root[l] * {n} + i] * D[({m} + q) * {nz} + j];
            }}
            A[i * {nz} + j] = a;
            A[({n} + i) * {nz} + j] = b;
        }}
    }}
    return 0;
}}

int {P}riding_hamiltonian(int c, double t, const double* z, const double* p, double* H) {{
    {P}R_LOCALS;
    double L;
    {P}r_value(t, z, p, u, H, &L);
    return 0;
}}

int {P}riding_cost_rate(int c, double t, const double* z, const double* p, double* L) {{
    {P}R_LOCALS;
    double H;
    {P}r_value(t, z, p, u, &H, L);
    return 0;
}}

int {P}riding_control(int c, double t, const double* z, const double* p, double* out) {{
    {P}R_LOCALS;
    for (int i = 0; i < {m}; ++i) out[i] = u[i];
    return 0;
}}

int {P}riding_multiplier(int c, double t, const double* z, const double* p, double* mu) {{
    {P}R_LOCALS;
    for (int i = 0; i < {P}r_size[c]; ++i) mu[i] = ms[i];
    return 0;
}}

static int {P}r_gradients(int c, double t, const double* z, const double* p, double* u,
                          double* ms, double* c0, double* D, double* Fu) {{
    double mr[{max(nr, 1)}], K[{dim * dim}], Fx[{n * n}], Hxx[{n * n}], Hxu[{n * m}];
    double L2[{n * n + n * m}], Gx[{nr * n}];
    int piv[{dim}], d;
    int rc = {P}r_solve(c, t, z, p, u, ms, mr, K, piv, c0, &d);
    if (rc) return rc;
    {P}r_derivatives(t, z, p, u, Fx, Fu, Hxx, Hxu);
    {P}r_lagrangian_hess(t, z, p, u, mr, L2);
    return {P}r_sensitivity(c, t, z, p, u, mr, K, piv, d, D, Fu, Hxu, L2 + {n * n}, Gx);
}}

int {P}riding_multiplier_grad(int c, double t, const double* z, const double* p, double* Mt,
                              double* Mz) {{
    double u[{m}], ms[{mset}], c0[{m}], D[{dim * nz}], Fu[{n * m}];
    int rc = {P}r_gradients(c, t, z, p, u, ms, c0, D, Fu);
    if (rc) return rc;
    for (int i = 0; i < {mset}; ++i) Mt[i] = 0.0;
    for (int i = 0; i < {mset * nz}; ++i) Mz[i] = 0.0;
    for (int i = 0; i < {P}r_size[c]; ++i)
        for (int j = 0; j < {nz}; ++j) Mz[i * {nz} + j] = D[({m} + i) * {nz} + j];
    return 0;
}}

int {P}riding_hamiltonian_grad(int c, double t, const double* z, const double* p, double* Ht,
                               double* Hz) {{
    double u[{m}], ms[{mset}], c0[{m}], D[{dim * nz}], Fu[{n * m}];
    double dx[{n}], Hx[{n}], Huv[{m}], R[{m * m}], c1[{m}];
    int rc = {P}r_gradients(c, t, z, p, u, ms, c0, D, Fu);
    if (rc) return rc;
    {P}r_flow(t, z, p, u, dx, Hx);
    {P}r_quadratic(t, z, p, R, c1);
    for (int i = 0; i < {m}; ++i) {{
        Huv[i] = c1[i];
        for (int j = 0; j < {m}; ++j) Huv[i] += R[i * {m} + j] * u[j];
    }}
    Ht[0] = 0.0;
    for (int j = 0; j < {nz}; ++j) {{
        double g = j < {n} ? Hx[j] : dx[j - {n}];
        for (int i = 0; i < {m}; ++i) g += Huv[i] * D[i * {nz} + j];
        Hz[j] = g;
    }}
    return 0;
}}
"""
    return functions, "\n\n".join(parts), text
