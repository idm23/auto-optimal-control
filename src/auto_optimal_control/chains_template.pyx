# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, language_level=3
"""The closed-form free arc of integrator chains (``closed_form.py``), in C.

Generic: the chains, the law's constants and the quadrature come in as arrays,
so one compiled module serves every certified problem."""

import numpy as np

from libc.math cimport sqrt, fabs, copysign, pow

cdef enum:
    MAXL = 8
    MAXE = 16


cdef inline int _roots(int L, const double* a, double lo_t, double hi_t, double* out) noexcept nogil:
    """Real roots in (lo_t, hi_t) of a[0] + a[1] s + ... (L <= 3); returns their number."""
    cdef int k = 0
    cdef double A, B, C, disc, sq, q, x
    if L == 2:
        if a[1] != 0.0:
            x = -a[0] / a[1]
            if lo_t < x < hi_t:
                out[k] = x; k += 1
        return k
    A, B, C = a[2], a[1], a[0]
    if A == 0.0:
        if B != 0.0:
            x = -C / B
            if lo_t < x < hi_t:
                out[k] = x; k += 1
        return k
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return 0
    sq = sqrt(disc)
    q = -0.5 * (B + copysign(sq, B))
    x = q / A
    if lo_t < x < hi_t:
        out[k] = x; k += 1
    if q != 0.0:
        x = C / q
        if lo_t < x < hi_t:
            out[k] = x; k += 1
    return k


cpdef double flow(double tau, const double[::1] z0, const long[:, ::1] ch, int n,
         const double[::1] lo, const double[::1] hi, const double[::1] r, const double[::1] c,
         const double[::1] cl, const double[::1] cq, const double[::1] gx, const double[::1] gw,
         double[::1] z_out, double[:, ::1] phi, bint stm, bint cost):
    """Fill ``z_out`` (and ``phi``, zeroed by the caller) with the free arc's flow over
    ``tau``; return ``int (L(u) - L(0)) dt`` when ``cost``. Chains of length 2 or 3."""
    cdef int nc = ch.shape[0], L = ch.shape[1], G = gx.shape[0]
    cdef int i, m, k, d, e, g, piece, ne, nr, j
    cdef double fact[MAXL + 1]
    cdef double S[MAXL], Lam[MAXL], coef[MAXL], a[MAXL], I[MAXL], dI[MAXL][MAXL]
    cdef double edges[MAXE + 2], roots[4]
    cdef double lo_t = tau if tau < 0.0 else 0.0
    cdef double hi_t = 0.0 if tau < 0.0 else tau
    cdef double total = 0.0, s, wq, half, mid, lam_top, u_int, u, sp_, x, tmp, Kv, ri
    cdef double levels[2]
    cdef bint inside
    if L < 2 or L > 3:
        raise ValueError("compiled chains have length 2 or 3")
    fact[0] = 1.0
    for k in range(1, MAXL + 1):
        fact[k] = fact[k - 1] * k
    with nogil:
        for i in range(nc):
            ri = r[i]
            for m in range(L):
                S[m] = z0[ch[i, m]]
                Lam[m] = z0[n + ch[i, m]]
                I[m] = 0.0
                for k in range(L):
                    dI[m][k] = 0.0
            sp_ = 1.0
            for d in range(L):
                coef[d] = Lam[L - 1 - d] * sp_ / fact[d]
                sp_ = -sp_
            coef[0] += c[i]
            # The pieces: 0, tau, and where u reaches a bound.
            ne = 0
            edges[ne] = 0.0; ne += 1
            edges[ne] = tau; ne += 1
            levels[0] = -ri * hi[i]
            levels[1] = -ri * lo[i]
            for j in range(2):
                for d in range(L):
                    a[d] = coef[d]
                a[0] -= levels[j]
                nr = _roots(L, a, lo_t, hi_t, roots)
                for k in range(nr):
                    edges[ne] = roots[k]; ne += 1
            for k in range(1, ne):                    # insertion sort, ascending
                x = edges[k]
                m = k - 1
                while m >= 0 and edges[m] > x:
                    edges[m + 1] = edges[m]
                    m -= 1
                edges[m + 1] = x
            if tau < 0.0:                             # 0 down to tau
                for k in range(ne // 2):
                    x = edges[k]; edges[k] = edges[ne - 1 - k]; edges[ne - 1 - k] = x
            for piece in range(ne - 1):
                half = 0.5 * (edges[piece + 1] - edges[piece])
                if half == 0.0:
                    continue
                mid = 0.5 * (edges[piece + 1] + edges[piece])
                for g in range(G):
                    s = mid + half * gx[g]
                    wq = half * gw[g]
                    lam_top = 0.0
                    x = 1.0
                    for d in range(L):
                        lam_top += coef[d] * x
                        x *= s
                    u_int = -lam_top / ri
                    inside = lo[i] < u_int < hi[i]
                    u = u_int if inside else (hi[i] if u_int >= hi[i] else lo[i])
                    for m in range(L):
                        e = L - 1 - m
                        Kv = pow(tau - s, e) / fact[e]
                        I[m] += wq * Kv * u
                        if stm and inside:
                            for k in range(L):
                                dI[m][k] -= wq * Kv * pow(-s, L - 1 - k) / (fact[L - 1 - k] * ri)
                    if cost:
                        total += wq * (cl[i] * u + cq[i] * u * u)
            for m in range(L):
                x = I[m]
                for k in range(m, L):
                    x += S[k] * pow(tau, k - m) / fact[k - m]
                z_out[ch[i, m]] = x
                x = 0.0
                for k in range(m + 1):
                    x += Lam[k] * pow(-tau, m - k) / fact[m - k]
                z_out[n + ch[i, m]] = x
            if stm:
                for m in range(L):
                    for k in range(m, L):
                        phi[ch[i, m], ch[i, k]] = pow(tau, k - m) / fact[k - m]
                    for k in range(m + 1):
                        phi[n + ch[i, m], n + ch[i, k]] = pow(-tau, m - k) / fact[m - k]
                    for k in range(L):
                        phi[ch[i, m], n + ch[i, k]] = dI[m][k]
    return total


def sample(const double[::1] taus, const double[::1] z0, const long[:, ::1] ch, int n,
           const double[::1] lo, const double[::1] hi, const double[::1] r, const double[::1] c,
           const double[::1] cl, const double[::1] cq, const double[::1] gx, const double[::1] gw,
           double[:, ::1] Z):
    """``Z[j]`` the free arc's state from ``z0`` after ``taus[j]``, for every ``j``."""
    cdef double[:, ::1] phi = np.zeros((1, 1))
    cdef Py_ssize_t j
    for j in range(taus.shape[0]):
        flow(taus[j], z0, ch, n, lo, hi, r, c, cl, cq, gx, gw, Z[j], phi, False, False)
