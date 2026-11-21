# Lemma register

This note lists every lemma the solver's rules rest on, in a form meant for
formalizing later (in Lean with Mathlib, or by hand to journal standard). The
arguments themselves are in `structure_certificates.md`. Each entry records:

- the statement with all its hypotheses;
- the status of the proof;
- what it depends on;
- what a formal proof would need;
- the rules that use it.

Status keys:

| Key | Meaning |
|---|---|
| **proved** | The argument is complete in the note, given the cited standard results. |
| **sketch** | Every step is in place, but one step leans on a standard result whose hypotheses have not been checked in full. |
| **conjecture** | Supported by computation only. Not used to accept an answer. |
| **refutation** | Shows that a candidate rule cannot work. |

Standard results cited, and so prerequisites of any formalization:

| | Result | Where it is used |
|---|---|---|
| S1 | The maximum principle with state constraints, in the direct-adjoining form, with a nonnegative measure (Hartl, Sethi & Vickson 1995, Thm 4.1), plus its relation to the indirect (BDD) form | H, J, K, Q |
| S2 | The implicit function theorem, and Hadamard's lemma (a smooth function that vanishes on $w = 0$ is $w$ times a smooth function) | E |
| S3 | Smooth dependence of ODE solutions on initial values, times and parameters, including backward integration | E, R10 |
| S4 | Sard's theorem, or the weaker fact that a $C^1$ image of a lower-dimensional manifold is Lebesgue-null | D |
| S5 | Variational inequalities for convex minimization over a convex set | J, I′, Q |
| S6 | The kinetic energy is $\tfrac12 \dot q^\top M(q) \dot q$ for the Newton-Euler model | A |
| S7 | Danskin's theorem: directional derivatives of $\max_{t \in T} \phi(t, s)$ over a compact $T$ | N, V2 |
| S8 | The parametric transversality (Sard) theorem, as in Chow, Mallet-Paret & Yorke (1978) | V3, V5 |
| S9 | Pseudo-arclength continuation through a simple fold (Keller 1977) | V4 (E8) |

Mathlib has S2 (`HasStrictFDerivAt.implicitFunction`), S5 (as first-order
optimality on convex sets), S4 for $C^1$ maps between equal dimensions
(`MeasureTheory.addHaar_image_eq_zero_of_differentiableOn_of_addHaar_eq_zero`),
and ODE existence and Lipschitz dependence (`IsPicardLindelof`, Gronwall).
Smooth dependence of ODE flows on data (S3) is partial. S1 is not in Mathlib
and would be the largest piece of work. Every lemma below that uses S1 can be
stated *relative to* S1, taking its conclusions as hypotheses, and formalized
that way first.

## The class

Throughout:

- $\dot x = F(x) + G u$ with $G$ constant and of full column rank (C1);
- the running cost is $1 + \tfrac\varepsilon2 u^\top R_0 u$ with
  $R = \varepsilon R_0 \succ 0$ constant (C2);
- $U = \prod_k [m_k - r_k, m_k + r_k]$ is a box;
- each limit $g(x) \le 0$ has order 1, with $\dot g = a(x)^\top u + d(x)$ and
  $a = G^\top \nabla g$ (C3).

For the arm, $a = \pm M(q)_{i,:}$ with $M \succ 0$ (C4). The *holdability
margin* is
$$m(x) = \sum_k |a_k(x)|\, r_k - |a(x)^\top \bar u + d(x)|, \qquad \bar u = m .$$
The *corner* for $g$ at $x$ is the vertex $u^v$ of $U$ that minimizes
$a^\top u$ when $a^\top \bar u + d > 0$, and maximizes it otherwise. The
margin is zero exactly when the corner is the only control that holds $g$.

## Entries

### A. Positive definite mass matrix — proved

- **Statement.** If every link's inertia about its centre of mass is positive
  definite and every joint axis is a unit vector, then $M(q) \succ 0$ for all $q$.
- **Depends on.** S6.
- **To formalize.** A serial chain as a list of links; the angular velocity
  of link $i$ as $\sum_{l \le i} \dot q_l z_{l-1}$; the "first nonzero rate"
  argument. Elementary linear algebra once S6 is stated as a definition.
- **Used by.** Certificate C4; Lemma B.

### B. Nonsingular riding — proved

- **Statement.** Assume C2, C3 and C4. Let $S$ be a set of limits on distinct
  joints and $B$ a set of held controls disjoint from $S$. Then the KKT
  matrix $K = \begin{pmatrix} R & C^\top \\ C & 0 \end{pmatrix}$, with
  $C = (\sigma_S M_{S,:};\ E_B)$, is nonsingular. The riding control is
  $C^\infty$ wherever $S$ and $B$ are fixed.
- **Depends on.** A; C5 for smoothness.
- **To formalize.** Pure linear algebra: a principal submatrix of a positive
  definite matrix is positive definite, and the saddle-point matrix lemma.
  This is a good first target.
- **Gap.** It says nothing when $S \cap B \ne \emptyset$, that is, at the
  edge of the holdable set.
- **Used by.** Riding code; Lemma E (H0).

### SP. The riding law sees the costates only through $F_u^\top\lambda$ — proved

- **Statement.** Let $H = L(x, u) + \lambda^\top f(x, u)$ with $f$ affine in $u$,
  $f = f_0(x) + F_u(x)\,u$, and let the ridden limits' rates
  $h_k^{(q_k)} = A_k(x)\,u + b_k(x)$ and the control bounds depend on $x$ only. Then
  on a riding arc the control and the multipliers $(u, \mu)$ (Lemma B's system, held
  controls included) depend on $\lambda$ only through $F_u(x)^\top\lambda$. So for
  every costate $\lambda_j$ whose row $j$ of $F_u$ is zero,
  $\partial(u, \mu)/\partial\lambda_j = 0$ on the arc.
  - For a jerk chain, $\dot x = (\dot q, \ddot q, u)$, $F_u = (0; 0; I)$: the law moves
    with $\lambda_{\ddot q}$ alone. 12 of the six-joint arm's 18 costates do not move it.
- **Proof.** In Lemma B's system, $R = \partial^2 H/\partial u^2 = \partial^2 L/\partial u^2$
  (since $f$ is affine in $u$), and the limits' rows and the bounds are functions of $x$.
  The right-hand side's first block is $-\partial H/\partial u|_{u=0} =
  -\partial L/\partial u(x, 0) - F_u(x)^\top\lambda$. So $\lambda$ enters only there,
  through $F_u^\top\lambda$, and by the implicit function theorem the solution is a
  function of $(x, F_u^\top\lambda)$ wherever the held set is fixed. ∎
- **What it saves.**
  - The riding Jacobian needs $\mathrm{d}(u, \mu)/\mathrm{d}z$: one solve with the
    factored system per column of $z$. By SP, the columns of the costates outside
    $F_u$'s rows have a zero right-hand side, so their solution is zero, and their
    terms in the costate rows' Jacobian need no products.
  - The code tests for a zero right-hand side at run time. That is exact, and it
    covers further zeros that SP does not predict.
  - On goal 3 at 80% ($k = 100$, release rules) this takes 45.1 s to 43.5 s, with the
    route unchanged.
- **Depends on.** B.
- **Used by.** The riding kernels (`riding.py`: `r_sensitivity`, `riding_rhs_jac`).

### SP2. A riding needs only its own roots — proved

- **Statement.** On a boundary arc riding set $S$, with roots $k(S)$, the multipliers of
  the other roots are zero ($\mu_k = 0$ for $k \notin k(S)$). So the arc's right-hand
  side, its Jacobian and its multipliers' sensitivities read only:
  - the rows $(A_k, b_k)$ of its own roots;
  - their rate gradients $\nabla_x g_k^{(q_k)}$;
  - $L_x = \sum_{k \in k(S)} \mu_k \nabla_x g_k^{(q_k)}$.
- **Proof.** Lemma B's system has one row per limit of $S$, and the costates' equation
  carries $\sum_k \mu_k \nabla_x g_k^{(q_k)}$ with $\mu$ zero off $S$. ∎
- **What it saves.** The generated code computed every root's rows and gradients at each
  evaluation. On the arm's rate-limited problem the tape counts are:

  | | all 12 roots | one torque root |
  |---|---|---|
  | rows | 3,207 steps | 251 (joint 6) to 2,562 (joint 1), about 1,500 on average |
  | gradients | 14,173 steps | 700 to 5,492 |
  | $\mu$-weighted $L_x$ | 6,491 steps | — |

  The rate roots take none: their rows are constant (T″). Each root now has its own
  rows and gradient function. A set uses its own roots' functions where their steps
  add up to less than 0.8 of the whole, a table fixed when the code is generated, and
  takes $L_x$ from its roots' gradients.
- **Seen.** On goal 3 at 80% ($k = 100$, release rules with R60): 34.8 s to 30.7 s. The
  arm's Jacobians pass the same central-difference test.
- **Depends on.** B.
- **Used by.** The riding kernels (`riding.py`: `r_limits_of`, `r_gx_of`, `r_lx_of`).

### C. The old path makes births coincide — proved

- **Statement.** Take the route $o(s) = (1 - s)(p + \epsilon)$ from the
  unconstrained solution. Limit $c$ is born at
  $s_c = \epsilon_c / (p_c + \epsilon_c)$, provided no other limit is born
  before it.
- **To formalize.** Trivial once "the extremal does not depend on inactive
  offsets" is a hypothesis.
- **Used by.** Motivation for R1.

### D. Random offsets separate events — sketch

- **Statement.** If $\mathcal{E}_2 \subset \mathbb{R}^N$ is a countable
  union of $C^1$ submanifolds of dimension at most $N - 2$, and $o(0)$ has a
  density, then almost surely the ray $\{(1 - s)\, o(0)\}$ misses
  $\mathcal{E}_2$ for $s < 1$.
- **Depends on.** S4, in the form "the $C^1$ image of an $(N-1)$-manifold
  in $\mathbb{R}^N$ is null".
- **Gap.** The hypothesis on $\mathcal{E}_2$ is not proved for the class. It
  needs a transversality argument: each event is a regular level set, by
  Lemma E's L1 to L3. Formalize the conditional statement first.
- **Used by.** R1, R15.

### E. Birth normal form (two-sided, interior jerks) — proved

- **Statement.** Assume:
  - H0: C2 to C4, with the jerks strictly interior at the contact $t^*$;
  - L1: $\ddot g(t^*) < 0$;
  - L2: the shooting Jacobian $\Phi_v$ is nonsingular;
  - L3: the peak excess crosses zero transversally, $\pi_c'(s^*) \ne 0$.

  Then the shooting system with the new arc of width $w$ and jump
  $\pi = w \tilde p$ has a unique smooth branch $(v, t_m, s, \tilde p)(w)$
  near $w = 0$, and $s - s^* = -\ddot g\, w^2 / (8 \pi_c') + O(w^3)$.
- **Depends on.** B, S2, S3, and the envelope theorem.
- **To formalize.** The determinant factorization of step 4 is linear
  algebra. Step 3 needs Hadamard's lemma in the parameter $w$. The flows
  are taken as smooth maps by hypothesis (S3).
- **Gap.** Jerks at bounds at $t^*$ (a corner contact) are not covered; see
  the one-sided entry below.
- **Used by.** R2, R10, R26 (a violation separated from its limit's arc by a
  stretch with $g < 0$ is a birth, not a widening), R32 (a birth the exact
  check sees below the route's threshold).

### F. Time reversal — proved

- **Statement.** Suppose involutions $\mathcal{R}$ on states and $\Sigma$
  on controls satisfy all of:
  - $f(\mathcal{R}x, \Sigma u) = -\mathcal{R} f(x, u)$;
  - $L(\Sigma u) = L(u)$;
  - the set of limits is mapped to itself;
  - $\Sigma U = U$.

  Then extremals map to extremals by $\tilde x(t) = \mathcal{R} x(T - t)$,
  $\tilde\lambda(t) = -\mathcal{R}\lambda(T - t)$, and
  $\tilde\mu(T - t) = \pi - \mu(t)$ on order-1 arcs.
- **Depends on.** The extremal conditions as a definition, so that S1 is not
  needed.
- **To formalize.** Direct computation. A good first target.
- **Used by.** R4.

### G. Cyclic coordinates — proved

- **Statement.** If $x_k$ appears nowhere in the dynamics, costs or limits,
  then $\lambda_k$ is constant.
- **To formalize.** One line.

### H. The multiplier does not rise — proved, relative to S1

- **Statement.** In the indirect form $H + \sum_c \mu_c \dot g_c$, on each
  order-1 arc of $c$: $\mu_c \ge 0$, $\dot\mu_c \le 0$,
  $\mu_c(t_{\rm out}) = 0$ (off the corner; see K) and
  $\mu_c(t_{\rm in}^+) = \pi_c$. This holds for each limit separately,
  also while other limits are active.
- **Depends on.** S1 (the measures $\eta_c \ge 0$), and the identity
  $\lambda_{\rm ind} = \lambda_{\rm dir} - \sum_c \mu_c \nabla g_c$.
- **To formalize.** Given S1's conclusions as hypotheses, it is
  differentiation and sign bookkeeping.
- **Used by.** Acceptance (the $\dot\mu \le 0$ check), R5, R33 (which arc R5
  splits: H holds per limit, and scaling $g_c$ scales $\mu_c$).

### H2. On an order-2 arc the multiplier is convex, falls and ends at zero — proved, relative to S1

- **Statement.** Let $c$ be a limit of order 2 ($g$ and $\dot g$ free of the
  controls, $\ddot g$ affine in them), ridden on $[t_{\rm in}, t_{\rm out}]$, with
  $\mu$ the multiplier of $\ddot g$ in the indirect form $H + \mu \ddot g$ and
  $\lambda$ continuous at the exit. Then on the arc
  $$\mu \ge 0, \qquad \dot\mu \le 0, \qquad \ddot\mu \ge 0, \qquad \mu(t_{\rm out}) = 0,$$
  and the entry jump $\pi = (\pi_1, \pi_2)$, with
  $\lambda(t_{\rm in}^+) - \lambda(t_{\rm in}^-) = -\pi_1 \nabla g - \pi_2 \nabla \dot g$
  (the sign convention of order 1's $\pi \nabla g$), has $\pi_2 = \mu(t_{\rm in}^+) \ge 0$ and
  $\pi_1 \ge 0$. So $\pi_2$ is the largest value of $\mu$ on the arc, and the arc
  carries no multiplier past its entry when $\pi_2 = 0$. As H, this holds for each
  limit separately, also while other limits are active.
- **Proof.** In the direct form the limit is adjoined with a measure
  $d\eta \ge 0$ on $[t_{\rm in}, t_{\rm out}]$: a density $\eta$ inside and atoms
  $\nu_{\rm in}, \nu_{\rm out}$ at the junctions (S1). Try
  $\lambda_{\rm ind} = \lambda_{\rm dir} + a \nabla g - \mu \nabla \dot g$ on the arc, with
  $\dot\lambda_{\rm dir} = -\partial_x(L + \lambda_{\rm dir} f) - \eta \nabla g$.
  With $\dot g = \nabla g\, f$, $\ddot g = \nabla \dot g\, f$ and the identity
  $\tfrac{d}{dt}\nabla h + \nabla h\, f_x = \nabla \dot h$ along the motion (for $h$
  free of $t$), differentiating and comparing with
  $-\partial_x (L + \lambda_{\rm ind} f + \mu \ddot g)$ leaves
  $$(\dot a - \eta)\,\nabla g + (a - \dot\mu)\,\nabla \dot g = 0 .$$
  So, where the two gradients are independent (Lemma B's rank), $a = \dot\mu$ and
  $\ddot\mu = \eta \ge 0$; the $u$-terms agree because $\dot g$ is free of $u$. Past
  the exit the two costates are equal and $\lambda_{\rm ind}$ is continuous, while
  $\lambda_{\rm dir}$ jumps by $-\nu_{\rm out} \nabla g$: so $\mu(t_{\rm out}) = 0$ and
  $\dot\mu(t_{\rm out}) = -\nu_{\rm out} \le 0$. A convex function that ends at zero
  with a slope $\le 0$ has $\dot\mu \le 0$ and $\mu \ge 0$ before:
  $\mu(t) = \nu_{\rm out}(t_{\rm out} - t) + \int_t^{t_{\rm out}} (s - t)\, \eta(s)\, ds$.
  At the entry, $\lambda_{\rm dir}$ jumps by $-\nu_{\rm in} \nabla g$ and
  $\lambda_{\rm ind}$ by $-(\nu_{\rm in} - \dot\mu(t_{\rm in}^+))\nabla g - \mu(t_{\rm in}^+)\nabla \dot g$,
  so $\pi_2 = \mu(t_{\rm in}^+)$ and
  $\pi_1 = \nu_{\rm in} + \nu_{\rm out} + \int \eta \ge 0$. $\square$
- **Seen.** On the rejected answer of `vgoal0@80` (2026-11-18), the order-2 arcs
  that the theorem allows (`w4_min`, `w6_max`) have $\mu$ convex and falling to 0,
  with $\pi_2 = \mu(t_{\rm in}^+)$ to six digits. `w5_max` and `w2_min` have
  $\pi_2 < 0$. Along the route, $\mu$ on `w5_max` first rises inside the arc
  (about $|o| = 1.25$), dips below zero there, and only then does $\pi_2$ turn
  negative.
- **Depends on.** S1, B (independent gradients on the arc).
- **Used by.** Acceptance ("mu negative" applies at every order; R36 adds the rise
  check at order 2), R36's split of an order-2 arc where $\mu$ rises, and R43's junction
  rows: $\mu(t_{\rm out}^-) = 0$ at the exit and $\mu(t_{\rm in}^+) = \pi_2$ at the entry,
  simple roots in place of $H$ continuity.

### H′. An arc whose entry multiplier reaches zero dies without moving the motion — proved, relative to S1

- **Statement.** Let $c$ be an order-1 limit ridden on an arc, and suppose
  that along the route its entry multiplier $\pi_c(s) \to 0^+$ as
  $s \to s^*$ with the arc's width staying positive. At $s^*$ the extremal
  also solves the structure without the arc, with the arc's stretch free and
  $g_c \equiv 0$ on it. Past $s^*$, keeping the arc gives $\pi_c < 0$, which
  H forbids. So the only regular continuation is the structure without the
  arc: a death by multiplier, where E2 is a death by width.
- **Proof.**
  1. By H, $0 \le \mu_c \le \pi_c$ on the arc, so $\mu_c \equiv 0$ there at
     $s^*$, and with it $\eta_c = -\dot\mu_c \equiv 0$.
  2. The costate's entry jump is $\pi_c \nabla g_c = 0$.
  3. With $\mu_c = 0$, the minimum condition on the arc is the free one, so
     the riding control is the free minimizer, and the free minimizer keeps
     $\dot g_c = 0$.
  4. So the free flow from the arc's entry state, with the same costate,
     reproduces the arc. Every equation of the structure without the arc
     holds. $\square$
- **Gap.** After $s^*$ the freed stretch may rise above the limit: at
  $s^*$ its peak monitor is zero on a whole interval, not at an isolated
  point. If it does rise, the event is not regular (E9), and the route sees
  a violation that brings back an arc.
- **Evidence.**
  - Goal 5 at 70%: `tau2_min` carried $\pi \approx 5.5 \times 10^{-5}$
    when `tau3_min` was born beside its exit. The birth pushed it to
    $-5.4 \times 10^{-4}$. The route kept the arc for 100 s, $\pi$ fell to
    $-2.4 \times 10^{-2}$, and its exit and `tau3_min`'s crossed back and
    forth.
  - Goal 3 at 80%: the same arc went from about $2 \times 10^{-5}$ to
    $-1.7 \times 10^{-3}$ at `tau3_min`'s activation.
- **Used by.** R31.

### I. Contacts are holdable — proved

- **Statement.** If $u(t) \in U$ a.e., $g(x(t)) \le 0$ near $t^*$ and
  $g(x(t^*)) = 0$ at an interior time, then $m(x(t^*)) \ge 0$. Convexity and
  compactness of $U$ are all that is needed.
- **Depends on.** Continuity of $x(t)$, $a$ and $d$.
- **To formalize.** Elementary real analysis: the image of $U$ is a
  compact interval that varies continuously with $x$, and a positive lower
  bound on $\dot g$ gives a contradiction. A good first target.
- **Used by.** R6, R8, R14.

### I′. A peak on the edge is a corner — proved

- **Statement.** At an interior maximum of $g$ along an admissible motion,
  $m \ge 0$, with equality iff $u(t^*)$ is the corner on every $k$ with
  $a_k \ne 0$.
- **To formalize.** The triangle inequality, with its equality case.
- **Used by.** R6 (one-sided).

### J. Junctions are tangential, bounds or not — proved, relative to S1

- **Statement.** Assume C1, C2 and a convex $U$. At an entry or exit of an
  order-1 arc, with no other limit changing there, the control is
  continuous and $a^\top u + d = 0$ on both sides.
- **Depends on.** S1 (a jump $\nu \ge 0$ of the costate at the junction),
  S5.
- **To formalize.** Add the two variational inequalities to get
  $\nu\, a^\top(u^+ - u^-) \ge \|u^+ - u^-\|_R^2$, then use the sign of
  $a^\top(u^+ - u^-)$ from the side conditions. Short once S1 is a
  hypothesis.
- **Used by.** R7.

### J′. Coincident junctions are tangential too — proved, relative to S1

- **Statement.** Assume C1, C2 and a convex $U$. At an instant $t^*$ where several
  limits of order 1 or 2 enter, leave or are touched at once, the control is
  continuous.
- **Proof.** S1 gives $\lambda^+ - \lambda^- = -\sum_c \nu_c \nabla g_c$, with atoms
  $\nu_c \ge 0$. For a limit of order 2, $\dot g_c = \nabla g_c F$ is free of $u$, so
  $G^\top \nabla g_c = 0$. Hence $G^\top(\lambda^+ - \lambda^-) = -\sum_{q_c = 1} \nu_c a_c$:
  only the order-1 atoms reach the control. The pointwise minimisation on each side
  gives (S5) $(R u^\pm + G^\top \lambda^\pm)^\top (v - u^\pm) \ge 0$ for all $v \in U$.
  Taking $v = u^\mp$ and adding gives
  $$\|\Delta u\|_R^2 \le \sum_{q_c = 1} \nu_c\, a_c^\top \Delta u, \qquad \Delta u = u^+ - u^-.$$
  For an order-1 limit, each case gives $a_c^\top \Delta u \le 0$:
  - entering: $\dot g_c(t^{*-}) = d_c + a_c^\top u^- \ge 0$, since $g_c$ rises to 0,
    and $\dot g_c = d_c + a_c^\top u^+ = 0$ on the arc;
  - leaving: the same with the sides exchanged;
  - touched: $\dot g_c(t^{*-}) \ge 0 \ge \dot g_c(t^{*+})$;
  - ridden through $t^*$: $\dot g_c = 0$ on both sides.

  So the right-hand side is $\le 0$, and $\Delta u = 0$. $\square$
- **Consequence.** Every junction row that encodes the control's continuity (rate rows,
  multiplier rows, R43's rows) is necessary at a coincidence too.
- **Depends on.** S1, S5.
- **Used by.** X.

### X. Two entries at one instant: where the orders meet, and when they continue each other — proved, relative to J′ and B (regularity assumed)

- **Setting.** Limit $A$ has order 1, with $a_A = (\nabla g_A G)^\top$. Limit $B$ has
  order 1 or 2, with $b_B$ the gradient in $u$ of its first rate that depends on $u$.
  The two are on distinct joints, so $a_A$ and $b_B$ are independent (B).
  - Structure $\mathcal A$ enters $A$ first: free, then $\{A\}$, then $\{A, B\}$.
  - Structure $\mathcal B$ enters $B$ first: free, then $\{B\}$, then $\{A, B\}$.

  Each is a square system $R_{\mathcal A}(s, w, t_A, t_B) = 0$ or
  $R_{\mathcal B}(s, w, t_A, t_B) = 0$, with the same unknowns and the route's
  parameter $s$. Write $\delta = t_B - t_A$. $\mathcal A$'s solution is valid where
  $\delta \ge 0$, and $\mathcal B$'s where $\delta \le 0$.
- **(i) The joint junction.** At a solution with $t_A = t_B = t^*$ and the control
  interior to $U$ there, J′ and B give the conditions of a joint entry:
  - the states: $g_A = 0$, plus $g_B = \dot g_B = 0$ (order 2) or $g_B = 0$ (order 1);
  - the multipliers: $\mu_A(t^{*+}) = \pi_A$ and $\mu_B(t^{*+}) = \pi_B$, with
    $\pi_B$ the coefficient that reaches the control (the second, $\pi_2$, at order 2).

  For the multipliers, subtract the stationarity of the free law from that of the
  joint riding law, at the same $u$: $(\mu_A - \pi_A) a_A + (\mu_B - \pi_B) b_B = 0$.
  Counting, the joint entry has one fewer unknown than two separate entries and the
  same number of conditions. So it is overdetermined by one, and occurs only at
  isolated values of $s$ (a hypersurface $\Sigma$ in the space of route parameters).
- **(ii) Both orders meet there.** The joint point solves $R_{\mathcal A}$ and
  $R_{\mathcal B}$ at $\delta = 0$. In $\mathcal A$ at zero width, the arc $\{A\}$
  starts from $\lambda' = \lambda^- - \pi_A \nabla g_A$. The free control $u$
  (interior, so stationary for $\lambda^-$) satisfies the riding law there:
  - its stationarity, since
    $R u + r_0 + G^\top \lambda' + \mu_A a_A = (\mu_A - \pi_A) a_A = 0$, where $r_0$ is the
    cost's linear term in $u$;
  - and $\dot g_A = 0$, which holds at the joint point.

  So $\mathcal A$'s entry row for $A$ holds, and $B$'s entry rows are the joint ones.
  $\mathcal B$ is the same with the roles exchanged.
- **(iii) Crossing or fold.** Let both Jacobians in $(w, t_A, t_B)$ be nonsingular at
  the joint point. By the implicit function theorem each order has a unique smooth
  branch through it, and the gaps along them are
  $\delta_{\mathcal A}(s) = \alpha (s - s^*) + O(|s - s^*|^2)$ and
  $\delta_{\mathcal B}(s) = \beta (s - s^*) + O(|s - s^*|^2)$. Take the crossing
  transversal: $\alpha, \beta \ne 0$.
  - If $\alpha\beta > 0$, exactly one order is valid on each side of $s^*$, and it
    continues the other: E5.
  - If $\alpha\beta < 0$, both orders are valid on one side, and neither has a
    solution near the joint point on the other. The coincidence is a fold of the
    route's solution curve, which turns back in $s$ there.

  Both $\delta_{\mathcal A}$ and $\delta_{\mathcal B}$ vanish exactly on $\Sigma$, so
  their gradients in the route's parameters are both normal to it. The sign of
  $\alpha\beta$ is therefore a property of the point of $\Sigma$, not of the
  direction the route crosses it in. $\square$
- **Consequence.** A route that reaches a fold with $s$ increasing finds each order's
  solution on its invalid side (the churn of goal 1 at 70%). No reorder, tie or drop
  of an arc can continue it there. The solution curve continues on the other order's
  branch with $s$ decreasing, which is a turning point. Following it needs arclength
  continuation (Allgower and Georg), not steps in $s$.
- **To check where it is used.**
  - $\alpha$ and $\beta$ come from each order's Jacobian and $\partial_s R$ at a
    converged point near the coincidence: the tangent of the branch, at no extra
    integration.
  - The hypotheses are the control interior at $t^*$ (the box margins), B's rank for
    the pair, and both Jacobians nonsingular.
  - Where any of these fails, the lemma is silent. The cases are Lemma K's corner, a
    singular riding set, or a degenerate junction.
- **Gap.** The $O(|s - s^*|^2)$ terms are uniform only while the Jacobians stay
  nonsingular. This is a standard implicit-function-theorem estimate, but not yet
  written out.
- **Seen.** Where the route solves both orders of a crossed pair at one step and both
  come out crossed (the churn), $\alpha$ and $\beta$ come from each order's tangent
  $-J^{-1}\partial_s R$ along the route's direction.
  - *Goal 1 at 70%, before R43 (2026-11-19).* At the entries of `tau5_min` and `w3_max`,
    $\alpha\beta < 0$ every time. But the Jacobians were near-singular: equilibrated
    condition about $10^{10}$, $\sigma_{\min} \approx 10^{-9}$. The weak direction was
    an order-2 entry's $(t_{\rm in}, \pi_2)$ against its tangency and $H$-continuity
    rows, elsewhere in the structure. At a tangential order-2 entry $g$, $\dot g$ and
    (by J′) $\ddot g$ vanish, so no row fixes $t_{\rm in}$ to first order: the double
    root R43 removes. The fold reading there is not reliable, and the case solves with
    R43.
  - *With R43.* Most churn points are well conditioned: equilibrated condition about
    $10^5$. Goal 0 at 70% has two exits, `w5_max` and `tau5_max`: $\alpha = -0.011$,
    $\beta = +0.062$. Goal 4 at 70% has the exit of `w5_max` and the entry of
    `tau4_max`: $\alpha = -1.1$, $\beta = +2.1$. Both show the fold signature.
  - The free law evaluated at $t^*$ has jerks at their bounds in both cases, but the
    motion is not on the free law there. It rides other limits through $t^*$: `w2_min`,
    `w3_min` and `w4_min` in goal 0 at 70%, `w4_max` in goal 4 at 70%. The riding
    controls of all four arcs around $t^*$ hold no jerk at a bound. The pair's control
    gradients are independent: normalized $\sigma_{\min}$ 0.37 and 0.99.
- **(iv) Held controls.** "Control interior" can be relaxed to "the same held set
  $\mathcal H$ on all four arcs at $t^*$". The held controls sit at their bounds, with
  strictly positive bound multipliers, so $\mathcal H$ is locally constant. Write $F$
  for the free controls. Then:
  - J′ holds as it is, since it needs only a convex $U$.
  - The free law and every riding law are stationary in $u_F$ only, with $u_{\mathcal H}$
    fixed: $\big(R u + r_0 + G^\top \lambda + \sum_c \mu_c a_c\big)_F = 0$.
  - The proof of (i) runs over the components in $F$:
    $(\mu_A - \pi_A)(a_A)_F + (\mu_B - \pi_B)(b_B)_F = 0$. So $\mu_A = \pi_A$ and
    $\mu_B = \pi_B$ follow whenever $(a_A)_F$ and $(b_B)_F$ are independent.
  - (ii) and (iii) are unchanged.
  - The independence is B's rank condition for the pair restricted to $F$. It holds when
    the pair's own controls are free and distinct, since a positive definite mass matrix
    makes the $2 \times 2$ minor on those components nonsingular. It can also hold for two
    limits of one joint.
  - Exits follow from entries by exchanging the sides, and a hand-over by combining the
    two.
- **(v) A background riding set.** Limits ridden through $t^*$ (a set $S_0$ on every
  arc) change nothing in (i) to (iii). "Free" becomes "riding $S_0$", and the rank
  condition is B's for all of $S_0 \cup \{A, B\}$. For a hand-over (one exit, one entry)
  the rows of both orders hold at the joint point:
  - the control and $\lambda^+$ are the same on the joint arc and on the arc after the
    exit;
  - subtracting the two stationarities leaves
    $\sum_c (\mu_c - \mu_c')\, a_c + \mu_{\rm exit}\, a_{\rm exit} = 0$ over $S_0$ and the
    entering limit;
  - independence then gives $\mu_{\rm exit} = 0$, the exit row, in both orders.
- **Applied, with R43.** Goal 0 at 70% (two exits of joint 5's limits, over riding
  `w2_min`, `w3_min`, `w4_min`) and goal 4 at 70% (`w5_max` leaving where `tau4_max`
  enters, over riding `w4_max`) satisfy the hypotheses:
  - interior riding controls;
  - B's rank, checked on the pair;
  - nonsingular Jacobians, raw condition 1.7e11 and 4.5e11 against about 1e15 before
    R43.

  And $\alpha\beta < 0$. Under X they are genuine folds: no structure with the pair
  adjacent continues the route forward there.
- **What a genuine fold asks of the route.** The solution curve continues on the other
  order's branch with $s$ decreasing, until another event changes the structure. Steps
  in $s$ cannot follow it, but arclength continuation can (Allgower and Georg): take the
  other order, and step back along its branch. This only steers. Whether the turned
  curve reaches $s = 1$ is the global question of homotopy path-following: it does when
  the solution set is a bounded regular one-manifold through the start. That is not
  shown here.
- **Followed, R44 (2026-11-19).** At the folds of goals 0 and 4 at 70%, R44 takes the other
  order's branch and follows it by arclength, toward the leg's other end and turning at
  any further fold.
  - In goal 4 at 70% the branch leads back to the leg's start without a second fold. The
    curve through the fold joins two solutions at the start, and does not reach the
    target on this leg's line in the offsets.
  - In goal 0 at 70% the backward run stalls among collapses of short arcs.
  - So these answers lie on another component of the solution set. Reaching them takes
    another path in the offsets, or a structure that differs away from the pair.
- **Used by.** The diagnosis of the entry swaps; R44.

### K. A forced exit is a corner, and its rate row is the margin — proved, relative to S1

- **Statement.** Under J, with $a_k \ne 0$ for all $k$: an exit on the edge
  ($m = 0$) has the corner as its control on both sides. Wherever the free
  law is at that corner, the rate row equals $\mp m(x)$.
- **Depends on.** J, I′.
- **Used by.** R7, R9.

### L. A touch at the corner is degenerate — proved (refutation)

- **Statement.** If the free control is saturated at the corner on an
  interval after a touch at $t_c$, then the costate jump $-\nu\nabla g$ does
  not change the control there. The shooting residual is locally
  independent of $\nu$, and its Jacobian is singular.
- **Depends on.** The box projection is locally constant where saturated.
  Here $G^\top\nabla g = a$ moves the unclipped control along $-R^{-1}a$,
  which keeps it outside the box on the saturated side.
- **Gap.** "On an interval" needs a strict-saturation margin. A formal
  statement should assume the unclipped control is at least $\delta > 0$
  outside the box on $(t_c, t_c + h)$.
- **Used by.** Refutes R12.

## Entries in progress

### L2. A touch of an order-2 limit at its own control's bound is ill-conditioned — sketch, with evidence

- **Statement (sketch).** Let an order-2 limit $g$ (rate $w_j$, $\ddot g = u_j$ on the
  jerk chain) touch at $t_c$, with its own control $u_j$ held at a bound on a
  neighbourhood of $t_c$. Then $\ddot g(t_c) = \pm J$, and the touch is concave (E″) when
  the bound brakes the rate.
  - Under the clipped free law, or a riding law holding $u_j$, the costate directions
    that act on the motion only through $u_j$ reach it only through the times where
    $u_j$ enters and leaves the bound.
  - The touch's jump $-\nu \nabla g$ lies in such a direction: $\lambda_{\dot q_j}$, and
    through the chain $\lambda_{\ddot q_j}$.
  - So the shooting Jacobian has a near-null direction combining $\nu$ with those
    costates. With the jerk cost $\varepsilon$, its smallest singular value goes to 0
    with $\varepsilon$. At $\varepsilon = 0$ it is exactly null, the order-2 analogue of
    L's degeneracy at order 1.
- **Why it is only a sketch.** The rate at which $\sigma_{\min}$ goes to 0, and whether
  the switching times' sensitivity keeps the system regular for every $\varepsilon > 0$,
  are not derived.
- **Seen (2026-11-19).**
  - *The births.* With R43, three rest cases stall at the birth of a rate limit whose
    peak has $\ddot g = -30.000$, the jerk bound: rest 10 (`w6_max`, ridden there
    `tau1_max` and `w3_max`), rest 4 (`w1_max`, free) and rest 5 (`w3_max`, over
    `w1_min`).
  - *The touch solves at rest 10's located birth.* From the seed (excess $2 \times
    10^{-4}$ of scale, residual $1.6 \times 10^{-2}$), the Newton step takes the residual
    to 52, with the Jacobian correct to within 6% of central differences. The
    equilibrated $\sigma_{\min}$ is $6 \times 10^{-8}$. Its direction is the initial
    costates of joints 1 and 6 with the touch's $\nu$, against joint 1's terminal rows.
  - *The direct comparison.* In rest 10's answer, `w6_max` is ridden on
    $[0.748, 0.840]$, and joint 6's jerk is held only on $[0, 0.098]$ and
    $[1.451, 1.576]$. The corner touch is a stage of the route, not the answer: the
    jerk leaves its bound before the arc forms (E″'s touch turning into an arc).
- **What it asks of the route.** The weight of the jerk cost is a route parameter whose
  growth widens Newton's radius (W). A route that meets this birth at a larger weight,
  and lowers the weight after the arc has formed, crosses it where $\sigma_{\min}$ is
  not small.
- **Depends on.** E″, L (the order-1 case), W.

### M. A corner entry is where the margin crosses zero — proved, relative to J

- **Statement.** Assume J and $a_k \ne 0$ for all $k$. Let an order-1 arc
  of $g$ be entered at $t_{\rm in}$, with the arriving control at the
  corner ($u(t_{\rm in}^-) = u^v$). Then:
  - $m(x(t_{\rm in})) = 0$;
  - just before the entry, wherever the control is still the corner,
    $m(x) = -\dot g \le 0$;
  - just after it, $m(x) \ge 0$ on the arc.

  So the motion crosses the edge of the holdable set exactly at the entry,
  from outside to inside.
- **Proof.**
  - At the corner, $a^\top u^v + d = \mp m(x)$, with the sign such that
    $\dot g = -m$ for the side the drift pushes towards (the computation of
    Lemma K).
  - J gives $a^\top u + d = 0$ under the arriving control at $t_{\rm in}$, so
    $m = 0$ there.
  - Before the entry, $g < 0$ and $g \to 0$, so $\dot g \ge 0$ on a sequence
    of times approaching the entry, hence $m \le 0$ there.
  - On the arc, the margin is nonnegative by Lemma I applied pointwise.
    $\square$
- **Consequence.** Riding forward from a corner entry is feasible only if
  $m$ is non-decreasing along the riding flow at $t_{\rm in}^+$. A rider
  entered at the corner where $\dot m < 0$ along the riding flow fails at
  once. That is not a numerical accident: no arc of this limit starts
  there.
- **Gap.** "Non-decreasing" is a first-order condition. The second-order
  case $\dot m = 0$ is not treated.
- **Used by.** Conjectured rule R16 (below).

### N. The peak excess is smooth before a birth — proved

- **Statement.** Along a $C^1$ branch $v(s)$ of a fixed structure, the peak
  excess $P_c(s)$ of an inactive limit has two properties:
  - it is Lipschitz, with one-sided derivatives;
  - it is $C^1$, with derivative $\partial_s g_c$ at the peak, wherever the
    peak is unique, interior and nondegenerate (L1).
- **Depends on.** S2 (for $v(s)$ and $t^*(s)$), S3, Danskin's theorem
  (the derivative of a max over a compact set).
- **To formalize.** Danskin's theorem is not in Mathlib in this form. The
  $C^1$ case needs only the implicit function theorem and the chain rule,
  and is a good first target. Reparametrizing the moving free segments over
  fixed intervals needs care.
- **Gap.** Two peaks of equal height (a kink in $P_c$) give no rate for the
  secant. The rule only needs bracketing there.
- **Used by.** R16.

### E′. One-sided birth at the corner — conjecture

- **Statement (conjectured).** At a first contact whose control is
  saturated at the corner, so that the contact is on the edge (I′), the
  newborn arc is $[t^* - \alpha w, t^* + (1 - \alpha) w]$. Its entry stays
  within $O(w^2)$ of the edge, and $\alpha \to \alpha_0 \in [0, 1)$ as
  $w \to 0$.
- **Evidence.** A direct transcription of the drift toy (comparison only)
  shows the entry within $\pm 0.02$ of the edge at every limit value. The
  entry creeps earlier while the exit moves later fast.
- **What a proof needs.** Lemma E's step 3 replaced by the corner's riding
  law (μ from the last interior control, R14). L1 has to be restated,
  because along a saturated motion $\ddot g$ involves $\dot m$.

### E″. A touch of an order-2 limit is concave — proved (the sign); the event's regularity a sketch

- **Statement.** At a touch point $t^*$ of an order-2 limit $g$ on a feasible motion
  whose control is continuous, $g(t^*) = 0$ and $\dot g(t^*) = 0$, and
  $\ddot g(t^*) \le 0$. A converged structure with $\ddot g(t^*) > 0$ at a touch has
  the limit violated on both sides of it, so it is not an extremal of that structure's
  kind: the touch has become an arc (event E1″).
- **Proof.** $t^*$ is an interior maximum of $g \le 0$, and $g$ is $C^2$ there
  (the control, which enters $\ddot g$, is continuous under the jerk-cost policy).
- **Gap.** The event's normal form: that the arc born at $\ddot g(t^*) = 0$ grows
  linearly in the route parameter (Bryson–Denham: width $1 - 6\ell$), and that the
  touch's jump goes to the arc's entry, are seen, not proved.
- **Evidence.** Goal 0 at 80% with rate limits: a touch with $\ddot g = +9.1$ becomes
  the arc [0.57798, 0.58219] from any width tried; the alternative, two touches at
  its ends, has $\ddot g = +0.05$ at both.
- **Evidence of the linear law (2026-11-20).** Goal 3 at 70% at $k = 100$, the arc of
  `w2_min` near 0.588 dying: over four accepted points at
  $s = 0.882528, 0.882674, 0.882893, 0.883058$ its width falls through
  $0.01150, 0.00846, 0.00384, 0.00032$. The slopes are $20.8$, $21.1$ and $21.3$: linear
  to 2%, where Lemma E's order-1 law would make $w^2$ linear instead.
- **Used by.** R35, R50.

### Z. An order-2 arc dies into a touch with the entry's first jump — proved, relative to H2

- **Statement.** Take an order-2 arc $[t_\mathrm{in}, t_\mathrm{out}]$ of limit $g$ on a
  family of extremals of one structure, with entry jump $(\pi_1, \pi_2)$ and with its
  multiplier's density $\eta = -\dot\mu$ bounded by $\bar\eta$ on the arc. As the
  width $\delta = t_\mathrm{out} - t_\mathrm{in} \to 0$:
  1. $0 \le \pi_2 \le \bar\eta\,\delta \to 0$;
  2. the arc's conditions at its ends tend to those of a touch at
     $t_c = t_\mathrm{in} = t_\mathrm{out}$, namely $g(t_c) = \dot g(t_c) = 0$ and the
     costate jump $\nu\,\nabla g$ with $\nu = \pi_1$.

  So at $\delta = 0$ the arc's solution is the touch's, with jump $\pi_1$. With E″'s
  linear law ($\delta$ linear in $s$), two solutions of the arc's structure that bracket
  the death locate it by the secant in $\delta$ to $O(\Delta s^2)$, and the solution
  interpolated there is a touch's solution to the same order.
- **Proof.**
  1. By H2, $\mu(t_\mathrm{in}^+) = \pi_2$ and $\mu(t_\mathrm{out}^-) = 0$, and $\mu$
     falls on the arc. So $\pi_2 = \int_{t_\mathrm{in}}^{t_\mathrm{out}} \eta\,dt
     \le \bar\eta\,\delta$.
  2. The arc's state conditions are $g = \dot g = 0$ at entry. On the arc the motion
     rides the limit for a time $\delta$, so the state at exit differs from the state at
     entry by $O(\delta)$. The costate's change across $[t_\mathrm{in}, t_\mathrm{out}]$
     is the entry jump $\pi_1 \nabla g + \pi_2 \nabla \dot g$, plus the integral of
     $\mu$ times the riding terms over a length $\delta$, which is $O(\delta)$. By (1)
     everything except $\pi_1 \nabla g$ is $O(\delta)$: the touch's jump.
- **Why it matters.** Before R50 the route stepped past the death with the arc's
  structure, to a solution of negative width (the equations hold there, by continuation),
  and turned that into a touch at the arc's midpoint. From $\delta = -10^{-3}$ that guess
  is poor. The touch solve then fails after 2.3 s, and the step halves toward the death
  (Lemma S's controller), failing again on each try.
- **Evidence (2026-11-20).**
  - Goal 3 at 70%, $k = 100$, the arc of `w2_min` above:
    - without R50, three failed tries and about 7 s pass between $s = 0.88306$ and the
      touch's acceptance at $0.883096$;
    - the touch tried from width $-10^{-5}$ (at $0.883073$) solves at once.
  - With R50, the secant puts the death at $0.883073$, and the touch solves at
    $0.883076$ on the first try.
- **Used by.** R50.

### E3. The split's gap (moment conditions) — leading order derived; sketch

- **Statement.** Take an order-1 arc of limit $c$ whose multiplier $\mu$
  rises somewhere inside it ($\eta = -\dot\mu < 0$). Assume (A3) and L2 as in
  E, and that the released motion keeps the arc's held set to leading order.
  Then the split extremal leaves the limit on a gap $[a, b]$ that straddles
  the rise, with:
  - (i) $\mu(a) = \mu(b)$, the level $\ell$;
  - (ii) $\int_a^b D\,(\mu - \ell)\,dt = 0$, where
    $D = 1/[(A_F R_F^{-1} A_F^\top)^{-1}]_{cc}$. Here $A$ has the rate
    coefficients $a$ of the limits ridden there (other limits stay held),
    and $F$ is the set of controls not at a bound. For one limit, $D$ is
    Lemma P's $a_F^\top R_F^{-1} a_F$.

  The first piece leaves with $\mu = 0$, so its entry jump falls by $\ell$.
  The second enters with $\pi_2 = \ell$.
- **Special cases.**
  - *Smooth dip* (H0, $D$ constant,
    $\eta = \eta_{\min} + \tfrac12\ddot\eta (t - t_m)^2$): the gap is centred
    on $t_m$ with $h^2 = -6\eta_{\min}/\ddot\eta$. That is $\sqrt 3$ times the
    stretch where $\eta < 0$, and $s - s^* = \ddot\eta h^2/(6\kappa)$ under
    L3′.
  - *Held-set window* (Lemma P: $\eta$ negative, $-W$, on one window of
    length $\omega$ between two jumps, with $L, R > 0$ beside it): the gap is
    the window widened by about $W\omega/(2L)$ and $W\omega/(2R)$.
- **Derivation (leading order).**
  1. Released on the gap, the control is
     $u_b + \tilde\mu\,R_F^{-1}P\,a_F$, with $P$ projecting out the limits
     still ridden. So $\dot g = D\,\tilde\mu$.
  2. At the exit the control is continuous, so $\tilde\mu = \mu - \mu(a)$
     on the gap ($\dot{\tilde\mu} = -\tilde\eta$, the density continued from
     before the split).
  3. The limit returns to its level tangentially at $b$:
     - $\dot g(b) = 0$ gives (i);
     - $g(b) = \int_a^b \dot g = 0$ gives (ii).

  The toy $\dot x = u$, $L = \tfrac12 u^2 + \rho(t)x$, $x \le 0$ (where
  $\eta = -\rho$, $D = 1$) has exactly this gap.
- **To formalize.** The implicit function theorem in the gap's width, as in
  E's steps 3 and 4, with the held set's switching times on the gap as
  extra unknowns (Lemma R).
- **Evidence.** At splits on the census cases' first routes:
  - *Smooth dips* (goal 4 at 80% and at 70%): the splits solve to 0.98 and
    1.00 of $h$, centred on $t_m$ to $3 \times 10^{-4}$.
  - *A held-set window* (goal 2 at 70%, `tau5_max` ridden with `tau1_min`,
    where the jerks of joints 1–4 change bounds): the moment seed
    $[0.37732, 0.38005]$ solves to $[0.37734, 0.38009]$. On that gap,
    $-1.6\times 10^{-6} \le g \le -1.2\times 10^{-7}$, and the rise of $\mu$
    falls from $1.8\times 10^{-4}$ to $10^{-12}$. R5's own seed there
    solves to a gap of zero width.
- **Merge (E4).** E3 read backwards; proved below.
- **Gap.** The held set must not change at the gap's ends in a way the
  released motion does not share. Rest 12 (O2), where a ridden limit's own
  jerk saturates, still cycles: it splits, the gap closes, it splits again.
- **Used by.** R27, and R21 through it (the held-set window of goal 8 at 70%);
  R34 (the gap contains the window; wider seeds, goal 3 at 70%).

### E4. A closed gap is the merged arc — proved (order 1), sketch (order 2)

- **Statement.** Let two arcs of one order-1 limit $c$ meet at $t_m$: the first leaves
  at $t_m$ with $\mu(t_m^-) = 0$ (its exit row), the second enters at $t_m$ with jump
  $\pi_2$ and $\mu(t_m^+) = \pi_2$ (its multiplier row). Then the motion and the controls
  are those of one arc on the union, entered with jump $\pi_1 + \pi_2$, whose multiplier
  is $\mu + \pi_2$ on the first piece and $\mu$ on the second, continuous at $t_m$. The
  two-arc parametrization has a one-dimensional kernel there (where the split point
  sits, with the first piece's jump moving with it), so its Jacobian is singular at a
  closed gap.
- **Proof.** On the first piece put $\delta\lambda = -\pi_2 \nabla g$ and
  $\delta\mu = \pi_2$.
  - *Riding law.* $G^\top \delta\lambda = -\pi_2 a$ and $\delta\mu\, a = \pi_2 a$, so the
    riding control's stationarity $R u + r_0 + G^\top\lambda + \mu a = 0$ is unchanged,
    and so is $u$.
  - *Costate equation* $\dot\lambda = -\partial_x(\lambda^\top f + \mu \dot g)$.
    - The change on the right is
      $-\delta\lambda^\top f_x - \delta\mu\, \partial_x \dot g
      = \pi_2 f_x^\top \nabla g - \pi_2 (\nabla^2 g\, f + f_x^\top \nabla g) = -\pi_2 \nabla^2 g\, f$.
    - The change on the left is $\tfrac{d}{dt}(-\pi_2 \nabla g) = -\pi_2 \nabla^2 g\, f$.
    - They agree, so the shifted pair solves the arc's equations.
  - *Jumps.* At $t_m$ the shifted costate meets the second piece's without a jump. At
    $t_{\rm in}$ the first piece's jump grows by $\pi_2$.
  - *Continuity of $\mu$.* $\mu(t_m^-) + \pi_2 = \pi_2 = \mu(t_m^+)$.

  The same extremal is described with any split point inside the union, by the same
  shift. That is the kernel. $\square$
- **Order 2 (sketch).** The interior jump is $-\pi_1 \nabla g - \pi_2 \nabla \dot g$. By
  H2's representation $\lambda_{\rm ind} = \lambda_{\rm dir} + \dot\mu \nabla g - \mu \nabla \dot g$,
  the shift $\delta\mu(t) = \pi_2 + \pi_1 (t_m - t)$ on the first piece produces it.
  Checking the costate equation is the computation of H2's proof, not written out.
- **Consequence.** A gap of width $\le 0$ between two arcs of one limit is the merged arc
  (R11), wherever other limits' arcs start between them. Kept as two arcs, the shooting
  is singular in the split point's direction. A predictor that carries the second arc
  into the first gives a structure with the limit entered twice.
- **Seen.** Goal 3 at 80% with R43 (2026-11-19).
  - Two `tau4_min` arcs met at 0.538538 to six digits, with the second's entry jump
    0.043. The equilibrated Jacobian's smallest singular value was $8 \times 10^{-16}$,
    in the direction of the shared time and the first jump.
  - R11 compared only consecutive boundary arcs of any limit. A `w3_min` arc starting
    between the two (at 0.537) hid them from each other, so they were never merged.
  - The route then failed 1,976 steps with `'tau4_min' is entered twice without an exit`.
- **Depends on.** The indirect form of S1; H2 for order 2.
- **Used by.** R11.

### Observation O1 — answers stay off the edge

On every solved census case, and on the toy's direct comparisons, the margin
of the answer stays at least 0.23 (relative) on free stretches. Corner
states appear only on the route. Not a lemma; it guides where to look.

### Observation O2: rises start where a limit's own jerk saturates

These are probe results on the route's intermediate states, taken before the
route stalls.

- **Rest 12.** On the `tau1_max` arc, $\mu$ falls until joint 1's own jerk
  reaches its bound ($t = 0.3411$). It rises from there, while the limit is
  held through joints 2 and 3 alone. The margin is about 6 throughout, far
  from the edge. This is the case Lemma B excludes ($S \cap B \ne \emptyset$).
  The riding KKT solution stays continuous there, but $\alpha_F = 1/(a_F^\top
  R_F^{-1} a_F)$ jumps by the factor $|a|^2/|a_F|^2 \approx 25$, and so does
  the slope of $\mu$.
- **Goal 9 at 80%.** $\mu$ of `tau2_min` turns negative just before an arc of
  `tau6_min`, the wrist joint. Its $a$ is about $10^{-4}$, its margin 0.012,
  and its entry jump about 1, against about $10^{-5}$ for the other limits.
- **A lemma for the held set changing on an arc: Lemma U.** When $\eta$
  turns negative where the limit's own control joins $B$, the piece leaves
  before that moment. The conjecture was right for rest 12. R5's split there
  is the wrong seed: its gap starts at the saturation, and Lemma U says the
  gap must start before it.

### P. The arc density jumps when a ridden limit's own control saturates — sketch

- **Statement.** Assume C1 and C2, and a single order-1 limit ridden with
  held set $B$ and free set $F$. Then in the direct form the density of the
  measure on the arc is
  $$\eta = -\frac{\dot a^\top u + \dot d + a_F^\top R_F^{-1} (G^\top H_x)_F}{a_F^\top R_F^{-1} a_F}.$$
  It is continuous while $B$ is fixed. It jumps when a control with
  $a_k \ne 0$ joins $B$, since that control's terms leave both the numerator
  and the denominator.
- **Consequences.**
  - Lemma H ($\eta \ge 0$) can fail right after such a moment, although it
    held before.
  - An exit at that moment needs $\mu = 0$. The jump $\nu a$ would move the
    controls still free, unless $a_F = 0$, so Lemma J allows no jump.
  - So an arc whose $\eta$ changes sign at a saturation must already have
    left before it.
- **Proof.** Differentiate the identity $\dot g(x, u(\lambda_d)) = 0$ along
  the arc, with $u_F = -R_F^{-1}(G^\top \lambda_d)_F$ and
  $\dot\lambda_d = -H_x - \eta \nabla g$. The terms in $\eta$ collect to
  $\eta\, a_F^\top R_F^{-1} a_F$.
- **Gap.** Assumes $u_B$ constant on the stretch, true when it is held at a
  bound, and the direct form of S1.
- **Used by.** Diagnosis of rest 12 (O2). R33: a rise one limit's held window
  induces in another arc's multiplier.

### U. An arc leaves before a change of its held set that turns its density negative — proved relative to P and J

- **Statement.** Assume C1 and C2, and a single order-1 limit $c$ ridden on
  an arc. Let $t_s$ be a time on the arc where a control $k$ with
  $a_k \ne 0$ joins the held set $B$. Suppose the density just after it is
  negative, $\eta(t_s^+) < 0$, as Lemma P's jump allows. Then no extremal
  rides $c$ through $t_s$. An extremal that rides $c$ up to $t_s$ exits
  there only if $\mu(t_s) = 0$. So, generically ($\mu(t_s) > 0$), the arc's
  piece leaves strictly before $t_s$. What follows is a free stretch: either
  a gap closed by a later piece (Lemma E3's moment conditions with
  $a = t_e$), or the rest of the motion.
- **Proof.**
  1. Riding past $t_s$ gives $\eta < 0$ on an interval, which Lemma H
     forbids.
  2. An exit at $t_s$ has no jump in the costate, by J: a jump $\nu a$ would
     move the controls still free, unless $a_F = 0$. With control
     continuity, that needs $\mu(t_s) = 0$ (P's consequence).
  3. Riding up to $t_s$ with $\mu(t_s) > 0$ and leaving there is therefore
     not an extremal. The piece must end earlier, where $\mu$, shifted by the
     entry jump, can vanish.
- **Gap.** Where the piece leaves is not given by the lemma; the exit row
  of the shooting system decides it. Neither is the existence of the
  continuation: that is E3's moment conditions, or the free motion staying
  below the limit.
- **Evidence (rest 12, O2).**
  - In the solved answer, `tau1_max` leaves at 0.41610, and joint 1's jerk
    saturates at 0.42511.
  - On the first route, every rise of $\mu$ starts at joint 1's saturation
    (0.34123 vs 0.34123, and 0.34579 vs 0.34579).
  - The piece trimmed to leave just before it (2% and 5% earlier) solves to
    an exit at 0.34525, with no rise and no violation.
  - Trimmed to leave exactly at the saturation, it converges back to the
    solution that rises.
- **Any control, either direction (2026-11-19).** The proof uses neither that $k$ is
  the limit's own control nor that it joins $B$:
  - step 1 needs only $\eta(t_s^+) < 0$;
  - step 2 needs only control continuity at an exit (J).

  So U holds at any change of the held set, joining or leaving, by a control with
  $a_k \ne 0$, after which the density is negative. P's jump is symmetric: the leaving
  control's terms re-enter the numerator and the denominator.
- **Seen with R43 and E4.**
  - Goal 3 at 80%: `tau4_min`'s rise (η < 0 on [0.5287, 0.5356]) sits in the window
    where joint 2's jerk is held, [0.5274, 0.5423]. Joint 4's own jerk is held only
    near the entry.
  - Goal 3 at 70%: the rise starts where joint 3's jerk leaves its bound (0.571).
  - R29's detection, widened to any such change, then fires at every split (136 in
    goal 3 at 70%, 52 in rest 12). The trimmed pieces settle, but the route still cycles:
    the trim, a closing gap, a merge (E4), a rise again.
  - So U fixes where a piece leaves, but not what continues after it. That is E3's open
    question, with the held set changing at the gap's ends.
  - On the weight's walk (W (3)) the same event recurs as the jerks grow. On rest 12 at
    $k = 100$, R5 split `tau1_max` 92 times on the walk without R29, and 2 times with
    R29's widened detection. The route takes 50.5 s instead of 67.4 s, and both solve.
- **Used by.** R29. Settles O2's conjecture for this case.

### Q. An order-1 touch carries no jump unless the controls absorb it — proved, relative to S1

- **Statement.** Assume C1 to C3 and a time-invariant limit. Let $t_c$ be an
  isolated contact: $g(x(t_c)) = 0$, $g \le 0$ near $t_c$, free arcs on both
  sides, and a costate jump $\lambda^+ = \lambda^- - \nu \nabla g$ with
  $\nu \ge 0$. If $\nu > 0$, then:
  1. $\dot g^- = \dot g^+ = 0$;
  2. the control is continuous at $t_c$;
  3. $u^-$ minimizes $\tfrac12 u^\top R u + (G^\top\lambda^- - \nu a)^\top u$
     over $U$ as well, so the jump is absorbed by the normal cone of $U$
     at $u^-$. With a diagonal $R$, every control with $a_k \ne 0$ is at a
     bound, and the shift $\nu a_k / R_{kk}$ leaves it there (Lemma L's
     degenerate case).

  So, outside that case, a contact of an order-1 limit carries no jump, and
  a multiplier at a point has to be spread over an arc. This is Jacobson,
  Lele & Speyer's "no touch points at order 1" for a strictly convex
  Hamiltonian.
- **Proof.**
  - Since $\nabla g^\top G = a^\top$, adding the jump changes $H$ by a
    multiple of the rate:
    $H(\lambda^+, u) = H(\lambda^-, u) - \nu\,(a^\top u + d) = H(\lambda^-, u) - \nu\,\dot g(u)$.
  - Minimality of $u^+$ gives $H^+ \le H(\lambda^+, u^-) = H^- - \nu \dot g^-$.
    Minimality of $u^-$ gives $H^- \le H(\lambda^-, u^+) = H^+ + \nu \dot g^+$.
  - $H$ is continuous at $t_c$ (time-invariant data, S1's jump conditions).
    So $\nu \dot g^- \le 0 \le \nu \dot g^+$. Feasibility gives
    $\dot g^- \ge 0 \ge \dot g^+$, hence (1).
  - Then $H^- \le H(\lambda^-, u^+) = H^+ = H^-$. So $u^+$ also minimizes
    $H(\lambda^-, \cdot)$, which is strictly convex, and $u^+ = u^-$: (2).
  - (3) is the variational inequality (S5) for both minimizations at the
    same point.
- **Depends on.** S1 (the jump and $H$ continuity), S5.
- **To formalize.** Short once S1 is a hypothesis: two inequalities, and
  uniqueness of the minimizer of a strictly convex function on a convex set.
- **Seen in.** Rest 11's `tau1_max` contact at $t \approx 0.631$. Joints 1–3
  are at their bounds there. Joints 4 and 5 are free, with tiny $a_k$
  (0.015 and −0.005). So the route's solution is *nearly* degenerate. As an
  order-1 touch it solves, with $\nu = 5.3 \times 10^{-6}$, but the rate
  after the jump is $1.2 \times 10^{-4}$, not 0. The extremal is an arc too
  short to solve as one: as an arc at fixed route parameter, the Jacobian's
  condition number is $4 \times 10^5$ to $5 \times 10^{10}$, against
  $4 \times 10^4$ as a touch.
- **Used by.** R20, which carries the touch only as a stand-in on the route
  and gives the exact problem the arc.

### R. A change of held set is continuous, so the STM does not jump — proved

- **Statement.** Assume C1 and the hypotheses of Lemma B on a boundary arc,
  so the riding program (minimize $\tfrac12 u^\top R u + c(z)^\top u$
  subject to $A(z) u + b(z) = 0$ and $u \in U$, a box) has a unique solution
  $u^*(z)$. For a held set $B$, let $u_B(z)$ be the solution of the KKT
  system with the controls in $B$ fixed at their bounds and the rest free,
  with bound multipliers $\nu_B(z)$. Let $t_e$ be a time where control $k$'s
  guard is zero: either $k \notin B$ and $u_{B,k} $ is at a bound, or
  $k \in B$ and $\nu_{B,k} = 0$. Let $B' = B \,\triangle\, \{k\}$. Then:
  1. $u_B(z(t_e)) = u_{B'}(z(t_e))$, so the vector field is continuous at
     $t_e$;
  2. if the guard crosses zero transversally, the saltation matrix is the
     identity, and the STM passes through $t_e$ unchanged.
- **Proof.**
  - Case $k \notin B$ at a bound. The point $u_B$ satisfies stationarity with
    no multiplier on control $k$. Fixing $k$ at that bound, with
    $\nu_k = 0$, it satisfies the $B'$ system too. Lemma B makes that system
    nonsingular, so its solution is unique and equals $u_B$.
  - Case $k \in B$ with $\nu_{B,k} = 0$. Releasing $k$ removes a constraint
    whose multiplier is zero. So $u_B$ satisfies the $B'$ system's
    stationarity, and $k$'s value is its bound. By uniqueness again,
    $u_{B'} = u_B$.
  - The saltation matrix at a transversal crossing of a guard $h$ is
    $I + (f^+ - f^-)\,\nabla h^\top / (\nabla h^\top f^-)$. By (1),
    $f^+ = f^-$, so it is $I$.
- **Depends on.** Lemma B; the saltation formula (S3 for each smooth piece,
  plus the implicit function theorem at the crossing, S2).
- **Gap.** A grazing guard ($\nabla h^\top f^- = 0$) is not covered. The
  integrator then re-probes the guard and, if the held set it computes is
  the same, stays on it; more than 64 changes on one arc fall back to the
  integration without events.
- **To formalize.** Short: uniqueness for a nonsingular linear system, and
  the saltation formula.
- **Used by.** R22. It is a rule of the machinery: the flow integrated is
  the same flow, so no answer is accepted on its account.

### T. Free arcs of integrator chains have a closed form — proved

- **Statement.** Assume the states split into chains $s_0, \dots, s_{L-1}$ with
  $\dot s_m = s_{m+1}$ and $\dot s_{L-1} = u$, one control per chain. Assume
  the running cost is $L(u) = L_0 + \sum_i L_i(u_i)$ with each $L_i$ quadratic
  and $L_i'' = r_i > 0$ constant, the controls lie in a box with constant
  bounds, and there is no penalty. On a free arc, with $\sigma$ the time since
  its start:
  1. $\lambda_m(\sigma) = \sum_{k \le m} \lambda_k(0)\,(-\sigma)^{m-k}/(m-k)!$;
  2. $u(\sigma) = \operatorname{clip}(-(\lambda_{L-1}(\sigma) + c)/r,\ lo,\ hi)$,
     a clipped polynomial of degree $L - 1$. It is polynomial between the real
     roots of $\lambda_{L-1} + c = -r\,hi$ and $\lambda_{L-1} + c = -r\,lo$;
  3. $s_m(\sigma) = \sum_{k \ge m} s_k(0)\,\sigma^{k-m}/(k-m)! + \int_0^\sigma K_m(\sigma - \rho)\,u(\rho)\,d\rho$,
     with $K_m(x) = x^{L-1-m}/(L-1-m)!$. So $L$-point Gauss–Legendre
     quadrature on each piece is exact;
  4. the STM is block diagonal, one $2L \times 2L$ block per chain. Its
     $\partial s / \partial \lambda(0)$ block is
     $\int_0^\sigma K_m(\sigma - \rho)\,\partial u/\partial \lambda_k(0)\,d\rho$,
     with no terms from the moving roots.
- **Proof.**
  - $H = L(u) + \sum \lambda_m s_{m+1} + \lambda_{L-1} u$ has
    $H_{s_0} = 0$ and $H_{s_{m}} = \lambda_{m-1}$, which gives (1).
  - (2) is the minimizer of a strictly convex quadratic over an interval.
  - (3) is the variation-of-constants formula for a nilpotent linear system.
    The integrand has degree at most $2L - 2$ per piece, within the
    $2L - 1$ that $L$ Gauss points integrate exactly (the code uses $L + 1$).
  - For (4), differentiate (3) in $\lambda_k(0)$. The integrand is
    Lipschitz, and $u$ is continuous where it meets a bound, so the
    derivative passes under the integral (dominated convergence).
    $\partial u / \partial \lambda_k(0)$ is zero where $u$ is at a bound and
    $-(-\rho)^{L-1-k}/((L-1-k)!\,r)$ inside, except at the finitely many
    roots.
- **Depends on.** Nothing beyond calculus. The certificate is checked
  symbolically from the problem's own expressions
  (`IntegratorChains.certify`).
- **Gap.** A double root (the control touching a bound without crossing it)
  makes $u$ only Lipschitz there. The flow is still exact, and so is the STM
  away from that measure-zero set of $\lambda(0)$.
- **To formalize.** Short: the closed forms are polynomial identities.
- **Used by.** R23: the free arcs of multiple shooting, and (since page 106) the
  single-shooting seed and every sample the checks take on free arcs and on R60's
  arcs. It is a rule of the machinery: it computes the same flow without
  integration error, so no answer is accepted on its account. On the seed it
  lowers the residual Newton reaches from about $10^{-10}$ to $10^{-14}$, and the
  route takes the same steps.

### T″. An arc riding limits on single chain states is a free arc with those controls at 0 — proved

- **Statement.** Assume T's certificate: integrator chains $s_0' = s_1, \dots,
  s_{L-1}' = u$, with $H$ quadratic in each control and decoupled across them. Let a
  boundary arc ride limits $g_k = a_k s_{j_k, \ell_k} - b_k$, each on one state of
  chain $j_k$, with $a_k \ne 0$ and $a_k, b_k$ free of the states and of time. Then on
  the arc:
  1. each ridden chain's control is $u_{j_k} \equiv 0$;
  2. every other control follows the free law $u = \mathrm{clip}(-(\lambda_{\rm top} +
     c)/r, \mathrm{lo}, \mathrm{hi})$;
  3. the costates satisfy the free arc's equations.

  So the arc's flow, its STM and its cost are T's closed form with the bounds of the
  ridden chains set to $[0, 0]$. Position, rate and acceleration limits of the jerk
  chain all qualify: orders 3, 2 and 1.
- **Proof.**
  - Limit $k$ has order $q_k = L - \ell_k$. Its first $q_k - 1$ derivatives are free
    of $u$, and the last is the top state's: $g_k^{(q_k - 1)} = a_k s_{j_k, L-1}$.
    Its rate is $g_k^{(q_k)} = a_k u_{j_k}$, so in Lemma B's system the limit's row is
    $A_k = a_k e_{j_k}$, with $b_k = 0$. That forces $u_{j_k} = 0$, which lies within
    a box that contains 0.
  - $R$ is diagonal and the rows touch only the ridden controls, so the program
    separates by control. Each free control minimizes its own quadratic over its
    box: the clipped free law.
  - The costates' equations are $\lambda' = -H_x - \nabla_x \sum_k \mu_k
    g_k^{(q_k)}$, and $\nabla_x (a_k u_{j_k}) = 0$. So they are the free arc's
    equations.
  - The closed form with bounds $[0, 0]$ on chain $j$ is the flow with
    $u_j \equiv 0$, and its STM has $\partial u_j / \partial z = 0$, as the riding
    law does. ∎
- **Seen.** On the arm's rate-limited problem, the closed form and the integrated
  riding agree on six single-limit sets from random states:
  - end states to $2 \times 10^{-14}$;
  - STMs to $1.4 \times 10^{-7}$ on entries up to 225, the integrated STM's own
    tolerance, since the closed form's STM is exact.

  Arcs riding only rate limits take 13% of the riding time on goal 3 at 80%.
- **Depends on.** T, B.
- **Used by.** R60.

### W. The secondary weight is a route parameter, and Newton's radius grows with it — proved (1), sketch (2)

- **Statement.** Take a problem solved under the policy `SecondaryCost` with a
  numeric weight $w_* > 0$: running cost $L_0 + w \sum_i (u_i / s_i)^2 / 2$,
  each $u_i$ in a box $[-s_i, s_i]$ and entering $H$ as $S_i u_i$, so that off
  the limits $u_i = \operatorname{clip}(-s_i^2 S_i / w,\ -s_i,\ s_i)$.
  1. *(The route.)* Every problem with $w \in [w_*, k w_*]$ belongs to the
     same family: dynamics, limits and ends are unchanged, and $H$ is strictly
     convex in $u$ with Hessian $\operatorname{diag}(w / s_i^2)$. The shooting
     residual $R(v; w)$ is $C^1$ in $w$ wherever it is $C^1$ in $v$, because
     $w$ enters only the control law and the cost, both smooth in $w > 0$. So
     where $\partial R / \partial v$ is nonsingular, the extremal of a fixed
     structure continues $C^1$ in $w$ (S2). Births and deaths along $w$ are the
     events the route already handles along the offsets, and Lemma N is stated
     for any route parameter. The answer is the exact solve at $w = w_*$,
     judged by the same checks.
  2. *(Why it helps.)* Suppose $S_i$ crosses the interior band
     $|S_i| < w / s_i$ transversally ($\dot S_i \ne 0$) at the ends of a ramp
     of $u_i$ on a free arc. The ramp lasts $2w / (s_i |\dot S_i|)$. On it,
     $\partial u_i / \partial v = -(s_i^2 / w)\, \partial S_i / \partial v$.
     So the flow's first derivative in the unknowns picks up a term of order 1
     from the ramp: duration $O(w)$ times slope $O(1/w)$. Its second
     derivative picks up a term of order $1/w$, from the ramp's ends moving:
     $(s_i^2 / w)\,(\partial S_i / \partial v)^{\otimes 2} / |\dot S_i|$ at each
     end. So the affine-covariant Lipschitz constant $\omega$ of the shooting
     map (Deuflhard) is $O(1/w)$, and the Newton–Kantorovich radius $2/\omega$
     is $\Omega(w)$.
- **Proof.**
  - (1) is the implicit function theorem applied to $R(v; w) = 0$, given S3.
  - For (2), differentiate the variation-of-constants form of the flow twice
    in $v$ (as in Lemma T(3)). The second derivative of
    $\int \operatorname{clip}(a(\rho, v))\,d\rho$ with $a = -s^2 S / w$ is the
    interior integral of $\partial^2 a$, plus $\pm(\partial a / \partial v)^2 / |\partial a / \partial \rho|$
    at the two band crossings. That is $(s^2 / w)\,(\partial S / \partial v)^2 / |\dot S|$.
    Through the dynamics the kernel multiplies this by a bounded factor.
- **Depends on.** S2, S3, Lemma N.
- **Gap.** (2) bounds the terms at the ramps. It does not show that they
  dominate $\omega$; other terms (riding, $M(q)$) do not scale with $w$.
  - The census says they dominate. On rest 4 the route's failed path steps
    are 148 of 400 at $w = 0.01$, 77 of 217 at 0.03 and 15 of 51 at 0.1.
  - Scans along failed Newton steps at $w = 0.01$ show the linearization error
    growing as $\alpha^2$, concentrated in R7's rate rows, which read the
    arriving control.
- **To formalize.** (1) is routine. (2) needs the transversality of the band
  crossings, which holds away from a measure-zero set of $v$ (a double root,
  as in Lemma T's gap).
- **(3) Any path in the plane.** By (1), every point of the plane $(\log w, o)$ with
  $w \in [w_*, k w_*]$ and offsets $o \ge 0$ is a real problem of the family. So the
  route may take any path from $(\log(k w_*), o_0)$ to $(\log w_*, 0)$:
  - R24's corner path tightens the limits at $k w_*$, then lowers $w$ with every limit
    exact;
  - R45's diagonal path lowers $w$ together with the offsets.
- **Seen with L2 (2026-11-19).** The births at a rate peak where the rate's own jerk
  is saturated (L2) pass at a larger weight. On the first route with R43, the ten
  failures of the census page all reach zero offsets at $k = 100$ except goal 3 at 70%,
  against none at $k = 1$. What remains is the path in $w$:
  - on the corner path at $k = 100$, rests 5 and 12 solve;
  - at $k = 30$, goal 3 at 80% solves;
  - on the diagonal at $k = 100$, rests 1, 5 and 12 solve, in 69 to 110 s.

  The corner path's walk in $w$ meets turning points: on goal 4 at 70%, an arc dying
  with the route's parameter decreasing (R2's width steps).
- **Seen with R29 on (2026-11-20).**
  - Larger $k$ on the diagonal reach more answers: at $k = 300$ rests 10 and 6 solve on the
    first route (14.5 s and 33.9 s), where every other path stalls on the walk.
  - The same path loses rest 1, which only the diagonal at $k = 100$ solves, and rest 5.
  - No path in the plane serves every case. Each route of the portfolio is the only
    solver of some case: the second draw at $k = 1$ (goal 5 at 90%, and goal 4 at 70% with
    torques only), $k = 10$ with the second draw (rest 4 with torques only), $k = 100$ on
    the diagonal (rest 1) and $k = 300$ on the diagonal (rests 6 and 10).
  - The walk's failures are spread evenly along it in $\log w$: about 40% of its solves
    at every weight on goal 3 at 80%. So it is eventful throughout, not only near
    $w_*$. Where it stalls varies by case: goal 0 at 70% at $\log_{10} w = -1.86$ of
    $-2$, rest 4 at $-0.56$.
- **Used by.** R24, R45, and the portfolio's weight routes. Soundness rests on (1)
  only: no answer is accepted on (2)'s account, which only explains why the route is
  shorter.

### S. The route's step: the controller's failures and the predictor's order — proved (elementary)

- **Statement.** Along a stretch of the route with one structure, let Newton converge
  from a predictor $p(h)$ exactly when its error $\|p(h) - v(s + h)\| \le \rho$, Newton's
  radius. For a predictor of order $q$ the error is $C h^q$, the secant has $q = 2$ and
  the quadratic through three solutions $q = 3$. Write $h^* = (\rho / C)^{1/q}$ for the
  largest convergent step.
  1. A controller that multiplies the step by $g > 1$ after a success and by $1/2$ after
     a failure, on a stretch where $h^*$ is constant, fails a fraction
     $\ln g / \ln(2g)$ of its attempts in the long run: 37% for $g = 1.5$.
  2. If $\rho$ shrinks like the weight $w$ (W (2)), then $h^* \propto w^{1/q}$. So a
     higher-order predictor's steps shrink more slowly as $w \to w_*$, and the number of
     steps over a walk of the weight falls.
- **Proof.**
  1. $\log h$ moves by $+\ln g$ or $-\ln 2$ and stays within a bounded band about
     $\log h^*$, since it fails above and succeeds below. Over $N$ attempts with $F$
     failures, the net change $(N - F)\ln g - F\ln 2$ is bounded, so
     $F / N \to \ln g / \ln(2g)$.
  2. Solve $C h^q = \rho$ for $h$.
- **Seen (2026-11-20).**
  - On the weight's walk the failures are spread evenly along it, about 40% of the
    solves on goal 3 at 80%: (1)'s equilibrium, not a property of particular points.
  - The walk's steps average about 0.006 decades of $w$, as (2) with a small $\rho$
    predicts.
  - With the quadratic predictor (R48), rest 12 at $k = 100$ takes 33.3 s instead of
    42.3 s, and goal 3 at 80% 76.4 s instead of 84.3 s, with 201 steps instead of 244 and
    73 failures instead of 100.
  - Remembering the last failed step (R46) cuts goal 3 at 80%'s failures from 100 to 82,
    but not its time.
  - *On the census page.*
    - With R48 on every leg: 84 of 90. Rest 12 is gained, and goal 4 at 70% (torques
      only, 14.2 s before, on a route that never moves the weight) is lost.
    - With R48 only on the legs that move the weight, where (2) applies: 85 of 90,
      with nothing lost and the common solves 7% faster.
  - *Where the hypothesis fails (2026-11-20).* (1) assumes Newton converges exactly
    below a radius. On goal 3 at 80%, most steps with $\Theta \ge 2$ have a first
    iterate outside the residual's domain, or across a held-set kink (Lemma AC), at
    particular points of the route. A controller built on (1), bisecting between the
    last success and the least failure (R59), took 133 s against 47 s: after such a
    point its bracket keeps the steps short. So (1) describes the controller's
    equilibrium, not where the steps are limited.
- **Used by.** R46, R48.

### AC. Newton's progress is measured in its own norm, not the residual's — proved

- **Statement.** Let $\Psi : \mathbb{R}^n \to \mathbb{R}^n$ be $C^1$ near $v_0$ with
  $J = D\Psi(v_0)$ nonsingular. Let $\Delta v = -J^{-1}\Psi(v_0)$,
  $v_1 = v_0 + \Delta v$, and let $\bar\Delta v = -J^{-1}\Psi(v_1)$ be the simplified
  correction. Write $\Theta = \|\bar\Delta v\| / \|\Delta v\|$ for the contraction.
  1. *(Invariance.)* For every nonsingular $A$, the system $A\Psi = 0$ has the same
     Newton iterates, the same $\Delta v$, $\bar\Delta v$ and $\Theta$. The residual
     ratio $\|A\Psi(v_1)\| / \|A\Psi(v_0)\|$ can take any value in
     $[\Theta/\kappa,\ \kappa\,\Theta]$ with $\kappa = \operatorname{cond}(J)$.
     That is, $$\frac{\|\Psi(v_1)\|}{\|\Psi(v_0)\|} \le \kappa(J)\,\Theta .$$
  2. *(Convergence.)* If $\|J^{-1}(D\Psi(v) - D\Psi(w))\| \le \omega\|v - w\|$ near
     $v_0$ (affine-covariant Lipschitz), then $\Theta \le \tfrac12\omega\|\Delta v\|$.
     If $h_0 = \omega\|\Delta v\| < 2$, the chord (simplified Newton) iteration from
     $v_0$ converges to the solution, with each simplified correction contracted by at
     least $\Theta$ to first order. So $\Theta < \tfrac12$ is a sufficient local test
     of progress, and a test on $\|\Psi\|$ is not: by (1) it can reject a step with
     $\Theta$ small whenever $\kappa(J)\Theta \ge \tfrac12$.
  3. *(Budget.)* The chord iteration keeps $J$, so it contracts linearly, by about
     $\Theta_k$ at step $k$. To go from a residual $e_0$ to the tolerance $\tau$
     within $N$ iterations it needs $\Theta_k \lesssim (\tau/e_0)^{1/N}$. With
     $N = 12$, $e_0 = 10^{-2}$ and $\tau = 10^{-10}$, that is about $0.2$. So a step
     accepted with $\Theta_k > \tfrac14$ is followed by a fresh Jacobian (Newton's
     quadratic rate), as Deuflhard's $\Theta_{\max} = \tfrac14$ prescribes.
  4. *(A carried Jacobian.)* Let $\tilde J = D\tilde\Psi(\tilde v)$ be the Jacobian at
     the last solution $\tilde v$ of a nearby problem $\tilde\Psi$ (the route's
     parameter $\tilde s$, the same structure). The chord with $\tilde J$ for $\Psi$
     (parameter $s$) contracts its corrections by at most about
     $$\Theta \lesssim \|\tilde J^{-1}(D\Psi(v) - \tilde J)\| \le
       \omega\,\|v - \tilde v\| + \tilde\omega\,|s - \tilde s| ,$$
     with $\tilde\omega$ the affine-covariant Lipschitz constant of $D\Psi$ in $s$.
     Along a path both terms are $O(ds)$. So a carried Jacobian serves while
     $\omega\|v - \tilde v\| + \tilde\omega\,ds \lesssim \tfrac14$, and (3)'s test,
     applied to the measured $\Theta$, refreshes it exactly when it no longer does.
     The answer is unchanged: the chord still stops at the tolerance in the residual.
  5. *(No admissible damping.)* Under (2)'s model, the damped step $v_0 + \lambda\Delta v$
     has simplified correction at most $(1 - \lambda + \tfrac12\lambda^2 h_0)\|\Delta v\|$,
     $h_0 = \omega\|\Delta v\|$, and the full step's contraction is
     $\Theta \le \tfrac12 h_0$. Deuflhard's restricted monotonicity test
     $\Theta(\lambda) \le 1 - \lambda/4$ can then hold only for
     $\lambda \le 3/(2h_0) \le 3/(4\Theta)$. The chord's dampings are $1/2, \dots, 1/16$,
     so for $\Theta > 12$ none of them is admissible: its backtracking cannot make
     progress that the model certifies, and the solve fails as it would, with fewer
     Jacobians.
- **Proof.**
  - (1) $D(A\Psi) = AJ$, so $(AJ)^{-1}A\Psi = J^{-1}\Psi$ at every point. The
    corrections and $\Theta$ do not depend on $A$.
  - $\|\Psi(v_1)\| = \|J\bar\Delta v\| \le \|J\|\,\Theta\,\|\Delta v\|$ and
    $\|\Delta v\| \le \|J^{-1}\|\,\|\Psi(v_0)\|$, which gives the bound. Choosing $A$ to
    rotate $\Psi(v_0)$ and $\Psi(v_1)$ onto $J$'s extreme singular directions attains
    the ends of the range.
  - (2) $\Psi(v_1) = \int_0^1 (D\Psi(v_0 + \tau\Delta v) - J)\,\Delta v\,d\tau$, so
    $\|J^{-1}\Psi(v_1)\| \le \int_0^1 \omega\tau\|\Delta v\|^2 d\tau =
    \tfrac12\omega\|\Delta v\|^2$.
  - The convergence statement is the affine-covariant Newton–Mysovskikh theorem for
    the simplified iteration (Deuflhard 2004, Thms 2.5 and 2.10).
  - (3): $N$ linear contractions by at most $\Theta$ take $e_0$ to $\Theta^N e_0$ in
    the natural norm.
  - (4): the chord's error map is $e \mapsto e - \tilde J^{-1}(\Psi(v^* + e) - \Psi(v^*))
    = \tilde J^{-1}\int_0^1 (\tilde J - D\Psi(v^* + \tau e))\,d\tau\; e$, whose norm is at
    most $\sup \|\tilde J^{-1}(D\Psi - \tilde J)\|$ on the segment; split
    $D\Psi(v; s) - D\tilde\Psi(\tilde v; \tilde s)$ through $(\tilde v; s)$.
  - (5): $\Psi(v_0 + \lambda\Delta v) = (1 - \lambda)\Psi(v_0)
    + \int_0^1 (D\Psi(v_0 + \tau\lambda\Delta v) - J)\,\lambda\Delta v\,d\tau$, so the
    simplified correction $-J^{-1}\Psi(v_0 + \lambda\Delta v)$ has norm at most
    $(1 - \lambda)\|\Delta v\| + \tfrac12\omega\lambda^2\|\Delta v\|^2$. The bound meets the
    test exactly when $\lambda \le 3/(2h_0)$, and $h_0 \ge 2\Theta$ by (2). The model
    certifies no damping beyond that; a test passed there is passed by an iterate the
    model does not vouch for. ∎
- **What it means for the route.** Near an L2 touch the path Jacobian is
  ill-conditioned: equilibrated $\sigma_{\min}$ of about $5 \times 10^{-4}$, so
  $\kappa$ is $10^3$ to $10^4$. The chord's old test, $\|\Psi(v_1)\| < \tfrac12
  \|\Psi(v_0)\|$, then rejects steps that Newton takes well.
- **Seen (2026-11-20).** Goal 3 at 80%, $k = 100$, main at 30d6b32:
  - Of the 35 path solves handed to MINPACK, 27 had no held-set change between the
    predictor and the first iterate. Their median $\Theta$ was 0.04 (range
    $4 \times 10^{-4}$ to 2.7), yet the chord's residual did not halve. A typical
    case ended its first step with the residual unchanged at $5 \times 10^{-3}$, then
    backtracked four times, then took 150 to 490 MINPACK residuals.
  - Those 35 solves cost 17.9 s of the route's 42.5 s of path solves.
- **The other causes of failed steps: the domain, and kinks.** The path residual is
  defined only where every riding arc's limits can be ridden within the control
  bounds (the holdable set, Lemma I). Inside that set it is only piecewise smooth: a
  riding arc's held set changes at events, and Lemma R keeps the STM continuous but
  not its derivative in $v$. On goal 3 at 80% (main), 105 chord solves had
  $\Theta \ge 2$:
  - in 76, the first Newton iterate is outside the domain, so the residual is not
    defined there (a riding fails to propagate);
  - in 23, it changes a riding arc's count of held-set events;
  - in 6, neither.
  (2)'s hypothesis fails in both of the first two cases, and AC claims nothing there.
  A large $\Theta$ there is the domain's edge or a kink crossed, not distance
  (R37's failure, `structure_certificates.md`).
- **Seen: (3).** In R55's first version, without the fresh Jacobian, a settle solve
  on rest 10 (route 7) used its twelve iterations at $\Theta \approx 0.4$ and stopped
  at $6 \times 10^{-7}$. The residual test had refreshed $J$ there and reached
  $6 \times 10^{-10}$.
- **Depends on.** Nothing of the problem's; the standard affine-covariant Newton
  theory (Deuflhard 2004, ch. 2).
- **To formalize.** (1) is linear algebra. (2) needs the integral form of the mean
  value theorem, which Mathlib has, and the Newton–Mysovskikh argument.
- **Used by.** R55; R61, through (4); R63, through (5); R37's $\Theta$ (sec. 5.1 of the same reference).

### V1–V5. The route as a one-parameter family

The arguments are in `route_family.md`. A *leg* is a segment
$\theta(s)$ of the route parameters (offsets, weight, smoothing), fixed when
it starts; $S$ is the structure, $\Psi_S(v, s) = 0$ the square shooting
system, and a *monitor* is a scalar that must stay $\ge 0$ (peak, width,
jump, density, gap, order, bound, end, held sign).

- **V1. Acceptance is the monitors — proved, relative to S1.** At $s = 1$,
  `_verdict` accepts exactly when every monitor is $\ge 0$. Depends on H.
  - *The resolution of the sampled monitors (update, 2026-11-20).* The peak and density
    monitors are functionals of one extremal, read at the samples of one integration
    of it (801 points over $[0, T]$, by length on each segment). Between samples spaced
    $h$, a limit with $|\ddot g| \le M$ exceeds its larger sampled value by at most
    $M h^2/8$. With the rate $\dot g_i$ at a sample, the bound
    $g \le g_i + \max(\dot g_i, 0)^2/(2M)$ holds up to the next sample.
    - For a rate limit on the jerk chain, $\ddot g = u_j$, so $M$ is the jerk bound
      itself (30 on the census).
    - With $T \approx 1.3$ and 801 samples, $M h^2/8 \approx 10^{-5}$. The check's
      floor is $10^{-6}$ of a rate limit of 0.3 to 2. So the sampled check resolves
      violations to about 5 to 30 times its floor, not to the floor.
    - This bounds what V1 can claim from samples. Using the second bound would
      certify the floor without more samples; it is not done yet.
- **V2. Continuation with a fixed structure — proved.** With $D_v\Psi_S$
  nonsingular and the implicit events transversal, there is a unique $C^1$
  branch; the monitors are continuous along it, and wherever they are all
  positive the branch is a curve of accepted extremals. Depends on S2, S3,
  S7, R, N.
- **V3. Events are isolated, one at a time — sketch.** Under a
  transversality hypothesis (T), for almost every draw of the offsets each
  event is one monitor crossing zero transversally, no two coincide, and
  folds are simple. Depends on S8. *Gap:* (T) is checked only for births
  (Lemma D's mechanism).
- **V4. The event table for order 1 — partly proved.** Birth (E1) and death
  (E2) by Lemma E; fold (E8) by S9; split (E3) and merge (E4) open (outside
  Bonnans–Hermant's strict complementarity); reorder (E5) a sketch from B
  and J; an arc reaching an end (E6) open; constraint regularity failing,
  the corner and saturation cases (E7), open.
- **V5. A leg ends at $s = 1$, or fails visibly — sketch.** From a regular
  start, if every event met has a proved continuation and the curve stays
  bounded, it reaches $s = 1$ after finitely many events. Depends on V2 to
  V4, S8. *Gaps:* the table (V4) and boundedness, which must be detected.
- **V6. A birth is predicted by the secant of its margin — proved, relative to V2.**
  On a branch of one structure (V2), let $m_c(s) = \max_{t \notin A_c} g_c(t; s)$ be the
  margin of a limit $c$ not in the structure ($A_c$: its arcs, here none), in its scale.
  1. $m_c$ is Lipschitz in $s$ (a maximum of a $C^1$ family over a compact set), and
     $C^1$ near an $s$ where the maximizer is unique and nondegenerate, with
     $m_c' = \partial_s g_c(t^*(s); s)$ (Danskin; the envelope theorem).
  2. With accepted points $s_0 < s_1$ and $m_c(s_1) < 0 < \sigma = (m_c(s_1) -
     m_c(s_0))/(s_1 - s_0)$, the secant's zero $\hat s = s_1 - m_c(s_1)/\sigma$ satisfies
     $m_c(\hat s) = O(\sup|m_c''| (\hat s - s_0)(\hat s - s_1))$, so a step to
     $\hat s + \delta/\sigma$ ends with $m_c \approx \delta$ to that order.
  - *Proof.* (1) is Danskin's theorem for $\max_t g_c(t; v(s))$, with $v$ the $C^1$
    branch of V2 and $g_c$ smooth in $(t, z)$. (2) is the interpolation error of the
    linear interpolant through $s_0, s_1$, extended to $\hat s$. ∎
  - *Use (R64).* $\delta$ is R16's target, $0.2$ of `locate_tol`: the step ends inside
    the window in which the settle brings a new limit in (V1's monitor), instead of
    beyond it, where the step is rejected and R16 locates the birth after the fact. If
    the peak jumps (a second local maximum overtakes the first, where (1)'s
    hypothesis fails) the prediction is wrong and the step fails as before.
- **V2′. Correction: a monitor at a fixed time is continuous only away from held-set
  sweeps — proved, relative to P.** V2's monitors are continuous along the branch as
  functionals: the density as a function on the arc, not at one instant. Fix a time
  $t^*$ in an order-1 arc. Its held set at $t^*$ changes at the value $s_h$ where a
  control's saturation boundary $\tau(s)$ passes $t^*$, and $\tau$ is $C^1$ with
  $\dot\tau \ne 0$ by V2's transversal events. By P, $\eta(t^*; s)$ jumps at $s_h$
  by P's jump. So $\eta(t^*; \cdot)$ is $C^1$ on each side of $s_h$, and not across it.
  - *Consequence for R25.* R25 places the step where the secant puts
    $\eta(t^*) = 0.1\,\eta_1(t^*)$. Where $\eta(t^*)$ is continuous, the shorter
    step's end value is about a tenth of the last. Where it jumps at the base
    ($s_h = s_0$), the end value does not shrink, so no shorter step reaches a zero,
    and the locations go on shortening the step forever.
  - *Seen (2026-11-20).* Goal 3 at 80%, $k = 100$, with R51 on: at $s = 0.9398$ of
    the weight's walk, the `tau5_max` arc's density at $t = 0.78501$ "falls from
    3.83 to $-0.95$" in each of 15 consecutive steps, as the step falls by 0.82 each
    time from $1.4 \times 10^{-3}$ to $10^{-4}$. The route then stalls.
  - *Rule R54.* A second location at the same limit and time whose end value is not
    smaller by half is a jump at the base: no more locations, and R5 splits from
    there. With it, the same route solves (63 s).
- **Used by.** R25 (V2, with V2′: the density at a fixed time is $C^1$ in the route
  parameter away from held-set sweeps), R54 (V2′). Phase 2's follower (planned). Today, R1 (V3), R2, R10, R16,
  R17 (E1, E2), R5, R18, R21, R25 (E3) and R11 (E4) are pieces of it, and
  R30 and R31 watch V1's order and jump monitors on the route.

### Observation O4: what the parallel portfolio costs (2026-11-20)

- **Speed per process.** On homebox (8 cores, a laptop i9-11980HK), a CPU-bound process
  runs at 1.0, 1.0, 0.74, 0.56 and 0.51 of its single speed with 1, 2, 4, 7 and 8 busy.
  Seven copies of goal 3 at 80% at $k = 100$ take 106 s each, against 56 s alone. The
  cause is the hardware: with OpenBLAS held to one thread the times are the same.
- **The page's routes.** From the route record (`Diagnostics.routes`), on page 96:
  - Almost no route ends before the winner: on most cases all seven run until one finds
    the answer, at about half speed.
  - The winners are spread over all seven routes: 18, 16, 9, 9, 16, 10 and 7 cases.
  - So no route can be dropped without losing cases.
  - A case that a route solves in $T$ alone takes about $1.9\,T$ on the page: the 60 s
    cap is about 32 s of one core.
- **Consequence.** Goal 3 at 80% (56 s alone on its one solving route) needs a route
  about 1.8 times faster, not a different schedule.

### Observation O5: where the path rules act (2026-11-21)

The path-level rules now sit behind one interface (`pipeline.py`): a rule names the hook
it acts at and answers `act(ctx)` from a context of that hook. The route runs each hook's
active rules in order (the first answer wins, or for the predictor, all guesses are
collected). The hook map in `pipeline.py` lists each hook, its rules and the ones still
inline.

| Hook | Rules |
| --- | --- |
| predict | R48 quadratic, secant, last solution |
| step failed: size | R16 located birth, R37 contraction cut, R46 halve |
| step accepted: size | R46 and R37 growth |
| settle start, settle | R25, R30/R38/R39 order, R44 fold, history, violation, dead arc (R18), bounds |
| step failed: split, rescue | R25 split, arcs to the ends, R17 deaths, R2 widths, R39 tie, fold turn |
| exact rejected | R20 collapsed touches, exact splits and births |

- **Behaviour.** The verbose route logs of rest 4, goal 9 at 70% and vgoals 1 and 3 at 80%
  are identical before and after, step for step. Page 105 solves the same 84 cases
  in 690 s against 689 s.
- **Cost.** The path rules act once per step or settle round: a few hundred times a solve.
  The rules that act once per residual evaluation (R7, R10b, R22, R23, R39's ties, R43,
  R55, R60) act thousands of times, so they stay in the shooting code, which is the part
  to compile next.

### Observation O6: where a solve's time goes once the free arcs are closed (2026-11-21)

Measured on one route, warm, with every kernel call timed (vgoal 1 at 80%, 2.1 s; vgoal 3
at 80%, 26.5 s):

| Part | vgoal 1 @80 | vgoal 3 @80 |
| --- | --- | --- |
| Riding with the STM (Jacobians) | 29% | 45% |
| Riding without it (residuals) | 12% | 13% |
| Closed-form chains, other kernels | 28% | 22% |
| Python and NumPy | 31% | 20% |

- **A riding STM evaluation.** About 23 µs against 1.9 µs without the STM. In C
  (the generated code, timed at recorded states): the Lagrangian Hessian
  $\nabla^2(\mu \cdot g^{(q)})$ of the torque limits takes 9.6 µs, the rest of the riding
  Jacobian 5.6 µs, $A\Phi$ (508 of 1296 entries nonzero) about 4 µs, the integrator about
  3 µs.
- **Tried: holding the Hessian over a window.** With the STM out of step control
  (its tolerance changes nothing for the exact Hessian: the steps follow $z$), a
  Hessian held over a window of 0.03 makes the STM calls 1.7 times faster with
  relative STM errors near $10^{-4}$, and $z$ is unchanged. But the routes then lose
  vgoal 1 at 80% (window 0.01) and vgoal 3 at 80% (0.03, and once per call). The
  Jacobian steers more than Newton: the path's tangent, the predictor, the step
  control by $\Theta$ and the location of births all read it. Not adopted.
- **Tried: per-root Hessians** (Lemma SP2 extended to $L_{xx}$). The page was 0.4%
  faster, the build took 52 s instead of 33. The six torques share one recursion,
  so one torque's Hessian costs nearly all of it. Not adopted.
- **Consequence.** The Hessian is exact or absent. What remains is the Python glue,
  and fewer Jacobians.
- **Compiled since (page 107).** With identical solutions (bit for bit, on the cases
  timed): R7's corner test and the regular-junction test in C; first-order tangency for
  residuals (no Hessians); the chains' law held as one compiled object; control bounds
  and boundary-arc samples batched (FNI's stepper from C, the same steps); the checks
  reuse the last residual's arcs, which 91% of their calls hit. Single routes are 10–17%
  faster. The Python left inside evaluations is then 9–14% of a solve: a fully compiled
  evaluator is worth at most that, so the next work is on the number of Jacobians.

### Observation O3: what the direct comparison shows for the unsolved cases

The direct transcription (`auto_optimal_control.direct`) was run at 120 and 240
intervals. It is a comparison only, never a seed. Final times agree to about
1e-4 between the grids, and the total multiplier of each limit agrees to a few
percent.

- **Rest 12** has two base-joint arcs: `tau1_max` on [0.111, 0.410] and
  `tau1_min` on [1.008, 1.482]. Joint 1's own jerk is interior on both. It
  leaves its bound just before each entry and reaches it again just after each
  exit, as Lemma P requires.
- **Rest 1 and rest 11** each have an isolated `tau1_max` contact. Its
  multiplier, 0.0016 and 0.0011, is the same on both grids, so it is an atom of
  the measure: an order-1 touch, or an arc shorter than the grid.
  - At the contact every jerk that moves `tau1` is at a bound **except one**
    (joint 2, in the middle of a switch).
  - So Lemma L's hypothesis (every control with $a_k \ne 0$ saturated after
    the touch) fails. The jump moves the interior control, and the touch rows
    are not singular.
  - R12 was refuted on the drift toy, where the single control is saturated.
    It has not been tried on the arm.
  - Lemma Q settles which it is: with a strictly convex Hamiltonian it is an
    arc. R20 carries it as a touch on the route only, and gives the exact
    problem the arc. Rest 11 and rest 12 solve with it.
- **Rest 4's contact is not settled.** Its multiplier halves from 120 to 240
  intervals.
- **Goal 9 at 80%** has a short `tau6_min` arc, [0.282, 0.287]. Its multiplier
  is stable, and joint 6's jerk switches at its start.
- **Reading the solver's logs.** The route's last structures in a failed solve
  belong to the time-reversed problem (R4). Map them back by $t \to t_f - t$
  before comparing.
