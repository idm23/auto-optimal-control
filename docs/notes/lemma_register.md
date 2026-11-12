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
| S1 | The maximum principle with state constraints, in the direct-adjoining form, with a nonnegative measure (Hartl, Sethi & Vickson 1995, Thm 4.1), plus its relation to the indirect (BDD) form | H, J, K |
| S2 | The implicit function theorem, and Hadamard's lemma (a smooth function that vanishes on $w = 0$ is $w$ times a smooth function) | E |
| S3 | Smooth dependence of ODE solutions on initial values, times and parameters, including backward integration | E, R10 |
| S4 | Sard's theorem, or the weaker fact that a $C^1$ image of a lower-dimensional manifold is Lebesgue-null | D |
| S5 | Variational inequalities for convex minimization over a convex set | J, I′ |
| S6 | The kinetic energy is $\tfrac12 \dot q^\top M(q) \dot q$ for the Newton-Euler model | A |
| S7 | Danskin's theorem: directional derivatives of $\max_{t \in T} \phi(t, s)$ over a compact $T$ | N |

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
- **Used by.** R2, R10.

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
- **Used by.** Acceptance (the $\dot\mu \le 0$ check), R5.

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
- **Open: a lemma for the held set changing on an arc.** It should say what
  Lemma H's $\dot\mu \le 0$ requires at the moment $B$ gains the joint of a
  ridden limit.
  - *Conjecture.* Generically the arc exits before that moment. If so, the
    split R5 builds, with its gap from the saturation time onward, is the
    right structure, and its failure to converge is a seeding problem.

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
- **Used by.** Diagnosis of rest 12 (O2). No rule yet.
