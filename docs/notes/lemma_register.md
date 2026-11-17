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
  stretch with $g < 0$ is a birth, not a widening).

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
- **Merge (E4).** E3 read backwards.
- **Gap.** The held set must not change at the gap's ends in a way the
  released motion does not share. Rest 12 (O2), where a ridden limit's own
  jerk saturates, still cycles: it splits, the gap closes, it splits again.
- **Used by.** R27.

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
- **Used by.** Diagnosis of rest 12 (O2). No rule yet.

### U. An arc leaves before its own control saturates — proved relative to P and J

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
- **Used by.** R23. It is a rule of the machinery: it computes the same flow
  without integration error, so no answer is accepted on its account.

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
- **Used by.** R24. Soundness rests on (1) only: no answer is accepted on (2)'s
  account, which only explains why the route is shorter.

### V1–V5. The route as a one-parameter family

The arguments are in `route_family.md`. A *leg* is a segment
$\theta(s)$ of the route parameters (offsets, weight, smoothing), fixed when
it starts; $S$ is the structure, $\Psi_S(v, s) = 0$ the square shooting
system, and a *monitor* is a scalar that must stay $\ge 0$ (peak, width,
jump, density, gap, order, bound, end, held sign).

- **V1. Acceptance is the monitors — proved, relative to S1.** At $s = 1$,
  `_verdict` accepts exactly when every monitor is $\ge 0$. Depends on H.
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
- **Used by.** R25 (V2: the density at a fixed time is $C^1$ in the route
  parameter). Phase 2's follower (planned). Today, R1 (V3), R2, R10, R16,
  R17 (E1, E2), R5, R18, R21, R25 (E3) and R11 (E4) are pieces of it, and
  R30 and R31 watch V1's order and jump monitors on the route.

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
