# Certificates and the rules they license

*Working note, 9 November 2026. Draft for review; nothing here is implemented yet.*

## Why this note

A user of this library writes their problem as algebra: SymPy expressions
for the dynamics, the cost and each limit. From them we derive every function
the solver uses. A general-purpose solver sees none of that; we see all of
it. Many properties that decide whether a continuation step is safe follow
from the *form* of those expressions: which variables appear linearly, which
matrices are positive definite, what the first time derivative of a limit
contains. Those can be established once, per problem, before solving.

The plan is to turn search heuristics into rules with three parts:

1. **Hypotheses**: properties of the problem's expressions, each with a stated
   way to verify it (below, *exact*, *by construction* or *local*).
2. **A proof** that the hypotheses make the rule correct.
3. **The rule**: what the solver does, enabled only when its hypotheses are
   verified for the problem at hand. Otherwise the solver behaves as today
   and reports which hypothesis was missing.

The benchmark then measures how much a rule helps. It does not justify the rule.

This note does the first instance, for the class that the six-joint arm
belongs to, and is explicit about which steps are proved here, which rest on
cited results, and which are still to be done.

**Kinds of verification.**

- *Exact*: the CAS proves the property from the expressions (an identity,
  a constant matrix with known eigenvalues, a derivative that vanishes).
- *By construction*: the property follows from how we built the expression,
  by a lemma proved here, plus a check of the inputs (for example, positive
  inertias in a link table).
- *Local*: the property is a condition at one point of the solve (an event)
  and is checked there with a rigorous error bound, never by the sign of an
  unguarded floating-point number.

Some properties cannot be decided automatically: positivity of a general
expression is undecidable. Then the answer is *unknown* and the rules that
need the property stay off.

## The class

States $x = (q, w, a) \in \mathbb{R}^{3n}$ (joint angles, rates and
accelerations), control the jerk $j \in \mathbb{R}^n$:

$$\dot q = w, \qquad \dot w = a, \qquad \dot a = j, \qquad |j_i| \le J_i .$$

Cost $t_f + \varepsilon \int_0^{t_f} \tfrac12 \sum_i (j_i / J_i)^2\,dt$ with
$\varepsilon > 0$ (the default policy's secondary cost). Torque limits

$$g_i^{\pm}(x) = \pm \tau_i(x) - \bar\tau_i \le 0, \qquad
\tau(x) = M(q)\,a + h(q, w),$$

with $\tau$ from the recursive Newton-Euler algorithm on a link table. The
active-set route relaxes each limit to $g_c(x) \le o_c$ and drives the offsets
$o \in \mathbb{R}^{N}$ ($N = 2n$ limits) to zero along a path $o(s)$,
$s \in [0, 1]$, $o(1) = 0$.

Notation: $H = L + \lambda^\top f$ is the Hamiltonian, $R = \partial^2 H /
\partial j^2$. While a set $S$ of limits is ridden and a set $B$ of jerks is
held at their bounds, the riding control and multipliers solve the KKT system
of the run-time rider (`riding.py`):

$$K \begin{pmatrix} j \\ \mu \\ \nu \end{pmatrix} = r(x, \lambda), \qquad
K = \begin{pmatrix} R & C^\top \\ C & 0 \end{pmatrix}, \qquad
C = \begin{pmatrix} \sigma_S\, \partial \dot g_S / \partial j \\ E_B \end{pmatrix},$$

where $E_B$ holds the unit rows of the held jerks and $\sigma_S$ the signs.

## Certificates for this class

| | Property | Verification |
|---|---|---|
| C1 | The dynamics are affine in the control, $f = F x + G j$, with $G$ constant and of full column rank | exact: $\partial^2 f / \partial j^2 = 0$, $\partial f / \partial j$ constant, rank check |
| C2 | $R = \partial^2 H / \partial j^2$ is constant and positive definite | exact: here $R = \varepsilon\, \mathrm{diag}(1/J_i^2)$ |
| C3 | Each torque limit has order 1, with $\partial \dot g_i^{\pm} / \partial j = \pm M(q)_{i,:}$ | by construction: $\tau$ is affine in $a$ with coefficient $M(q)$, so $\dot\tau = M(q)\, j + (\text{terms free of } j)$ |
| C4 | $M(q)$ is symmetric positive definite for every $q$ | by construction: Lemma A, given every link's principal moments of inertia are positive (true for `SIX_JOINT_LINKS`: all are positive rationals) |
| C5 | Away from jerk saturation, every function is smooth | exact: the only nonsmooth operations are the clips of the control law to $[-J_i, J_i]$ |
| C6 | Time reversal maps the problem to itself (Lemma F) | exact for the dynamics and cost; for Newton-Euler torques, by construction |
| C7 | Cyclic states: in no dynamics, cost or limit (Lemma G) | exact: free symbols |

All seven are computed by `auto_optimal_control.certificates.certify`. On the
six-joint arm this takes about a second.

## Lemmas proved here

### Lemma A (positive definite mass matrix)

*If every link's inertia tensor about its centre of mass is positive definite,
then $M(q) \succ 0$ for every $q$.*

**Proof.** $M(q)$ is the matrix of the kinetic energy,
$T = \tfrac12 \dot q^\top M(q) \dot q = \tfrac12 \sum_k \big( m_k |v_k|^2 +
\omega_k^\top I_k \omega_k \big)$, where $v_k$ and $\omega_k$ are the velocity
of link $k$'s centre of mass and its angular velocity (this is what the
recursive Newton-Euler algorithm differentiates: the coefficient of $\ddot q$
in the Euler-Lagrange equations is $\partial^2 T / \partial \dot q^2$). Take
$\dot q \ne 0$ and let $i$ be the smallest index with $\dot q_i \ne 0$. The
angular velocity of link $i$ is $\omega_i = \sum_{l \le i} \dot q_l\, z_{l-1}
= \dot q_i\, z_{i-1}$, since the earlier rates are zero, and $z_{i-1}$ is a unit
vector. So $\omega_i \ne 0$ and $T \ge \tfrac12 \omega_i^\top I_i \omega_i > 0$.
Hence $\dot q^\top M(q) \dot q > 0$ for every $\dot q \ne 0$. $\square$

Masses play no part; only the inertias do, and only their positive
definiteness.

### Lemma B (riding is never singular unless a joint's own jerk is held)

*Assume C2, C3 and C4. Let $S$ be a set of torque limits with at most one per
joint, and $B$ a set of jerks held at bounds, with no joint in both
($S \cap B = \emptyset$, identifying a limit with its joint). Then $K$ is
nonsingular, and the riding control and multipliers are unique and
$C^\infty$ functions of $(x, \lambda)$ wherever $S$ and $B$ stay fixed.*

**Proof.** First, $C$ has full row rank. Suppose $\alpha^\top \sigma_S
M(q)_{S,:} + \beta^\top E_B = 0$. The columns indexed by $S$ are zero in $E_B$
because $S \cap B = \emptyset$, so restricting to them gives $\alpha^\top
\sigma_S M(q)_{S,S} = 0$. $M(q)_{S,S}$ is a principal submatrix of a positive
definite matrix, so it is positive definite and $\alpha = 0$. Then $\beta^\top
E_B = 0$ with distinct unit rows gives $\beta = 0$.

Next, $K$ is nonsingular. If $K (x, y) = 0$ then $R x + C^\top y = 0$ and
$C x = 0$, so $x^\top R x = -(C x)^\top y = 0$, and $R \succ 0$ gives $x = 0$.
Then $C^\top y = 0$ and full row rank gives $y = 0$.

Finally $K$ and the right-hand side are smooth in $(x, \lambda)$ (C5 holds on
the arc, since the held jerks are constants and the free ones are interior),
so by Cramer's rule the solution is smooth. $\square$

Both limits of one joint cannot be active together (they differ only by sign,
and $\bar\tau_i > 0$), so "at most one per joint" is no restriction.

**What it rules out, and what it does not.** For this class, riding any
combination of torque limits on different joints never has singular
equations. When $S \cap B \ne \emptyset$ (a joint rides its torque limit
while its own jerk is at a bound) $C$ may lose rank; Lemma B says nothing
there, and that case needs a local check.

*Correction.* An earlier version concluded that a stall of the route is
never caused by the riding system. That does not follow. Lemma B is about
the equations once $S$ and $B$ are fixed; it says nothing about whether some
admissible $B$ exists, that is, whether the limit can be held at all within
the jerk bounds. Lemma I treats that question, and the rest-to-rest cases
show it is what stalls them.

### Lemma C (the present continuation path makes births coincide)

*The route starts from the solution $z_0$ of the problem with no limit
active, with $o_c(0) = p_c + \epsilon_c$, where $p_c = \max_t g_c(z_0(t))$ and
$\epsilon_c = 10^{-3}\,\mathrm{scale}_c$, and moves along $o(s) = (1 - s)\,
o(0)$. Every limit with $p_c > 0$ first comes into play at
$s_c = \epsilon_c / (p_c + \epsilon_c)$, as long as no other limit has come
into play before it.*

**Proof.** While no limit is active, the extremal does not depend on the
offsets, so it stays $z_0$. Limit $c$ comes into play when its offset falls to
the peak of $g_c$ along $z_0$: $(1 - s)(p_c + \epsilon_c) = p_c$. $\square$

So all the births happen in $s \in (0, \max_c \epsilon_c / p_c]$, an interval
of order $10^{-3}$. After the first birth the extremal moves, but only by the
effect of an arc born with zero width, so the others stay crowded into the
same short interval. This is the most degenerate path available: every event
at once, at the start. It matches the census, where failing cases stall at
$s \approx 3 \times 10^{-4}$ with several arcs of width around $10^{-5}$.

### Lemma D (random offsets separate events)

*Suppose the set $\mathcal{E}_2 \subset \mathbb{R}^N$ of offsets at which two
or more structure changes coincide is a finite union of $C^1$ submanifolds of
dimension at most $N - 2$. If $o(0)$ is drawn from a distribution with a
density and $o(s) = (1 - s)\, o(0)$, then with probability one the path meets
no point of $\mathcal{E}_2$ with $s < 1$.*

**Proof.** The path meets $e \in \mathcal{E}_2$ at some $s < 1$ exactly when
$o(0) = r\, e$ for some $r \ge 1$. So the bad starting points are the image of
$\mathcal{E}_2 \times [1, \infty)$ under the smooth map $(e, r) \mapsto r e$,
whose domain is a countable union of manifolds of dimension at most $N - 1$.
A $C^1$ image of a manifold of lower dimension has Lebesgue measure zero in
$\mathbb{R}^N$, and a countable union of null sets is null. $\square$

To start with every limit inactive, take $o_c(0) = p_c (1 + U_c) + \epsilon_c$
with independent $U_c$ uniform on, say, $[0.5, 1.5]$. The first births then
fall near $s_c \approx U_c / (1 + U_c)$, spread over roughly $(1/3, 3/5)$
rather than within $10^{-3}$ of the start.

**The hypothesis on $\mathcal{E}_2$** is where the work lies. Each kind of
structure change is a single equation on the offsets (a peak of a limit
reaching its offset; a multiplier reaching zero at an exit; a riding control
reaching a bound). Away from degenerate points, the implicit function theorem
makes each of these a $C^1$ hypersurface, and two independent ones meet in
codimension two. What "away from degenerate points" requires is what Lemma E
and the local checks supply.

### Lemma F (time reversal maps extremals to extremals)

*Suppose there are signs $r \in \{\pm1\}^{n_x}$ on the states and
$\sigma \in \{\pm1\}^{n_u}$ on the controls, with $\mathcal{R} =
\mathrm{diag}(r)$ and $\Sigma = \mathrm{diag}(\sigma)$, such that, for all
$x, u$:*

1. $f(\mathcal{R} x, \Sigma u) = -\mathcal{R} f(x, u)$,
2. $L(\Sigma u) = L(u)$, with $f$ and $L$ free of $t$,
3. the set of limits is mapped to itself: for each $c$ there is $c'$ with
   $g_c(\mathcal{R} x) = g_{c'}(x)$,
4. the control set is mapped to itself by $\Sigma$.

*Then if $(x, \lambda, u)$ is an extremal on $[0, T]$ from $x_0$ to $x_T$, so is*

$$\tilde x(t) = \mathcal{R}\, x(T - t), \quad \tilde u(t) = \Sigma\, u(T - t),
\quad \tilde\lambda(t) = -\mathcal{R}\, \lambda(T - t),$$

*from $\mathcal{R} x_T$ to $\mathcal{R} x_0$, with the same $T$ and cost, and
limit $c$'s arcs becoming limit $c'$'s arcs at the reflected times.*

**Proof.** $\dot{\tilde x}(t) = -\mathcal{R} f(x, u) = f(\mathcal{R} x,
\Sigma u) = f(\tilde x, \tilde u)$ by (1). The Hamiltonian is unchanged:
$\tilde\lambda^\top f(\tilde x, \tilde u) = (-\mathcal{R}\lambda)^\top
(-\mathcal{R} f) = \lambda^\top f$, since $\mathcal{R}^2 = I$, and $L$ is
unchanged by (2). So $\tilde H = H$ along the motion. The costate equation:
$\dot{\tilde\lambda}(t) = \mathcal{R}\,\dot\lambda(T - t) = -\mathcal{R}\,
\partial_x H$, and since $\tilde x = \mathcal{R} x$, $\partial_{\tilde x}
\tilde H = \mathcal{R}\, \partial_x H$, so $\dot{\tilde\lambda} =
-\partial_{\tilde x} \tilde H$. Minimality of the control: by (4) and (2),
$\tilde u$ minimizes $\tilde H$ over the control set exactly when $u$ minimizes
$H$. The limits' multipliers and arcs carry over by (3), with the same algebra
applied to $H + \mu^\top g$. $H = 0$ (free final time) is preserved because
$\tilde H = H$. $\square$

*What the CAS checks.* For the jerk chain ($\dot q = w$, $\dot w = a$,
$\dot a = j$) the signs are forced: $r = (+, -, +)$ on $(q, w, a)$ and
$\sigma = -$. (1) and (2) are checked exactly. (3) is checked exactly when the
limits are small expressions. For Newton-Euler torques it holds by
construction: $\tau = M(q) a + c(q, w) + G(q)$ with $c$ quadratic in $w$, so
$\tau(q, -w, a) = \tau(q, w, a)$. Rate limits come in $\pm$ pairs, which
reversal swaps, and angle limits do not involve $w$.

**Rule R4 (reversal).** For a rest-to-rest motion (or any motion whose ends are
reversed by $\mathcal{R}$), the problem from $x_T$ to $x_0$ is the reversal of
the problem from $x_0$ to $x_T$. An extremal of either gives one of the other,
exactly, through Lemma F. A failed solve can therefore be retried on the
reversed problem, and an answer mapped back; this needs no new search. The
route, its continuation path and its events differ between the two, because
the reversed problem's tightening starts from a different motion, so one may
succeed where the other stalls.

### Lemma G (cyclic coordinates)

*If a state $x_k$ appears in none of the dynamics, the running and terminal
costs and the limits, then its costate is constant along every extremal.*

**Proof.** $\dot\lambda_k = -\partial H / \partial x_k = 0$, since no term
of $H$, or of $H + \mu^\top g$ on an arc, contains $x_k$. $\square$

For both arms, gravity (none for the two-link, along the base axis for the six-joint)
leaves the base angle $q_1$ out of everything, so $\lambda_{q_1}$ is constant.
The motion is also unchanged by adding a constant to $q_1$ at both ends, which is
why the embedding treats $q_1$ as invariant.

### Lemma H (the multiplier of an order-1 arc does not rise)

*On a boundary arc of an order-1 limit, in the indirect formulation that the
shooting uses ($H + \mu\, \dot g$), the multiplier satisfies $\mu \ge 0$,
$\dot\mu \le 0$, $\mu(t_{\rm out}) = 0$ and $\mu(t_{\rm in}^+) = \pi$, the
jump at the entry.*

**Proof.** In the direct formulation, the limit is adjoined with a
nonnegative measure $\eta$ supported on the arc. That is the maximum principle
for state constraints; see Hartl, Sethi & Vickson (1995), whose Section 4 relates
the two formulations. The two costates differ on the arc by
$\lambda_{\rm ind} = \lambda_{\rm dir} - \mu\, \nabla g$, with
$\mu(t) = \int_t^{t_{\rm out}} d\eta$. Then $\mu \ge 0$ and $\dot\mu = -\eta \le 0$
because $\eta \ge 0$, and $\mu(t_{\rm out}) = 0$. The jump of
$\lambda_{\rm ind}$ at the entry is $\mu(t_{\rm in}^+) \nabla g$, which is
$\pi \nabla g$. $\square$

The solver checked $\mu \ge 0$ only. That is weaker: an arc whose $\mu$ dips and
rises again passes it, although $\eta < 0$ there. Lemma F exposed this. Under
reversal $\tilde\mu(T - t) = \pi - \mu(t)$, so a reversed extremal with $\tilde\mu$
rising above $\pi$ maps to a forward one with $\mu < 0$. The six-joint case
"goal 9 at 90%" did exactly that: accepted in the reversed direction, rejected
forward. The check $\dot\mu \le 0$ (no rise along an arc of order 1) is now part
of acceptance. For arcs of order 2 or more the sign conditions differ and are
not derived here.

**Several limits at once.** The lemma holds for each order-1 limit separately,
whatever other limits do. With limits $g_c$ adjoined by measures $\eta_c \ge 0$
and $\lambda_{\rm ind} = \lambda_{\rm dir} - \sum_c \mu_c \nabla g_c$, where
$\mu_c(t) = \int_t^{t_{{\rm out},c}} d\eta_c$, differentiate:

$$\dot\lambda_{\rm ind} = -L_x - \lambda_{\rm dir} f_x - \sum_c \eta_c \nabla g_c
  + \sum_c \eta_c \nabla g_c - \sum_c \mu_c \nabla^2 g_c\, f ,$$

which is $-\partial_x\big(L + \lambda_{\rm ind} f + \sum_c \mu_c \dot g_c\big)$,
since $\dot g_c = \nabla g_c f$ (the limits do not depend on $t$). So the
indirect formulation with all the $\mu_c$ is the direct one, term by term.
The sign condition is on each $\eta_c$, so $\dot\mu_c = -\eta_c \le 0$ for each
$c$ on its arc, also on stretches where another limit $d$ is active and the
control solves the riding equations of the set $\{c, d\}$. $\square$

*Observed.* The six-joint case "goal 4 at 90%" converges (residual 2e-13) with
$\mu$ of `tau5_min` rising from 0.0122 to 0.0126 exactly while `tau4_max` is
active, $t \in [0.6027, 0.6056]$. $\mu_{\tau5}$ is continuous at both
junctions of `tau4_max`; the rise is a smooth $\eta_{\tau5} < 0$ over the
coupled stretch, not a sampling artifact. It is not an extremal, and the
acceptance check rightly rejects it. "Goal 9 at 80%" does the same: `tau5_max`
rises by 0.027 while `tau6_max` is active, and `tau2_min` while `tau6_min` is.

### Rule R5 (split an order-1 arc where its multiplier rises)

*Hypotheses:* the limit is of order 1 (C3), and the arc is ridden by the
shooting in the indirect formulation of Lemma H.

*Rule:* along the route, when the solution at a step has an order-1 arc whose
$\mu_c$ rises (by more than rounding) over $[t_1, t_2]$, offer the structure
with that arc split in two, $[t_{\rm in}, t_1]$ and $[t_2, t_{\rm out}]$.
The seed follows from $\mu_c = \int_t^{t_{\rm out}} d\eta_c$ on each piece:
the first piece must leave with $\mu_c = 0$, so on it $\mu_c$ becomes
$\mu_c - \mu_c(t_1)$ and its entry jump is $\pi - \mu_c(t_1)$; the second
enters with jump $\mu_c(t_2)$. The direct costate before the entry is
unchanged. (Keeping $\pi$ on the first piece leaves $\mu_c(t_1) > 0$ at its
exit, a jump in the control there; the six-joint seeds built that way
diverged, with residuals near $10^7$.)

*Why it is sound:* by Lemma H the current structure is not an extremal at any
step where the rise is present, so continuing with it cannot end in an
accepted answer. The split is the change the sign condition points to:
$\eta_c < 0$ there, so the limit should not be held there. Like the other
structure changes it is only proposed. The solve of the new structure decides
it, and acceptance re-checks every condition. The rule changes which
structures are tried, never what is accepted.

*Open:* the event at which the rise first appears ($\dot\mu_c = 0$ at an
interior point, with $\ddot\mu_c = 0$) is the analogue of a death, and the gap
should be born at zero width there. The rule as built detects the rise after
a step, as the route detects free-arc violations.

### Lemma I (contacts with an order-1 limit lie where it can be held)

*Let the limit $g(x) \le 0$ have order 1, with $\dot g = a(x)^\top u + d(x)$
(affine in $u$ by C1, as $g$ depends on $x$ only), $a, d$ continuous, and the
control set $U$ convex and compact. Call $x$* holdable *when some $u \in U$
gives $a(x)^\top u + d(x) = 0$. If a trajectory with $u(t) \in U$ a.e. has
$g(x(t^*)) = 0$ at an interior time and $g(x(t)) \le 0$ near $t^*$, then
$x(t^*)$ is holdable.*

**Proof.** The image of $U$ under $u \mapsto a(x)^\top u + d(x)$ is a closed
interval (a continuous linear image of a convex compact set). Suppose it does
not contain 0 at $x^* = x(t^*)$; say it lies in $[\delta, \infty)$ with
$\delta > 0$ (the other sign is the same, in reverse time). The interval's
ends depend continuously on $x$, and $x(t)$ is continuous, so for $t$ near
$t^*$ every $u \in U$ gives $a(x(t))^\top u + d(x(t)) > \delta / 2$. Then
$\dot g > \delta / 2$ a.e. near $t^*$ and $g(x(t)) > 0$ for $t$ just after
$t^*$, a contradiction. $\square$

No smoothness of $u$ is used, so the lemma covers bang-bang controls and
every point of a boundary arc as well as touches. For a box
$U = \prod_k [m_k - r_k, m_k + r_k]$ the condition is explicit:
$|a(x)^\top m + d(x)| \le \sum_k |a_k(x)|\, r_k$. For the arm,
$a = \pm M(q)_{i,:}$ (C3), so the holdable set of joint $i$'s torque limit is
where the rate of $\tau_i$ at mid-range jerk is at most
$\sum_k |M_{ik}(q)| J_k$. At speed the velocity terms of $\dot\tau_i$ can
exceed that, and then joint $i$'s limit cannot be reached there by any
admissible motion.

*Corollary.* The peak of a violation on a free arc of the path problem is
holdable. Its control is in $U$, and $g$ has an interior maximum there, so
the argument applies to $g - g(x(t^*))$.

*Observed.* In rest-to-rest case 4 the base joint moves at up to 6 rad/s. At
the entry of a newborn `tau1_max` arc, $\dot\tau_1$ at zero jerk is 84
while the jerks can cancel at most 80.5. At a `tau1_min` entry it is
$-349$ against 62. Holdability changes sign inside the violated window: the
window starts where the limit cannot be held, and the peak, as the corollary
says, is holdable. Seeds that spanned the whole window failed to integrate
(the run-time riding solve correctly reports no admissible control). Every
`tau1` birth in that case failed this way.

### Rule R6 (newborn order-1 arcs are seeded where the limit can be held)

*Hypotheses:* C1, the limit is of order 1, and the control set is a box (the
kernel's riding solve then decides holdability exactly: it searches the
bounds' active sets, exhaustively if needed, and fails only when no
admissible control holds the limit).

*Rule:* a newborn boundary arc's ends are pulled in to the largest stretch
of its violated window that contains the peak and on which every sampled
state is holdable.

*Why it is sound:* by Lemma I an arc of any feasible trajectory lies in the
holdable set, so an arc seeded on a non-holdable stretch cannot be one, and
pulling the ends in removes only times that no answer can use. The peak is
holdable (corollary), so the stretch is never empty, up to sampling. Like
R5, R6 changes only which seed is tried, never what is accepted.

*Open:* holdability is sampled on the free trajectory, but riding changes
the state, and an arc can leave the holdable set partway. The exact
statement is that the arc must exit before the edge of the holdable set, an
event that could be watched during the integration. The edge is where every
jerk with $a_k \ne 0$ is at a bound, which is outside Lemma B
($S \cap B \ne \emptyset$).

### Lemma J (junctions of an order-1 arc are tangential, bounds or not)

*Assume C1 and C2 (the running cost is strictly convex in the control, with
constant Hessian $R \succ 0$) and a convex control set $U$. Let $g$ be an
order-1 limit with $\dot g = a(x)^\top u + d(x)$. At an exit $\tau$ of a
boundary arc of $g$ (with $g < 0$ just after), and at an entry (with
$g < 0$ just before), the control is continuous, and
$a^\top u(\tau^\pm) + d = 0$ on the free side as well. This holds whether or
not the control is at a bound there. No other limit may change at $\tau$.*

**Proof.** Use the direct formulation (Hartl, Sethi & Vickson 1995). The
control minimizes $H = L + \lambda f$ over $U$ at almost every time, and at
the one-sided limits by continuity. The costate may jump at $\tau$ by an atom
$\nu = \eta(\{\tau\}) \ge 0$ of the multiplier:
$\lambda(\tau^+) = \lambda(\tau^-) - \nu \nabla g$. Both sides minimize
$\tfrac12 u^\top R u + q^\top u$ over $U$. Since $G^\top \nabla g = a$, the
linear terms differ by $q^+ = q^- - \nu a$.

Write the two optimality conditions: $(R u^- + q^-)^\top (u^+ - u^-) \ge 0$
and $(R u^+ + q^+)^\top (u^- - u^+) \ge 0$. Adding them gives
$$\nu\, a^\top (u^+ - u^-) \ge \|u^+ - u^-\|_R^2 .$$
At an exit, $a^\top u^- + d = 0$ on the arc and $a^\top u^+ + d \le 0$ just
after (the limit is left), so $a^\top(u^+ - u^-) \le 0$. With $\nu \ge 0$ the
left side is $\le 0$, hence $u^+ = u^-$ and $a^\top u^+ + d = 0$. At an
entry the signs swap: $a^\top u^- + d \ge 0$ just before and $= 0$ just after,
and the same inequality, with $u^\pm$ exchanged, gives the same conclusion.
$\square$

**What it changes.** The solver used $\mu = 0$ at an exit, and $\dot g = 0$
at an entry, only when the free-side controls are strictly inside their
bounds (the regular case of Jacobson, Lele & Speyer 1971). Otherwise it used
continuity of $H$. Lemma J removes the restriction. At a junction where the
control is at a bound, $H$ continuity holds identically, since the control is
continuous and $\dot g = 0$, so it is a degenerate row. The rate row
$a^\top u + d = 0$, under the control of the side that is not the arc, is the
condition that fixes the junction. In the regular case it is a nonzero
multiple of $\mu$ (at an exit,
$\dot g_{\rm free} = -a^\top R^{-1} a\, \mu$), so nothing changes there.

At the edge of the holdable set of Lemma I, every control with $a_k \ne 0$
is at a bound. There the exit can carry $\mu(\tau^-) = \nu > 0$: the box's
corner absorbs $\nu a$, so the same vertex minimizes both sides. The rate row
still holds, and $\mu = 0$ does not. An arc forced off its limit at the edge
is therefore an extremal the old exit row could not represent.

*Observed.* In rest-to-rest case 11 the riding of `tau1_min` drives the
jerks of joints 2 and 3, then 1, to their bounds at $t \approx 0.333$, with
$\mu = 3.6 \times 10^{-4} > 0$ and falling. Past that point the riding solve
can hold the limit only with jerks of $\pm 30$ on joints 4 and 5, whose
coefficients are $0.016$ and $-0.007$. There $\mu$ turns negative
($-0.0196$), so no extremal rides there.

### Rule R7 (rate rows at every junction of an order-1 arc)

*Hypotheses:* C1, C2, a box of controls (or none), an order-1 limit, and no
bang-bang control (C2 excludes it).

*Rule:* the junction rows of an order-1 boundary arc are
$a^\top u + d = 0$, under the arriving control at the entry and the departing
control at the exit, in place of $\mu = 0$ or $H$ continuity
(`MultiArcShooting.rate_junctions`).

*Why it is sound:* by Lemma J these rows are necessary conditions at every
such junction, and in the regular case they have the same roots as the old
rows. Acceptance is unchanged: $\mu \ge 0$, $\dot\mu \le 0$ (Lemma H) and
the jump signs are still checked.

*Correction: where the row is used.* As first built, R7 used the rate row at
every order-1 junction. That broke two slow tests (an arc ridden from the
start, and the two-link arm's run-time arcs). The rate row is
$a^\top u_{\rm free} + d$, and wherever the free law is clipped at a bound it
does not depend on the costates. Its gradient vanishes there, so Newton
cannot move off it. In the failing test the seed's free jerk at the exit was
$-J$, so the row was stuck at $-1$, while the answer has the jerk interior
and $\mu = 0$ at the exit. Lemma J holds at every junction, but the row is
well posed only where it is needed: at the corner of Lemma K, where the
riding control has every control that moves $\dot g$ at a bound. R7 now
uses it there and keeps $\mu = 0$ (or $H$ continuity) elsewhere
(`MultiArcShooting._rate_row_here`). The lesson for rule-making: a necessary
condition can be the right equation and still the wrong row, because a
row's usefulness to Newton is its gradient, which the lemma says nothing
about.

*Off the corner, multiplier rows (a second consequence of Lemma J).* With
the rate row kept to the corner, the junctions off the corner where the free
side's control is clipped fell back to $H$ continuity. That row has a double
root (Newton converges only linearly), and the census ran 31% slower on the
common cases. Lemma J's argument gives better rows there. Off the corner,
some control $k$ with $a_k \ne 0$ is strictly inside its bounds on the arc.
Since the control is continuous at the junction, the $k$-th components of
the two optimality conditions ($R u + c + G^\top\lambda + \mu a = 0$ on the
arc, the same without $\mu a$ off it, and $\lambda^+ = \lambda^- - \pi
\nabla g$ at an entry) agree. So:
$$\mu(t_{\rm out}^-)\, a_k = 0 \;\Rightarrow\; \mu(t_{\rm out}) = 0,
\qquad (\mu(t_{\rm in}^+) - \pi)\, a_k = 0 \;\Rightarrow\; \mu(t_{\rm in}^+) = \pi ,$$
whatever the other controls are clipped to. Both rows have simple roots. So
under C2, $H$ continuity is never needed at order-1 junctions: the rows are
$\mu = 0$ (exit) and $\mu = \pi$ or $\dot g = 0$ (entry) off the corner, and
the rate rows at the corner (`MultiArcShooting._multiplier_row_here`).

## Lemma E (birth of an arc: normal form). Statement and plan of proof

This is the result behind continuing in an arc's width rather than in $s$.
The statement is standard in kind (compare Bonnans & Hermant 2008 for limits
of order one). It has to be proved for our shooting formulation, and that
proof is not done yet. Below are the statement and the plan.

*Setting.* At $s = s^*$ the extremal has structure $\mathcal{S}$ (possibly
with other arcs; for a nested birth, $c$ comes into play while a set $S$ is
ridden). Let $\Phi(v, s) = 0$ be the shooting equations for $\mathcal{S}$.
Limit $c$ is inactive except at one time $t^*$, where its peak touches its
offset.

*Hypotheses.*

- (H0) C2, C3, C4 hold, so the motion on either side of $t^*$, free or riding
  $S$, is smooth (Lemma B, needing $S \cap B = \emptyset$ there).
- (L1, local) the peak is nondegenerate: $\ddot g_c(t^*) < 0$ along the motion
  at $t^*$. For a nested birth this curvature is along the riding motion of $S$,
  not the free motion.
- (L2, local) $\partial \Phi / \partial v$ is nonsingular at $(v^*, s^*)$.
- (L3, local) the peak value $\pi_c(s) = \max_t g_c - o_c(s)$ crosses zero
  transversally: $d\pi_c / ds \ne 0$ at $s^*$.
- the jerks are strictly inside their bounds at $t^*$ (so C5 holds there and the
  entry is tangential).

*Claim.* For $s$ on the side where $\pi_c > 0$ and close to $s^*$, there is a
unique nearby extremal with structure $\mathcal{S}$ plus one arc of $c$ around
$t^*$. Its width $w$ satisfies $s - s^* = \kappa\, w^2 + O(w^3)$ with
$\kappa \ne 0$, and all its unknowns are $C^1$ functions of $w$ near $w = 0$.
Continuing in $w$ is therefore regular, while continuing in $s$ is not:
$dw/ds \sim (s - s^*)^{-1/2}$.

*Plan of proof.* Write the arc's entry and exit as $t_m \mp w/2$ and its jump
as $\pi = w\, \tilde p$; the unknowns become $(v, t_m, \tilde p, s)$ and $w$
becomes the parameter. The new rows are the entry conditions $g_c = 0$ and
$\dot g_c = 0$ (tangential) and the exit condition $\mu_c = 0$. At $w = 0$:

1. The old rows reduce to $\Phi(v, s) = 0$: an arc of zero width with zero jump
   changes nothing.
2. The entry rows reduce to "the motion touches the offset at $t_m$":
   $\dot g_c(t_m) = 0$ fixes $t_m$ (nondegenerate by L1), and $g_c(t_m) = o_c(s)$
   fixes $s$ (by L3).
3. The exit row: $\mu_c(t_m^+) = \pi$ for this formulation, so
   $\mu_c(t_m + w/2) = w\,(\tilde p + \dot\mu_c) + O(w^2)$. Divided by $w$, it
   gives $\tilde p = -\dot\mu_c$ at $w = 0$: a regular equation for $\tilde p$.

With those rows rescaled, the Jacobian at $w = 0$ is block triangular, with
nonsingular diagonal blocks (L2, L1, L3, and the identity block for
$\tilde p$). The implicit function theorem gives the $C^1$ dependence on $w$.
The relation $s - s^* = \kappa w^2$ then follows by expanding $g_c$ to second
order about $t_m$, using L1.

*To make rigorous:*

- the smoothness in $w$ at $w = 0$ of the flow composed through an arc of
  width $w$;
- the exact form of the rescaled rows in our implementation (`shooting.py`
  uses $\pi$ as the costate jump with $\mu_c(t_{\rm in}^+) = \pi$);
- the block-triangular structure, entry by entry;
- that the solution found is the one with $\mu_c \ge 0$ on the arc (the sign
  of $\tilde p$).

## Rules these license

| Rule | What the solver does | Needs |
|---|---|---|
| R1 | Start the offsets at $o_c(0) = p_c (1 + U_c) + \epsilon_c$ with random $U_c$, so that births do not coincide | Lemma D: C1-C5, plus its hypothesis on $\mathcal{E}_2$, which rests on Lemma E and the local checks |
| R2 | At a birth, continue in the new arc's width $w$, starting at $w = 0$ with the unknowns of Lemma E, rather than stepping in $s$ | Lemma E: C2-C4 globally; L1-L3 checked at the event |
| R3 | A nested birth is R2 with the curvature in L1 taken along the riding motion of the arcs already active | as R2, with $S \cap B = \emptyset$ at the event (Lemma B) |
| R4 | When a solve fails, solve the reversed problem and map its extremal back exactly | Lemma F: C6 |

When a hypothesis fails the rule stays off, and the solver records which
hypothesis failed and where. A failure of L1 means a degenerate touch (a
higher-order contact); of L2, a turning point (pseudo-arclength continuation
is the tool there, and it is not part of this note); of L3, an offset path
tangent to the event; of $S \cap B = \emptyset$, the corner case below.

## What this note does not cover

- **Jerk saturation.** C5 fails where a jerk reaches its bound. The control law
  is then only Lipschitz, the shooting map is piecewise smooth, and Lemmas D and
  E do not apply as stated. Either the clip is smoothed (a barrier on the jerk
  bounds, driven to zero) or the arguments are redone for piecewise smooth
  maps.
- **A riding control reaching its bound** ($S \cap B \ne \emptyset$; the
  two-link case that needed jerk 1.51 against a bound of 1). The expected
  result is a corner junction with $H$ continuous (mixed control-state
  constraints; Maurer). It needs its own normal form.
- **Turning points of the path,** global reach of $s = 1$ (bounded path,
  existence at the tightened limits), and whether the extremal reached is a
  minimum (second-order conditions). Probability-one homotopy theory
  (Chow, Mallet-Paret & Yorke 1978; Watson) handles reach given boundedness;
  boundedness is problem dependent.
- **Other classes.** Limits of order 2 and 3, the bang-bang policy, and other
  costs. Each lemma lists its certificates, so extending it means checking which
  certificates a new class can provide.

## How the certificates would be computed

For a problem given to `Homotopy`:

- C1 and C2: differentiate the SymPy dynamics and cost; exact.
- C3: differentiate each limit along the dynamics (already done by
  `constraint_derivatives`); record $\partial \dot g / \partial j$. For
  `joint_torques` limits this is $\pm M(q)_{i,:}$ by construction, and the
  construction records it.
- C4: from the link table, check every principal moment is positive (exact
  rationals here), then apply Lemma A. For limits not built by
  `joint_torques`, C4 is unknown unless proved otherwise, and Lemma B stays off.
- C5: list the nonsmooth operations in the derived functions (`Min`, `Max`,
  clips).

Local checks (L1-L3, $S \cap B = \emptyset$) are evaluated at the event, with
a rigorous error bound.

## References

- J. Bonnans, A. Hermant (2008), *Stability and sensitivity analysis for
  optimal control problems with a first-order state constraint and
  application to continuation methods*, ESAIM: COCV 14.
- H. Th. Jongen, P. Jonker, F. Twilt (1986), *Critical sets in parametric
  optimization*, Mathematical Programming 34.
- S.-N. Chow, J. Mallet-Paret, J. A. Yorke (1978), *Finding zeroes of maps:
  homotopy methods that are constructive with probability one*, Mathematics
  of Computation 32.
- E. L. Allgower, K. Georg (1990), *Numerical Continuation Methods*,
  Springer.
- D. H. Jacobson, M. M. Lele, J. L. Speyer (1971), *New necessary conditions
  of optimality for control problems with state-variable inequality
  constraints*, JMAA 35.
- H. Maurer (1977), *On the minimum principle for optimal control problems
  with state constraints*, Schriftenreihe des Rechenzentrums der Universität
  Münster 41.
