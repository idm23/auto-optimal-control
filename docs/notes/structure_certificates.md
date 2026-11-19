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

### Rule R8 (exits are pulled in to the edge of the holdable set)

*Hypotheses:* those of Lemma I and R7, with limits ridden at run time (the
riding solve fails exactly where no admissible control holds the limits).

*Rule:* when a seed does not propagate because a boundary arc's riding fails
partway, the last time to which it does propagate (the edge) is found by
bisection. The arc's exit is moved to just inside the edge and the seed is
solved again (`MultiArcShooting.pull_in_exits`).

*Why it is sound:* by Lemma I no answer rides a limit past the edge, so the
times removed are no answer's. By Lemma J an exit at the edge is a legitimate
junction, fixed by R7's exit row, even with $\mu > 0$. It changes only the
seed. Newton iterates that step past the edge are already handled: the
evaluation fails, and the chord iteration backtracks.

*Result: off by default.* On rest-to-rest cases 1, 4 and 11 it fired 4, 18
and 7 times and solved none. In case 11 the route converges with the exit
near 0.346 up to $s \approx 0.67$. Beyond that the exit approaches the edge
from inside. Pulled-in seeds start far from the solution (residuals 20 to
25), and Newton iterates near the solution keep landing past the edge, where
the residual is undefined. The lesson is that an exit reaching the edge is a
structure event in its own right: the arc goes from a regular exit
($\mu = 0$, inside) to an exit forced at the edge ($\mu > 0$). To follow it,
the edge must be a row of the shooting: the holdability margin, generated
from the expressions $a(x)$ and $d(x)$, equal to zero at the exit, with the
riding integration stopped there by an event, not by failure.

### Lemma K (at a forced exit, R7's row is the margin)

*Under the hypotheses of Lemma J, with a box $U$ and $a_k(x) \ne 0$ for every
$k$, let an order-1 arc exit at $\tau$ on the edge of the holdable set. Then
the control at $\tau$ is the corner $u^v$ of the box at which $a^\top u + d$
is extreme, and so is the free control law at the (continuous) indirect
costate. Wherever the free law is at that corner, R7's exit row equals
$\mp m(x)$, the holdability margin
$m(x) = \sum_k |a_k| r_k - |a^\top \bar u + d|$, with $\bar u$ the box's
centre.*

**Proof.** At the edge $m(x(\tau)) = 0$, so the only control in $U$ with
$a^\top u + d = 0$ is the corner $u^v$, which is therefore the riding
control. By Lemma J the control is continuous, so the free law gives $u^v$ at
$\tau^+$. At that corner $a^\top u^v + d = \pm(|a^\top \bar u + d| - \sum_k
|a_k| r_k)$, with the sign fixed by which way the drift points. $\square$

So the margin row the R8 lesson asked for is already in the shooting:
R7's row, wherever the free law sits at the corner (an open set of costates,
since the box projection saturates there). The margin does not need to be
generated separately. What was missing is a residual defined past the edge.

### Rule R9 (continue past the edge on the free law)

*Rule:* when the riding of a single order-1 limit fails before that arc's
exit time, ride to the edge (found by bisection), then continue on the free
law to the exit time (`MultiArcShooting.past_edge`). The residual is then
defined on both sides of the edge, and by Lemma K R7's row crosses zero
there.

*Why it is sound:* it defines the residual only where it was undefined (the
evaluation used to fail). At a solution with the exit before the edge
nothing changes. At a solution with the exit on the edge, the motion is the
riding motion up to the exit (Lemma K). Acceptance still checks every
condition. The Jacobian leaves out the edge time's own dependence on the
unknowns, which costs Newton accuracy only past the edge.

*Result: off by default.* On the census (with R1, R4, R5, R6 and R7) it
solved 24 of 45, against 27 without it. It lost rest-to-rest cases 0 and 5,
and goal 5 at 90%, which is timing noise. Answers common to both runs agree to
4e-16. On rest-to-rest case 11 it removed few of the failed evaluations. Most
were a different failure: Newton steps that take an arc's exit before its
entry ("arc times out of order", 895 of about 1200 failed evaluations). That
is the square-root law of Lemma E: near its birth or death an arc's width
moves like the square root of the homotopy parameter, so Newton in the
junction times overshoots zero width. It points to continuing in the width.

## Lemma E (birth of an arc: normal form)

This is the result behind continuing in an arc's width rather than in $s$.
It is standard in kind; compare Bonnans & Hermant (2008) for limits of order
one. Here it is proved for this shooting formulation: an arc of an order-1
limit, ridden at run time, with the entry rows $g = 0$ and $\dot g = 0$ and
the exit row $\mu = 0$, with $\mu(t_{\rm in}^+) = \pi$.

*Setting.* At $s = s^*$ the extremal has structure $\mathcal{S}$, which may
have other arcs. Its shooting equations are $\Phi(v, s) = 0$, at a solution
$v^*$. Limit $c$, with offset $o_c(s)$, is inactive except at one time $t^*$,
where its peak touches the offset: $g_c(t^*) = o_c(s^*)$ and
$\dot g_c(t^*) = 0$.

*Hypotheses.*

- (H0) C2, C3 and C4 hold, and the jerks are strictly inside their bounds at
  $t^*$. So the free and riding flows are smooth near $z(t^*)$: Lemma B, and
  the riding control exists near $z(t^*)$ by Lemma I's margin, which is
  positive there.
- (L1) the peak is nondegenerate: $\ddot g_c(t^*) < 0$ along the motion.
- (L2) $\Phi_v(v^*, s^*)$ is nonsingular.
- (L3) the peak excess $\pi_c(s) = \max_t g_c - o_c(s)$, along the branch
  $v(s)$ of L2, crosses zero transversally: $\pi_c'(s^*) \ne 0$.

*Claim.* Write the arc as $[t_m - w/2,\ t_m + w/2]$ with jump
$\pi = w\,\tilde p$. Near $w = 0$ there are unique smooth functions
$(v, t_m, s, \tilde p)(w)$ that solve the shooting equations of $\mathcal{S}$
plus this arc. At $w = 0$ they equal $(v^*, t^*, s^*, \tilde p_0)$ with
$\tilde p_0 = -\dot\mu_c$, and
$$s(w) - s^* = \kappa\, w^2 + O(w^3), \qquad
  \kappa = -\frac{\ddot g_c(t^*)}{8\, \pi_c'(s^*)} \ne 0 .$$
So $w \sim \sqrt{(s - s^*) / \kappa}$. Continuing in $s$ has
$dw/ds \to \infty$, while continuing in $w$ is regular.

**Proof.**

1. *The rows are smooth in $w$, including $w = 0$ and $w < 0$.* The state at the
   exit is $\varphi_{\rm ride}(t_m + \tfrac w2, t_m - \tfrac w2,
   J_{w \tilde p}(\varphi_{\rm free}(t_m - \tfrac w2)))$, with $J_\pi$ the
   costate jump. By H0 each piece is smooth in its times and initial values on a
   neighbourhood. The riding flow may run backward (for $w < 0$), since its
   vector field is smooth there too. So every old row,
   $\tilde\Phi(v, s, t_m, \tilde p, w)$, is smooth. At $w = 0$ the arc has zero
   length and zero jump, so $\tilde\Phi(\cdot, w = 0) = \Phi(v, s)$, whatever
   $t_m$ and $\tilde p$ are.

2. *Entry rows.* $E_1 = g_c(x(t_m - \tfrac w2)) - o_c(s)$ and
   $E_2 = \dot g_c$ under the free law there. At $w = 0$ they are
   $g_c(t_m) - o_c(s)$ and $\dot g_c(t_m)$.

3. *Exit row.* With jerks interior, the riding equations for one limit give
   $\mu_{\rm ride}(z) = \alpha(z)\, \dot g_{\rm free}(z)$, with
   $\alpha = 1/(a^\top R^{-1} a) > 0$. Indeed $u_{\rm ride} = u_{\rm free} -
   R^{-1} a\, \mu$, and $a^\top u_{\rm ride} + d = 0$. So the exit row
   $X = \mu_{\rm ride}(z(t_{\rm out}))$ satisfies, at $w = 0$, $X = \alpha E_2$
   *identically* in the other unknowns. By Hadamard's lemma,
   $\hat X = (X - \alpha E_2)/w$ is smooth. Moreover
   $\partial X / \partial \tilde p = w\, \partial \mu(t_{\rm out}) / \partial \pi$,
   and a jump $\pi \nabla g$ shifts $a^\top u$ by exactly
   $\pi\, a^\top R^{-1} a$. Hence $\partial \mu / \partial \pi = 1$ at $w = 0$,
   and $\partial \hat X / \partial \tilde p = 1$ there.

4. *The Jacobian at $w = 0$ is nonsingular.* Order the unknowns
   $(v, t_m, s, \tilde p)$ and the rows $(\Phi, E_2, E_1, \hat X)$. The rows
   $\Phi$ involve only $(v, s)$, $E_2$ only $(v, t_m)$, and
   $\partial E_1 / \partial t_m = \dot g_c(t_m) = 0$. Eliminate $dv =
   -\Phi_v^{-1}\Phi_s\, ds$ (L2). Then $E_2$'s pivot in $t_m$ is
   $\ddot g_c(t^*) \ne 0$ (L1). $E_1$'s pivot in $s$ is
   $\partial_s E_1 - \partial_v E_1 \Phi_v^{-1} \Phi_s = \pi_c'(s^*)$, by the
   envelope theorem, since $\dot g_c = 0$ at the peak (L3). $\hat X$'s pivot in
   $\tilde p$ is 1. The determinant is
   $\det \Phi_v \cdot \ddot g_c \cdot \pi_c' \cdot 1 \ne 0$, so the implicit
   function theorem gives the smooth branch in $w$.

5. *The square-root law.* Expand $E_1$. Riding keeps $g_c$ equal to its entry
   value over the arc, and the entry value is where the free motion meets the
   offset, $\tfrac w2$ before $t_m$. To leading order
   $g_c(t_m \pm \tfrac w2) = g_{\rm peak} + \tfrac12 \ddot g_c (w/2)^2$, so
   $\pi_c(s) = -\ddot g_c\, w^2 / 8 + O(w^3)$. With
   $\pi_c(s) = \pi_c'(s^*)(s - s^*) + O((s - s^*)^2)$, this gives $\kappa$.
   The $O(w)$ term vanishes because $\dot g_c(t_m) = 0$ (step 2). $\square$

*What is used, and what is not.* Interior jerks at $t^*$ (H0) are used in
step 3, for the explicit $\mu_{\rm ride} = \alpha\, \dot g_{\rm free}$. At a
birth where jerks sit at bounds, Lemma J still makes the junction
tangential, but $\alpha$ changes with the held set, and this proof does not
cover that case. With R7's exit row in place of $\mu = 0$, step 3 is the same
up to the factor $-a^\top R^{-1} a$ (Lemma J). The local checks L1 to L3 are
not verified by the solver. R2 uses the lemma only to choose the parameter of
continuation, never to accept an answer.

### Rule R2 (step in the width at a stalled birth or death)

*Rule:* when a step in the route's parameter fails with $ds < 10^{-3}$, and
the solution has a finite order-1 arc of width under $5\%$ of $T$, solve
$[\Phi(v; s),\ w - w_{\rm target}] = 0$ for $(v, s)$. Try
$w_{\rm target} = 1.5\,w$ (a birth) and then $0.5\,w$ (a death). The
Jacobian's $s$ column is a finite difference. The result is accepted as a
step only if $s$ moves forward, and it goes through `settle` and the usual
checks (`Homotopy.width_steps`).

*Why it is sound:* by Lemma E the branch is smooth in $w$, where it is not
in $s$, so the augmented Newton iteration is regular near the event. The rule
changes only how the route moves along its own path, never what is accepted.

*Result: off by default, for now.* On the census (with R1, R4, R5, R6 and
R7) it solved 26 of 45 against 27. It gained goal 9 at 70%, and lost rest 5
and goal 5 at 90% (noise). Width steps fired 52 times in 9 cases. At deaths
(goal 0 at 80%, goal 6 at 90%) halving the width moves $s$ forward only
geometrically toward the death. Lemma E's $w^2 \propto (s^* - s)$ says to
extrapolate to $s^*$ from two such points, retire the arc, and step past it.
At the birth in rest 11, halving the width moves $s$ back from $0.00029$ to
$-0.0103$, as the lemma predicts (giving $s^* \approx -0.014$). The growth step
fails, because it lands near $s \approx 0.018$ without a predictor; the branch
is smooth in $w$, so a secant in $w$ is the predictor to use.

### Rule R10 (integrate a negative-width arc backward)

*Rule:* a segment whose end time comes before its start is integrated
backward, not refused (`MultiArcShooting.backward_segments`). Answers whose
junctions are out of order are rejected ("junctions out of order"). Along
the route, `_retire` turns such a solution into a structure again: it
removes an arc of negative width, or re-sorts crossed events.

*Effectiveness:* step 1 of Lemma E. The shooting map is a composition of
flows, smooth in every junction time whatever their order, as long as each
flow is defined. So Newton sees a smooth residual through an arc's zero
width instead of a wall of failed evaluations.

*Its limit:* the flows must exist. Riding backward from an entry needs the
riding control to exist there (H0: the margin of Lemma I positive). Newton's
overshoots are not always small. On the drift-limited axis every
negative-width failure became a riding failure instead (155, 241 and 348 of
them): the backward ride left the holdable set. On the plain axis they
disappeared (318 to 0), with 10% fewer steps and the same answers.

*Census* (45 cases, 300 s, with R1, R2 and R4 to R7), compared with the
same run without R10:

| | Without R10 | With R10 |
|---|---|---|
| Solved | 28 | 28 (gained goal 0 at 90% and rest 9; lost goal 3 at 80% and goal 5 at 90%) |
| Under 60 s | 12 | 17 |
| Median time | 77.3 s | 48.8 s (0.89 on the 26 common cases, same answers to 1e-15) |
| Failed evaluations | 70,296 negative width, 3,390 riding | 1,446 negative width, 15,749 riding |
| Steps not converged / no structure converged | 1,892 / 1,169 | 936 / 692 |

The prediction (negative widths fall sharply, the median falls) held. The
cost is that riding failures are now the main class. They are backward rides
that leave the holdable set, so the edge of Lemma I is now the obstacle on
both sides of an arc. R10 is on by default.

### What the toy and a direct comparison showed

On the drift-limited axis with $B = 1$ and $B = 2$ the route failed with riding
failures. A direct transcription, used only to *look* at the answer (never as a
seed), shows the optimal motion has a simple structure: jerk $+J$, a long
`tau_max` arc, jerk $-J$, a short `tau_min` arc, jerk $+J$. Its holdability
margin stays at 0.23 or more on the free stretches and 0.6 or more on the arcs.
So the answer is nowhere near the edge. The failures come from the route's
intermediate states, not from the answer.

### Rule R11 (arcs of one limit that touch are one arc)

*Observed:* with $B = 1$ the route reached the right structure, then R5 split
the `tau_max` arc, and the gap between the pieces closed as the route
advanced. The pieces touched, then crossed (exit 1.08692, entry 1.08690). Two
arcs of one limit cannot be ordered, so every later step failed ("entered
twice without an exit"), and the route stalled and backed out.

*Rule:* boundary arcs of one limit that touch or overlap are merged into one
arc, whose entry jump is the sum of theirs (`Homotopy._merge_touching`).

*Why it is sound:* by construction. Their union is the set of times on the
limit, and $\mu = \int d\eta$ adds over the pieces.

*Result:* $B = 1$ now solves exactly, matching the direct comparison's
structure (`tau_max` on $[0.489, 3.150]$, `tau_min` on $[3.90, 4.41]$). On the
arm it fired in 6 cases: the same 30 of 45 solved, 4% faster on the common
cases (median 72.9 s, 13 under 60 s).

### Lemma I′ (a peak on the edge) and one-sided births

*At an interior maximum $t^*$ of $g$ along a motion whose control $u^*$ is in
the box, the margin is $\ge 0$, with equality exactly when $u^*(t^*)$ is the
box's corner for $g$.*

**Proof.** $\dot g(t^*) = 0$ gives $a^\top u^* + d = 0$, so
$|a^\top m + d| = |a^\top (m - u^*)| \le \sum_k |a_k|\, |m_k - u^*_k| \le
\sum_k |a_k|\, r_k$. Equality needs $|m_k - u^*_k| = r_k$ for every $k$ with
$a_k \ne 0$, with the signs aligned: that is the corner. $\square$

*Observed:* with $B = 2$ the first `tau_min` birth has its peak on the edge
(margin $-0.001$ at the sampled peak, $-0.144$ before it, $+0.139$ after it).
The free jerk is saturated there. The limit is holdable only after the peak,
so the newborn arc is one-sided, $[t^*, t^* + w]$. Lemma E's normal form is
symmetric about $t^*$ and assumes interior jerks, so it does not cover this.
R6 gave up when the peak sample failed the exact test. It now starts from the
nearest holdable sample on the side where the margin rises ("R6 one-sided").
$B = 2$ still fails: riding from the edge leaves the holdable set at once,
because riding holds $g$ fixed and does not follow the free motion whose margin
grew. The one-sided birth needs its own normal form. It never fired on the arm.

### Rule R10b, tried

Continuing a backward ride on the free law when riding backward fails
(continuous at zero width) gained nothing. The fallback failed as often. On
the toy, most riding failures turned out to be *forward* (1,318 against 284).
Off by default.

### R7's rows, replayed

Replaying 221 census snapshots under each row policy compared three settings:
rate rows at the corner with $H$ continuity elsewhere; Lemma J's rows (rate
rows at the corner, multiplier rows elsewhere); and rate rows everywhere. Of
the 157 steps that converged in the census, they converge again 157, 157 and
146 times. On the 146 that all three solve, they take 94.8 s, 87.0 s and
102.3 s. Lemma J's rows are the best per step.

### Lemma L (a touch at the corner cannot correct anything)

*Tried:* bring a peak on the edge in as an order-1 touch point, with rows
$g = 0$ and $\dot g = 0$ (that is, $-$margin $= 0$) and a jump $\nu \ge 0$
(rule R12, `Homotopy.corner_touches`, with `MultiArcShooting.order1_touches`).

*Lemma L.* If the free control is at the corner on an interval after a touch
at $t_c$, the jump $\lambda^+ = \lambda^- - \nu \nabla g$ does not change the
control there (the box projection is saturated), and so it does not change
the motion. The shooting rows are locally independent of $\nu$, and the
Jacobian is singular.

*Observed:* on the toy with $B = 2$ the touch solves reach residuals of
1e-3 to 4e-5 and stop there. The Jacobian's smallest singular value is 3e-17
(condition number 2e19), and $\nu$ drifts to $-8$. A touch cannot correct an
overshoot that its jump has no way to influence. R12 is off.

*What the true path does:* a direct transcription of the toy ($B = 2$, comparison
only), with the limit tightened through the first `tau_min` contact, shows a
contact at a point ($t = 4.628$) that grows into a two-sided arc
($[4.676, 4.731]$, then $[4.764, 4.902]$). Its margin at the contact is 0.25,
well inside the holdable set; the jerk arrives at $-1$ and leaves interior.
So in the problem itself the birth is ordinary, and holdable. The route's
birth sat on the edge because its intermediate problem is not the true path's.
R1's staggered offsets tighten the limits in a different order, and there the
motion is faster. **A corner birth is an artifact of the route's path.** The
remedy belongs to the route: when a birth's peak lies on the edge, change
which limits tighten first, so that the contact happens where the limit can
be held.

### First contacts with drift are corner births, and arcs entered at the corner

With drift, a limit's *first* contact along a near-minimum-time free motion is
on the edge in the problem itself, not only on the route. The free jerk is
saturated almost everywhere, and a peak of $g$ under a saturated control has
margin exactly 0 (Lemma I′'s equality case). A direct transcription of the
toy ($B = 2$, comparison only), with the limit tightened from just above the
free peak, shows what follows. The contact grows into an **arc entered at the
corner**: the margin at the entry is within $\pm 0.02$ of zero at every
limit value, the jerk arrives at $-1$, and the riding jerk is just inside
($-0.95$, $-0.89$, …). The arc grows asymmetrically: the entry creeps
earlier, the exit moves later fast. This is the time reverse of Lemma K's
forced exit. R7's rate row at the corner ($\dot g = -$margin $= 0$ under the
arriving control) is its entry condition.

### Rule R13 (bring the limits in one at a time)

The limit the free motion breaks most comes first (`offset_order =
"sequential"`). Tried on the toy: no gain. Even the first limit is born on
the edge (above). It is one route of R15's portfolio.

### Rule R14 (ride the corner at the edge, up to rounding)

*Observed:* a ride entered at the edge (margin $-0.0000$ to four decimals)
failed on its first internal step. The riding problem has no admissible
control just past the edge, and the integrator's stage points land there by
rounding.

*Rule (generated riding code):* when the riding problem is infeasible by no
more than $10^{-6}$ of the jerks' reach, every control is held at the corner
except the one with the largest $|a_k|$, and the limit fixes that one. The
riding system is then nonsingular, and $\mu$ comes from that control's own
equation. This is the continuous limit from the holdable side, where that
control is the last still interior.

*Why it is sound:* it changes the riding vector field only within rounding
of the edge, where the exact one does not exist (Lemma B fails there).

### Rule R15 (a portfolio of routes)

*Rule:* when the active-set route fails, try the next route of a portfolio:
another draw of the offsets, then sequential births, then (Rule R24) these
three routes again, started at ten times the secondary weight
(`Homotopy.portfolio`).
Every route may use all the time left, and a route that sets no new low of
its offsets for `stall_seconds` (60 s) yields to the next.

*Why it is sound:* each route is a valid route (Lemma D: almost every draw
separates the births), and acceptance is unchanged.

*Effectiveness:* heuristic, but grounded in the census history. Which cases
solve depends on the route, and the union of the checked runs solved 34
cases while no single run solved more than 30.

*Census* (45 cases, 300 s per direction, with R1, R2, R4–R7, R10, R11 and
R14, against the same run without R15, which solved 30):

| Variant | Solved | Under 60 s | Median |
|---|---|---|---|
| Routes in turn, time split (first route half) | 29 | 14 | 68 s |
| In turn, each may use all the time left, yields after 60 s without progress | 34 | 13 | 104 s |
| The same with 120 s | 30 | 12 | 74 s |
| 60 s, offsets at zero and structure changes count as progress | 33 | 12 | 101 s |
| **Routes at once in forked processes** (`parallel_routes`) | **35** | **18** | **52 s** |

Routes in turn trade cases for cases: any rule for cutting one route short
also cuts routes that would have finished. Goal 8 and goal 9 at 70% need
their first route uninterrupted for 240 to 275 s. Run at once, each route
has the whole budget, and the solved set is the union of the routes. That
run gained goal 0 at 80%, goal 3 at 70% and 80%, goal 5 at 80%, goal 7 at
70% and rest 2, and lost rest 9 (solved before only through the reversed
problem, after 526 s). Common answers agree to 1e-15, and the common cases
are 9% faster. A later route found 21 of the 35 answers. The cost is one
core per route: the census ran 5 cases at a time on 16 cores.

### Lemma N (the peak excess is smooth before a birth)

*Let the shooting equations of a fixed structure have a branch $v(s)$, $C^1$ by
L2 and the implicit function theorem, and let limit $c$ be inactive on it. Write
its peak excess as $P_c(s) = \max_{t \in T_{\rm free}} g_c(x(t; v(s))) - o_c(s)$,
with $T_{\rm free}$ the compact set of times off $c$'s arcs.*

1. *$P_c$ is Lipschitz, and has one-sided derivatives everywhere (Danskin).*
2. *If the maximum is attained at a single interior time $t^*(s)$, with
   $\ddot g_c(t^*) < 0$ (L1), then $P_c$ is $C^1$ near $s$, with
   $P_c'(s) = \partial_s\big[g_c(x(t; v(s)))\big]_{t = t^*} - o_c'(s)$.*

**Proof.** The map $(t, s) \mapsto g_c(x(t; v(s)))$ is $C^1$ on
$T_{\rm free} \times$ (a neighbourhood). The junction times move with $s$, but
smoothly, so reparametrize each free segment over a fixed interval. Part 1
is Danskin's theorem for a max over a compact set. For part 2: by the
implicit function theorem on $\partial_t g_c = 0$, with $\partial_t^2 g_c \ne 0$,
the maximizer $t^*(s)$ is $C^1$. Then $\frac{d}{ds} g_c(t^*(s), s) =
\partial_s g_c + \partial_t g_c \cdot t^{*\prime} = \partial_s g_c$, since
$\partial_t g_c = 0$ there. $\square$

**Consequence.** If $P_c(s_0) \le 0 < P_c(s_1)$ and part 2 holds on
$[s_0, s_1]$, then the secant root $\hat s$ of $P_c = \text{target}$ satisfies
$|\hat s - s^*| = O((s_1 - s_0)^2)$, with a constant set by $\sup |P_c''|$
when $P_c$ is $C^2$. With only part 1, the secant is still a bracketing step
(regula falsi), but no rate is claimed.

### Rule R16 (place the step at the birth)

*Observed:* a census run has about 4,100 steps that converge but overshoot a
birth ("settle failed: violation too large"), against 13,500 that succeed.
Each is followed by a halved step.

*Rule:* after such a step, take the next step at the fraction
$\theta = (\text{target} - P_0)/(P_1 - P_0)$ of the failed one, clipped to
$[0.05, 0.9]$, with the target $2 \times 10^{-4}$ of the limit's scale, and
the smallest $\theta$ over the limits that overshot (`Homotopy.locate_births`).

*Why it is sound:* only the step length changes. The step is solved, settled
and judged like any other.

*Result: on in the census.* With R1, R2 and R15 (parallel routes), 37 of 45
solve, against 35 without it. It gains rest-to-rest cases 6 (285 s; never
solved before) and 9, and loses none. 22 solve under 60 s, against 18, and the
median time is 36.3 s, against 52.0 s. The 35 common answers agree to 1e-15,
and take 23% less time in total. Overshooting steps fell from 4,126 to 1,357,
and steps in all from about 26,500 to 17,400.

### Rule R17 (step past a death), tried

*Rule:* when a step fails while a finite order-1 arc is shrinking, extrapolate
$w^2$ linearly in $s$ through the last two consistent points. Lemma E makes
$w^2$ a $C^1$ function of $s$ with a nonzero slope at a death. If the
predicted death $s^*$ falls inside the failed step, solve the structure
without the arc just past $s^*$ (`Homotopy.locate_deaths`).

*Result: neutral, off by default.* With R16 on as well, 36 of 45 solve
against 37. R17 fired 144 times. It lost goal 9 at 70%, a case that already
took 168 s, and the total time of the common cases is about the same (3%
less). Deaths are not what stops the "not converged" steps.

### Measuring the census on physical cores

Censuses up to this point ran five cases at once with three routes each: 15
processes on homebox-node-1's 8 physical cores. Two processes shared each core,
and every case ran at about half speed.

Run two cases at a time instead (6 processes), with the rules unchanged (R1, R2,
R15 with parallel routes, R16) and the riding warm start of PR #48. Then 39 of
45 solve, 26 under 60 s, with a median of 23.1 s. Goal 0 at 70% and rest 1
are gained: their routes had the right structure and only ran out of
wall-clock budget. The common answers agree to 1e-15. Unsolved: goal 2 and
goal 4 at 70%, goal 9 at 80%, and rest 4, 11 and 12.

### Rule R20 (an arc too short to solve is carried as a touch)

*Observed:* in rest 11 two `tau1_max` contacts are born about 0.003 apart in
the route parameter: a late arc on $[1.1665, 1.1679]$, and then a contact
near $t = 0.631$. Every offer of the second as an arc fails, even right at
its birth.

- MINPACK, from the census's own seed, finds the answer only now and then.
  When it does, the arc at 0.631 has zero width and a jump of
  $2.7 \times 10^{-6}$.
- `_retire` then drops it as a dead order-1 arc. The violation returns, and
  the route cycles.
- The direct transcription (comparison only) shows the contact as an atom of
  the multiplier, as it does in rest 1 and rest 4.
- As an arc at a fixed route parameter, the Jacobian's condition number is
  $4 \times 10^5$ to $5 \times 10^{10}$.
- As an order-1 touch point, the seed residual is 0.05, not 30, and the
  condition number is $4 \times 10^4$. The cheap chord solve converges in
  0.1 s to the same point.

*Lemma Q* (in the register) says the touch is not itself an extremal. With a
strictly convex Hamiltonian, a contact with a jump $\nu > 0$ needs
$\dot g = 0$ on both sides and a continuous control. That is only possible
when every control that moves $g$ is at a bound that absorbs the jump.

At rest 11's contact, joints 1–3 are at their bounds, and the free joints 4
and 5 barely move $g$ ($a_k = 0.015$ and $-0.005$). So the contact is nearly
degenerate: the touch's rate after the jump is $1.2 \times 10^{-4}$, not 0.
It stands for an arc too short to solve as one.

*Rule:* `Homotopy.collapsed_touches`.

- An order-1 arc that collapses to zero width with a positive jump becomes
  a touch point carrying that jump, as arcs of higher order already do.
- An order-1 birth is also offered as a touch at its peak, after the arc
  offers.
- A violation beside the touch grows it back into an arc with its jump as
  $\pi$, through the existing rule for touch points.
- The exact problem never sees an order-1 touch. Each one goes in as an arc
  centred on it, carrying its jump, with widths $10^{-4}$, $10^{-3}$ and
  $10^{-2}$ of the motion's time tried in turn.

*Why it is sound:* the touch exists only on the route, which is never an
answer. What the exact problem is given changes only its seed. Its
acceptance checks are unchanged.

*Result: on in the census.* With R1, R2, R15 (parallel routes) and R16, on
physical cores:

- 41 of 45 solve, against 39.
- It gains rest 11 (160 s) and rest 12 (265 s; never solved before), and
  loses none. Rest 12's final time, 1.56473, is just above the direct
  transcription's 1.56052, which enforces the limits only at its nodes.
- 27 solve under 60 s, against 26, and the median time is 26.7 s.
- The 39 common answers agree to 9e-16, and their total time is 2% lower.
- R20 collapsed an arc into a touch 10 times. In rest 11, the exact solve
  accepted the arc grown from the touch.

Every case still unsolved (goal 2 and goal 4 at 70%, goal 9 at 80%, rest 4)
has an isolated point contact in the direct transcription, except goal 4.

### Rule R21 (split an exact solution whose multiplier rises), tried

*Observed:* with R20, goal 9 at 80% and goal 4 at 70% reach the end of the
offsets, and the exact solve converges (residual about $10^{-12}$, no
violations). It is rejected because an order-1 multiplier rises (Lemma H).

- In goal 9 the rise is $2.6 \times 10^{-5}$ on a `tau5_max` arc whose
  $\mu$ reaches $3.7 \times 10^{-4}$. The direct transcription (comparison
  only) has two `tau5_max` arcs there, $[0.822, 0.827]$ and $[0.864, 0.926]$.
- The route's own R5 uses the same threshold. It did not split the arc,
  because the route's solution, at its looser tolerance, does not show the
  rise.

*Rule:* when the exact solve is rejected for a rising multiplier, split the
exact solution's arc where it rises (R5's split). Solve the split on the route
at the end of the offsets, and settle it there. A violation the split causes
is brought in as after any structure change: there is no step to shorten.
Then solve exactly again, at most two levels deep (`Homotopy.exact_splits`).

*Why it is sound:* every structure it produces is solved and judged by the
same exact checks.

*Result: no gain, off by default.*

- **Goal 9 at 80%:** the split solves on the route ($2.4 \times 10^{-7}$),
  and the split exact problem solves with no rise left. But it violates
  `tau6_min` by $1.9 \times 10^{-2}$ of its scale. Widening the `tau6_min`
  arc over the violation does not converge (residual 1.7).
- **Goal 4 at 70%:** the split leaves a rise of 0.815 on two `tau6_min`
  arcs whose $\mu$ goes negative ($-0.67$). That structure is wrong in a way
  no split can fix.

So the route arrives at the wrong arrangement of arcs, and a single
discrete change at the end is too far from the right one.

### Rule R22 (stop at each change of the held set)

*Observed:* on a boundary arc ridden at run time, the riding program's held
set changes where a free jerk reaches its bound or a held jerk's multiplier
changes sign. The flow has a kink there. DOP853 steps across it with many
rejected steps, and the STM it carries is only as good as the step that
straddles the kink.

*Rule:* fix the held set between events. Then the riding law is one linear
solve per stage, with no active-set search. One guard per control (the
distance to the nearer bound for a free jerk, the bound's multiplier for a
held one) stops the step at each change, where the held set is flipped. A
change the riding cannot follow, or more than 64 on an arc, falls back to the
integration without events (`MultiArcShooting.held_events`, on by default).

*Why it is sound:* Lemma R. The flow is continuous at each change, so the
state and the STM carry through unchanged. The rule changes how the same
flow is integrated, not which answer is accepted.

*Result (census on 8 physical cores, against #52):*

- 43/45 solved, against 41/45. Goal 2 at 70% and rest 4 are gained.
- 747 s against 2409 s on the 41 common cases, 3.2× faster. 36 cases are
  faster and none slower. The median is 9.6 s against 22.8 s.
- Final times agree to $10^{-8}$ on every common case but one. Rest 9 now
  solves on the forward route in 98 s, at $t_f = 1.370023$, where the
  reversed route (R4) found $1.368515$ in 521 s: a different extremal, 0.11%
  slower. The route accepts the first extremal that passes its checks.

### Rule R23 (free arcs of integrator chains in closed form)

*Observed:* on the arm, free arcs took 30–40% of the integration time (rest 4:
43 s of 107 s), though their equations are six decoupled triple integrators.
The torque limits act only on boundary arcs.

*Rule:* where the problem is certified as integrator chains (each state's rate
a state or a control, one control per chain, a cost of the controls alone,
quadratic and uncoupled, box bounds, no penalty: `IntegratorChains.certify`),
a free arc's flow, STM and cost are evaluated in closed form instead of
integrated. The costates are polynomials, the control a clipped polynomial
whose bound crossings are polynomial roots, and the states their exact
integrals (`closed_form.py`, compiled from `chains_template.pyx`;
`MultiArcShooting.closed_free`, on by default).

*Why it is sound:* Lemma T. It is the same flow without integration error, so
no answer is accepted on its account.

*Result:*

- **Per arc on the arm:** the STM agrees with DOP853 at $10^{-12}$ to
  $8 \times 10^{-11}$, the integrator's own error, and $z$ to $2 \times 10^{-14}$.
  It costs 34 µs with the STM against 4.0 ms integrated, and 30 µs without
  against 0.41 ms.
- **Census** (R1 R2 R15 R16 R20, 300 s, 2 workers, parallel routes), against
  main with R22: still 43/45; 755 s against 958 s on the 43 solved (1.27×),
  37 faster; the median 8.7 s against 9.6 s. Final times agree to $10^{-8}$
  on every case. Goal 2 at 70% drops below 60 s (55 s); rest 4 goes from
  142 s to 102 s, rest 9 from 98 s to 66 s.
- The route's path changes slightly: the exact STM moves the chord's
  accept-or-reject decisions. So single cases can be slower (six were, by at
  most 0.4 s); only the totals compare.

### Rule R24 (the secondary weight as a route parameter)

*Observed:* on rest 4, 148 of the route's 400 path steps failed with "not
converged", and failed chord calls took 38 s of the 81 s solve. Every path step
runs at the stated weight: the arm's jerks are not bang-bang as solved,
because the default policy adds the secondary cost
$w \sum_i (j_i / s_i)^2 / 2$ with $w = 0.01$. So there is no smoothing
homotopy, and the jerk law $\operatorname{clip}(-s_i^2 \lambda_{a,i} / w)$
has slope $9 \times 10^4$ in $\lambda_a$. Tracing every chord call showed:

- the full Newton step fails to contract even on steps that end up
  converging (median $\|R(v + \Delta v)\| / \|R(v)\| = 1.7$), with
  $\|\Delta v\| \approx 0.01$;
- along a failed step, $\|R(v + \alpha \Delta v) - (1 - \alpha) R(v)\|$
  grows as $\alpha^2$, smooth rather than kinked, concentrated in R7's rate
  rows (they read the arriving control) and in the end states;
- the failures fall with the weight. Solving rest 4 as if $w$ were 0.03 or
  0.1 (other problems, other answers) gives 77 of 217 and 15 of 51 failed
  steps, in 41 s and 10 s.

*Rule:* the secondary weight is a runtime parameter of the kernels
(`w_secondary`, after the problem's own parameters). With
`Homotopy.weight_start = k`, the active-set route:
1. carries the unconstrained solution from $w$ to $k w$ by Newton steps in
   $\log w$;
2. brings the limits in there;
3. walks the weight back down to $w$ with the structure following (the same
   `advance` and `settle` as for the offsets);
4. solves the exact problem at $w$.

*Why it is sound:* Lemma W(1). Every problem on the path is one of the
family, and the answer is the exact solve at the stated weight, judged by
the same checks. W(2) explains why it helps: the curvature of the flow is
$O(1/w)$, so Newton's radius is $O(w)$.

*Result* (R1 R2 R15 R16 R20, 300 s, 2 workers, parallel routes, against main):

- **R24 off, on the new kernels:** the weight is a loaded parameter instead
  of a folded constant, so rounding moves by an ulp. On the six cases
  checked, times match main (3.8, 10.7, 18.8, 17.0, 2.7, 2.9 s against 3.8,
  10.4, 21.7, 16.3, 3.1, 2.7 s), with the same answers.
- **All three routes at $10w$, slow cases only:**

  | Case | main | R24 |
  |---|---|---|
  | rest 4 | 107.6 s | 33.1 s |
  | rest 9 | 59.0 s | 20.4 s |
  | goal 0 at 70% | 68.3 s | 36.5 s |
  | goal 9 at 80% | unsolved | 183 s |
  | goal 2 at 70% | 55.0 s | 256 s |
  | goal 4 at 70% | unsolved | unsolved |

  At $3w$: goal 9 at 80% takes 17 s, goal 2 at 70% 115 s, rest 4 81 s.
- **Rest 9** now reaches $t_f = 1.368515$, the better extremal, which the
  forward route had lost with R22 ($1.370023$).
- **Routes at $10w$, $3w$ and $w$** (in place of the other offset draw and
  sequential births): 42/45. Goal 5 at 80% and 70% fail, because the
  seed-1 route that solves them in main was dropped. The members are
  complementary.
- **The three routes of main plus a fourth at $10w$** (8 processes): 44/45,
  with goal 9 at 80% solved (220 s). 644 s against 738 s on the 43 both
  solve (1.15×); the median 7.5 s against 8.8 s; 40 under 60 s against 41.
- **All three routes at $10w$:** 43/45 and 41 under 60 s, but not the same
  41 as main.
  - Rest 4 (33 s) and goal 0 at 70% drop below 60 s, and goal 9 at 80% is
    solved (162 s).
  - Goal 9 at 70% fails, and goal 2 at 70% takes 329 s.
  - No variant dominates. Which weight a case needs is route chaos, like the
    offset draw, so the weight belongs in the portfolio beside the draw.

- **Six routes: main's three at $w$ and the same three at $10w$** (one case
  at a time, 6 processes): 44/45 and 42 under 60 s, 536 s against 738 s on
  the 43 that main also solves (1.38×), median 7.9 s.

  | Case | main | four routes | six routes |
  |---|---|---|---|
  | rest 4 | 107.6 s | 108.4 s | 33.2 s |
  | rest 9 | 59.0 s | 17.7 s | 19.5 s |
  | goal 0 at 70% | 68.3 s | 72.0 s | 36.3 s |
  | goal 2 at 70% | 55.0 s | 72.6 s | 67.1 s |
  | goal 9 at 70% | 28.7 s | 40.4 s | 54.6 s |
  | goal 9 at 90% | 43.0 s | 9.9 s | 9.8 s |
  | goal 9 at 80% | unsolved | 219.8 s | 217.0 s |
  | goal 4 at 70% | unsolved | unsolved | unsolved |

*Deployed:* on in R15's portfolio as main's three routes again at $10w$ (the
six-route variant above: `{"weight_start": 10}` alone and with each of the
other two members, in `DEFAULT_PORTFOLIO`). The member is dropped when
the policy has no numeric weight, or when the problem has bang-bang smoothing
(the route walks $\epsilon$ there, and the member would repeat the first route).

### Rule R25 (place the step at the split)

*Observed:* R5 looks at the rise of $\mu$ on every consistent point inside
`settle`, but it acts wherever the step lands. A step that crosses a split
(V4's event E3, in `route_family.md`) lands past it, and R5 splits the arc far
from the event. The split it offers either fails to solve or is solved and
rejected at the end ("mu negative", "mu rises"). In the census below, R5
fired 597 times.

*Rule:* on each step's own solution, before any repair, compute the density
$\eta = -\dot\mu$ of every order-1 arc (`Homotopy._densities`). Act when both
of these hold:
- on some arc, $\eta < 0$ at its minimum $t^*$, and the continuous part of
  $\mu$ rises by more than $10^{-3} \max(1, |\mu|)$ along it;
- at the step's start, an arc of the same limit covered $t^*$ with
  $\eta(t^*) > 0$.

Then the event is inside the step. Take the next step at the fraction
$\theta = (\eta_0 - 0.1\, \eta_1) / (\eta_0 - \eta_1)$ of it, clipped to
$[0.05, 0.9]$ (`Homotopy.locate_splits`). Here $\eta_0$ and $\eta_1$ are
$\eta(t^*)$ at the two ends, so $\theta$ is just past the zero, where R5's
split is small. The continuous part of the rise leaves out a jump of $\mu$
between two samples at one time (another limit's junction), which is not the
density's doing.

*Why it is sound:* only the step length changes. The step is solved, settled
and judged like any other. Effectiveness is V2: on a fixed structure,
$\eta(t^*)$ at a fixed time is $C^1$ in the route parameter. So the secant is
within $O(ds^2)$ of the zero, as R16's is for a birth (Lemma N). The minimum
of $\eta$ over the arc is only Lipschitz, and $\eta(t^*) \ge \min \eta$. So
the secant is an upper bound on the event's position, and the next step
checks it again.

*Not what it was built for:* goal 4 at 70% fails on its first route for a
different reason. Along that route the density is positive all the way, and
the failure is a birth just beside an exit:
- `tau6_min` leaves at $t = 0.9497$, and its $g$ crosses the limit again
  near $t = 0.954$;
- the violation is under `settle`'s threshold, so it is brought in on the
  spot, and the arc grows over it;
- $\mu$ then rises inside that arc, R5 splits it, the violation returns, and
  so on until R5's three tries are spent.

*Result* (six routes, R1 R2 R16 R20, 300 s, one case at a time, against main):
45/45 solved and 44 under 60 s, against 44/45 and 42.
- **Common cases:** the 44 both solve take 693 s against 753 s, the
  answers agree to $4 \times 10^{-16}$, and the median is 8.5 s against 7.9 s.
- **Firings:** R25 fired 128 times, in five cases:

  | Case | main | R25 | R25 fired |
  |---|---|---|---|
  | goal 4 at 70% | unsolved (108 s) | 45.4 s | 21 |
  | goal 2 at 70% | 67.1 s | 39.9 s | 7 |
  | goal 9 at 70% | 54.6 s | 39.0 s | 2 |
  | goal 9 at 80% | 217.0 s | 185.7 s | 90 |
  | goal 0 at 80% | 19.9 s | 19.3 s | 8 |

  Two repeat runs of the first three give the same answers, within 1 s.
- **Goal 4 at 70%:** solved in another route of the portfolio, with
  $t_f = 1.167506$.
- **Still over 60 s:** only goal 9 at 80%. On it, 291 steps fail with
  "structure changes did not settle".

### Locating every event, tried

*Idea:* `settle` locates an event inside a step only when it is large: a peak
excess above $10^{-3}$ of the limit's scale (R16), or a rise of $\mu$ above
$10^{-3}$ (R25). A smaller one is repaired where the step landed. The event
follower of `route_family.md` locates every event. The threshold is now a
parameter, `Homotopy.locate_tol`, with R16's target at a fifth of it.

*Result* (six routes, slow cases): no threshold is better on every case.

| Case | $10^{-3}$ | $10^{-4}$ | $10^{-5}$ |
|---|---|---|---|
| goal 9 at 80% | 185.7 s | 26.7 s | 20.8 s |
| goal 4 at 70% | 45.4 s | unsolved | unsolved |
| goal 2 at 70% | 39.9 s | 249.5 s | 90.0 s |
| goal 0 at 70% | 38.3 s | 46.7 s | 77.8 s |
| goal 9 at 70% | 39.0 s | 42.6 s | 49.6 s |

Mixing thresholds across the portfolio's routes did no better: one case
always went over 60 s. The default stays $10^{-3}$.

*What the routes are sensitive to:* the 10w route alone solves goal 9 at
80% in 31 s when the case's torque limits come from a free solve also run
at 10w. In the portfolio, where those limits come from the free solve at
w, it fails. The two sets of limits differ only by rounding: the final
times differ by $6 \times 10^{-15}$. A route whose outcome changes with
rounding is deciding something by a threshold. The trace shows what that
was (Rule R26).

### Rule R26 (a violation beside an arc is a birth)

*Observed:* on goal 9 at 80% and goal 4 at 70%, the first route fails the
same way:
- a step brings in a small violation of a limit just past the exit of that
  limit's arc (0.004 of $T$ away on goal 4);
- `_activate` treats any violation within $0.05\,T$ of an arc of its limit
  as part of that arc, and widens the arc over it;
- the widened arc covers a stretch where the limit was strictly below its
  level, and its density is negative there: $\mu$ falls below 0 and rises
  ($-1.9 \times 10^{-5}$ and $3.3 \times 10^{-5}$ on goal 9's `tau2_min`);
- R5's split of it does not solve (residuals 35 and 27), and the route
  carries the bad arc to the end, where the exact solve rejects it.

*Rule:* a violation near an arc of its limit is born as a new arc
(Lemma E, event E1) when the limit is strictly below its level somewhere in
the free time between them: below $-10^{-6}$ of the limit's reach on the
free stretches (`Homotopy.separate_births`). Only when the limit stays at
its level between them does the arc widen over the violation, or do two
arcs merge across it.

*Why it is sound:* it changes only which structure is offered. Every
structure is solved, settled and judged as before. Effectiveness is
Lemma E: a peak separated from the arc by a stretch where $g < 0$ is an
interior contact, and its arc is born there with small width. Widening the
old arc over it instead holds the limit at its level across a stretch where
the nearby extremal is strictly inside it. So the widened structure is not
that extremal's, and on the traced cases its density is negative there.

*Result* (six routes, R1 R2 R16 R20 R25, 300 s, one case at a time,
against R25 alone): **45/45 solved, all under 60 s**, against 44.
- **Totals:** 579 s against 739 s on all 45. The median is 8.4 s against
  8.5 s, and the slowest is 46.9 s against 185.7 s.
- **Answers:** they agree to $2 \times 10^{-16}$.
- **Goal 9 at 80%:** 20.2 s, against 185.7 s.
- **Firings:** R26 fires rarely (goal 4 at 70%: 9 times; goal 9 at 80%:
  2; goal 2 and goal 0 at 70%: once each). Two repeat runs of the five
  slowest cases give the same answers, within 1 s.
- **On the first route alone** (traced, with the lemma oracles on), R26 turns
  three of four slow cases from failures into solves. Goal 4 at 70%, goal 9
  at 80% and goal 0 at 70% failed before; goal 2 at 70% took 39 s before.

  | Case | first route, R26 off | first route, R26 on |
  |---|---|---|
  | goal 4 at 70% | unsolved | 35.5 s |
  | goal 9 at 80% | unsolved | 15.3 s |
  | goal 2 at 70% | 39.4 s | 293.8 s |
  | goal 0 at 70% | unsolved | 442.9 s |

- **The first route alone on all 45** (R1 R2 R16 R20 R25 R26, six cases at
  a time, with the trajectory caches of the next PR): 39/45 solved, 33 under
  60 s, median 14.0 s.
  - Unsolved: goal 3 at 90%, 80% and 70%, goal 5 at 80% and 70%, and goal 7
    at 70%.
  - So the portfolio is still needed. Its other routes are what solve those
    six.

### Lemma E3 (the split's gap) and Rule R27

*Lemma E3* (in `lemma_register.md`, leading order derived) is Lemma E with
$g$ and $\eta$ exchanged. Released on a gap $[a, b]$, the limit moves at
$\dot g = D(\mu - \mu(a))$, where $D$ is Lemma P's
$a_F^\top R_F^{-1} a_F$, generalized to several limits ridden at once. It
must return to its level tangentially. So:
- $\mu(a) = \mu(b)$;
- $\int_a^b D(\mu - \mu(a)) = 0$.

For a smooth dip of $\eta$, this centres the gap on the minimum with
$h^2 = -6\eta_{\min}/\ddot\eta$. That is $\sqrt 3$ times the stretch where
$\eta < 0$, so R5's seed, which spans that stretch, is too narrow. Where
$\eta$ is negative on one window of a held set (Lemma P), the gap is that
window, slightly widened.

*Check* (every split R5 offered on the first route of all 45 cases):
- **Smooth dips** (goal 4 at 80% and at 70%): solved to 0.98 and 1.00 of
  $h$, centred on the minimum.
- **Held-set windows** (goal 2 at 70%, 38 splits, curvature about
  $6\times 10^5$ in a quadratic fit): R5's splits solve to a gap of zero
  width, away from the window. The moment seed solves to the gap
  $[0.37734, 0.38009]$ (seed $[0.37732, 0.38005]$). There $g < 0$, and the
  rise of $\mu$ falls from $1.8\times 10^{-4}$ to $10^{-12}$.

*Rule R27* (`Homotopy.split_normal_form`): when R5 fires, offer first the
gap of the moment conditions, with $D$ from the rate coefficients
(`Homotopy._split_weights`). Then come R5's own seeds.

*First version, tried:* a quadratic fit about the density's minimum. It
fits a held-set window as a spike and sends the split astray. Goal 2 at 70%
took 321 s (against 38 s), and R5 fired 2,613 times. Restricting it to
interior controls made it fire once in the census.

*Result of the moment version* (six routes, against the speed PR's census):
45/45 under 60 s.
- **Totals:** 478.0 s against 493.9 s. The median is 7.1 s against 6.9 s.
- **Answers:** they agree to $2\times 10^{-13}$.
- **Gains:** goal 9 at 70% takes 18.8 s against 35.4 s, and goal 0 at 80%
  14.2 s against 19.1 s.
- **Firings:** R27 fires 16 times. R11's merges fall from 111 to 64, and
  R20's collapses from 5 to 1.

*On the first route alone:* 38/45 against 39/45, a different mix.
- Goal 3 at 70% is now solved (176.7 s), and goal 2 at 70% takes 231.4 s
  against 422.1 s.
- Goal 0 at 70% and rest 12 are lost. On rest 12, `tau1_max` splits, its gap
  closes, and it splits again (R27 fires 161 times). This is O2's case, where
  the arc's own jerk saturates on it. O2's open lemma is what decides it:
  whether such an arc must exit before the saturation.

R27 is on in the census.

### Lemma U (an arc leaves before its own control saturates) and Rules R28, R29, tried

*Observed (O2, rest 12):* on the first route, `tau1_max` splits, its gap
closes, and it splits again, with R27 firing 161 times.
- **Where the rise starts:** every rise of $\mu$ on the arc starts exactly
  where joint 1's own jerk joins the held set.
- **What the answer does:** the arc leaves at 0.41610, before that jerk
  saturates at 0.42511.
- **Why the route cycles:** `settle` takes the first split option that
  converges. Here that is R27's seed, which converges to a split with a gap
  $3 \times 10^{-5}$ wide at the end of the rise. The rise is still there,
  so R5 fires again.

*Lemma U* (`lemma_register.md`, proved relative to P and J). Where the
density turns negative at the moment the limit's own control saturates, the
extremal cannot ride through (H). It cannot leave at that moment unless
$\mu = 0$ there (J: no jump at an order-1 exit). So the piece leaves before
it. Trims on the route bear this out: leaving 2% or 5% earlier, the piece
solves to an exit at 0.34525, before the saturation at 0.3458, with no rise
and no violation. Started exactly at the saturation, it converges back to
the rising solution.

*Rule R28* (`Homotopy.split_must_resolve`, from V4): a split R5 or R27
offers is taken only if the limit's multiplier no longer rises (to the final
check's tolerance) and no piece or gap of it has zero width. Otherwise the
next option is tried.

*Rule R29* (`Homotopy.exit_before_saturation`, from Lemma U): when the rise
starts within three samples of the limit's own control (the largest
$|a_k|$) joining the held set, it offers first the piece leaving 2% before
that moment. If the rise ends inside the arc, the gap runs to the rise's
end; if the rise reaches the exit, the arc is trimmed.

*Result: mixed, off by default.*

| | R27 | + R28 | + R28 + R29 |
|---|---|---|---|
| first route alone: solved | 38/45 | 39/45 | 39/45 |
| six routes: under 60 s | 45 | 44 | 44 |
| six routes: total | 478.0 s | 592.5 s | 596.5 s |

On the first route alone:
- **Gains:** rest 12 (26 s; it failed with R27 alone) and goal 7 at 70%
  (565 s; never solved by this route before). Goal 2 at 70% takes 184 s
  against 231 s.
- **Loss:** goal 4 at 70%. Its route had relied on a degenerate split of
  `tau6_min` (a gap of zero width, with $\mu$ rising by 0.8), which R28 now
  refuses; without it the route stalls.

With six routes, goal 2 at 70% takes about 149 s against 37 s.

So refusing inconsistent repairs is right by V4, but some routes make
progress only through them. The follower's discipline pays only once the
events those repairs stand in for have transitions of their own. Here that
is the `tau6_min` birth beside an exit on goal 4 at 70%.

### The order and jump monitors on the route: Rules R30, R31, tried

*Observed (goal 3 at 80% and 90%, goal 5 at 80% and 70%, on the first
route):* each of these routes stalls. The goal 3 at 80% stall goes like this:
- **What the route accepted:** a state with two exits crossed. In the
  structure's order `tau2_min` leaves before `tau3_min`. In the solution it
  leaves at 0.29592, after `tau3_min` at 0.29448. So one segment runs
  backward in time.
- **How it got in:** `settle` returned the state through its shortcut for a
  tangential junction (the same violation, not shrinking, at the path
  tolerance). That shortcut never asks whether the junctions are in order.
- **What else was wrong:** in the same state `tau2_min`'s multiplier was
  negative along its whole arc, with $\pi = -1.7 \times 10^{-3}$, and
  `tau6_min`'s rose by 0.11.
- **What the route did next:** R25 fired 68 times, chasing a negative
  density that was already there before the step. The route crawled from
  offsets 1.76 to 1.66 in a minute.

The answer, found by the portfolio, has `tau2_min` leaving at 0.2306,
before `tau6_min` enters. In it `tau1_min`, `tau2_min` and `tau3_min` all
carry multipliers below $3 \times 10^{-4}$, so their signs flip easily.

*Rule R30* (`Homotopy.watch_order`, from V1: acceptance is the monitors,
and order is one of them). `settle` does not accept a solution whose
junctions are out of the structure's order:
- **A crossing of more than `locate_tol` of $T$** in one step: the step is
  shortened, as R16 and R25 do for births and splits.
- **A smaller one:** the events are re-sorted (E5's swap).
- **A swap that crosses back** a third time has not restored the monitor
  (V4), so the step fails.

With R30, goal 3 at 80% and at 90% solve in 9 s, and goal 5 at 80% in 141 s.

*Goal 5 at 70% still cycled.* The exits of `tau2_min` and `tau3_min`
swapped back and forth hundreds of times. The cause was one step earlier:
- At the birth of `tau3_min` beside `tau2_min`'s exit, `tau2_min`'s $\pi$
  went from $+5.5 \times 10^{-5}$ to $-5.4 \times 10^{-4}$.
- `_retire` removes arcs whose width reaches zero, not those whose
  multiplier does, so the arc stayed. Its $\pi$ fell to
  $-2.4 \times 10^{-2}$.

*Lemma H′* (`lemma_register.md`, proved relative to S1). When an order-1
arc's $\pi$ reaches zero, $\mu \equiv 0$ on it (H gives $0 \le \mu \le \pi$).
The riding control is then the free one, so the extremal also solves the
structure without the arc. Past that point keeping the arc violates H. So
the event is a death by multiplier (E2″).

*Rule R31* (`Homotopy.retire_negative`, from Lemma H′): `_retire` removes
an order-1 arc whose $\pi < -10^{-7}$.

*Result: mixed, off by default.*

| | R27 | + R30 | + R30 + R31 |
|---|---|---|---|
| first route alone: solved | 38/45 | 38/45 | 40/45 |
| six routes: under 60 s | 45 | | 44 |
| six routes: total | 478.0 s | | 528.4 s |
| six routes: median | 7.1 s | | 6.9 s |

On the first route alone, with R30 and R31:
- **Gains:** goal 3 at 80% and 90% (9 s each, as traced), goal 5 at 80%
  (66 s), goal 5 at 70% (192 s) and goal 7 at 70% (516 s).
- **Losses:**
  - Goal 3 at 70% (177 s before).
  - Goal 8 at 70%. Its forward route fails either way, and its reversed
    route (R4) solved it at 58 s without the rules. R30 fired once on that
    reversed route and steered it elsewhere.
  - Goal 4 at 70%. Its reversed route drifts from the baseline's from
    offsets 1.4 on, then stalls at 0.14 with R25 firing on a split.

With six routes, goal 4 at 70% takes 82.3 s against 41.8 s, and goal 2 at
70% 46.7 s against 37.1 s.

So the first route alone gains two, and the six-route census loses one
case's margin to route chaos rather than to a wrong transition. Both
monitors are V1's; what remains is the transitions those routes now reach.

### Repairs at the exact solve: Rule R21 again, and Rule R32

With R30 and R31 the first route alone still failed goal 4 and goal 8 at
70%. Both forward routes end their offsets with nearly the answer's
structure. The exact solve converges and is rejected, each time for a
monitor the route did not see:

- **Goal 8 at 70%: "mu rises" on `tau5_min`** (0.0033 against a peak of
  0.33).
  - The route carries `tau1_max` as an order-1 touch inside `tau5_min`'s
    arc (R20). The exact solve widens it into a short arc, and while
    `tau1_max` is held, `tau5_min`'s density is negative (Lemma P).
  - The answer is the route's structure with `tau5_min` split around that
    window: [0.56492, 0.57155], then `tau1_max` on [0.57229, 0.57326], then
    [0.57399, 0.70809]. That is Lemma E3's held-set-window case.
- **Goal 4 at 70%: `tau6_min` violated off its arcs by $1.4 \times 10^{-7}$.**
  - The route's threshold is $10^{-6}$ of the limit's scale, and the exact
    check's is $10^{-7}$, so the route never saw this violation.
  - The answer has a second short `tau6_min` arc, [0.96094, 0.96405], just
    after the first.

*Rule R21* (exact splits, tried before with R5's split, no gain then). With
R27 on, `_split_rising` offers Lemma E3's gap first. Goal 8 at 70% then
solves on the forward route in 21 s.

*Rule R32* (`Homotopy.exact_births`, from Lemma E and V1). When the exact
solution is rejected for a violation off its arcs:
- The violation is found at the exact check's threshold.
- It is brought in as the route brings in any violation (`_activate`:
  birth, widening, R26).
- Each option goes to the exact solve directly, with order-1 touches as
  R20's short arcs.

The options do not go through the route: the route's tolerance (about
$10^{-6}$) cannot resolve a violation this small, and there it retires the
newborn arc again. Goal 4 at 70% then solves on the forward route in 11 s.

*Result (first route alone, with R30 and R31 as well):*

| | R27 | + R30 + R31 | + R21 + R32 |
|---|---|---|---|
| solved | 38/45 | 40/45 | 42/45 |
| under 60 s | | 33 | 36 |
| total | | 3264.9 s | 2516.5 s |

With six routes and the same rules, all 45 solve under 60 s: total 494.1 s
against 478.0 s, median 7.3 s, slowest 52.9 s (goal 2 at 70%). Goal 4 at 70%
takes 15.9 s, against 82.3 s with R30 and R31 alone. The rules stay off by
default until the first route alone solves all 45 under 60 s.

The cases left on the first route:
- **Failing:** goal 0 at 70%, goal 3 at 70% and rest 12.
- **Over 60 s:** goal 5 at 70%, goal 2 at 70%, rest 4, rest 9, rest 2
  (through R4) and goal 7 at 70% (through R4).

### Which arc to split, and the split's seed: Rules R33, R34

With R21 and R32 the first route alone failed goal 3 at 70%. A trace of
`tau2_min` along the route shows why:

- Before `tau3_min` is born, `tau2_min` rides [0.137, 0.318] with
  $\pi = 2.2 \times 10^{-4}$ and $\mu \ge 0$ on the whole arc.
- `tau3_min` is then born inside it, at [0.289, 0.293]. While `tau3_min` is
  held, `tau2_min`'s density is negative (Lemma P). So $\mu$ before the window
  falls by about the window's mass, which grows with its width. At a window
  of $7.5 \times 10^{-4}$, $\mu$ before it is $-1.5 \times 10^{-4}$, and a step
  later $\pi$ itself is negative.
- R31 then retires `tau2_min`. The structure without it does not converge
  (residual 63). Neither does `tau2_min` with its entry moved to where $\mu$
  turns positive (residuals $10^2$ to $10^7$). The arc's tail, where
  $\mu > 0$, is still wanted.

At the first state with `tau3_min` held and $\pi > 0$, R5 sees two rises on
the window:
- `tau2_min`: $\mu$ from $-1.5 \times 10^{-4}$ to $4.0 \times 10^{-5}$;
- `tau6_min`: a larger absolute rise, on a multiplier of about 3.7.

R5 splits the largest absolute rise, so it splits `tau6_min`. That solves,
but `tau2_min` stays negative, and from there no split of `tau2_min` solves
(residuals 31 to 98).

Splitting `tau2_min` instead, from the same state, solves to a structure
whose monitors are all non-negative:
- `tau2_min` on [0.13252, 0.28732] with $\pi = 2.0 \times 10^{-4}$;
- `tau3_min` on [0.28849, 0.29173];
- `tau2_min` again on [0.29294, 0.32173];
- `tau6_min` one arc again (its rise was the one induced);
- no violation off the arcs.

The answer has the same exchange near the end: `tau2_min` leaves at 0.218,
just before `tau3_min` enters at 0.220.

*Rule R33* (`Homotopy.relative_rise`). R5 splits the arc whose rise is
largest relative to the arc's largest $|\mu|$. Scaling a limit's function
$g_c \to k g_c$ scales $\mu_c \to \mu_c / k$ and changes nothing else. So
only a scale-free measure can compare rises on different limits. It is an
order of trial; acceptance is unchanged.

*Rule R34* (`Homotopy.wide_gaps`, a seed). From that state, a `tau2_min`
split solves or fails by the first piece's seed, at the same gap:

| gap (window $w$ widened on each side by) | first piece's $\pi$: R5's $\pi - \mu_{\rm lo}$ | first piece's $\pi$: unchanged |
|---|---|---|
| $0.5\,w$ | fails (96) | solves ($1.6 \times 10^{-8}$) |
| $2\,w$ | fails (90) | solves ($1.0 \times 10^{-7}$) |

R5's value is the right answer to leading order: the solution's $\pi$ is
$2.03 \times 10^{-4}$ against R5's $2.09 \times 10^{-4}$. But a different $\pi$
moves the costate along the whole first piece. The unchanged $\pi$ reproduces
the current solution up to the gap, so Newton only has to close the gap.
With the gap R5 offers, 2% wider than the window, the seed fails whatever
$\pi$ is ($10^6$ and up): the junctions nearly coincide with the held
limit's (Lemma D). R34 therefore also offers the split seeded by
continuity, with gaps wider by $0.5\,w$ and $2\,w$ on each side.

*Result.* Goal 3 at 70% on the first route alone:

| Rules on | Outcome |
|---|---|
| R33 and R34 | solves forward in 12 s (18 s in the census) |
| R33 alone | solves only through R4, at 311 s |
| R34 alone | fails: `tau6_min` is split first |

*Census, first route alone (R30, R31, R21, R32 on as well):*

| | + R21 + R32 | + R33 + R34 |
|---|---|---|
| solved | 42/45 | 42/45 |
| under 60 s | 36 | 36 |
| total | 3556.2 s | 3252.7 s |

The final times agree where both solve (to $4 \times 10^{-16}$).
- **Gained:** goal 3 at 70% (failed, now 18 s).
- **Lost:** goal 7 at 70%. It solved at 510 s through R4, near the cap; now
  it runs out of time at 595 s.
- **Slower:** goal 5 at 80%, 59 s to 65 s (R33 fires there).

Cases where neither rule fires take the same route; their times moved by up
to about 20% between the two runs.

With six routes and the same rules, all 45 solve under 60 s: total 464.5 s
against 494.1 s, median 6.9 s, slowest 47.0 s (goal 2 at 70%). The rules
stay off by default until the first route alone solves all 45 under 60 s.

### Order-2 touches that have become arcs: Rule R35, tried

The census's rate-limited cases (`vgoal`, `vrest`) add order-2 limits: with jerk
control a rate limit $w_i \le w_{\max,i}$ has $\ddot g = \pm j_i$. On the first route
alone, 9 of the 45 solve within 60 s. In goal 0 at 80% a converged structure carries a
touch of `w4_min` at 0.58008 where the limit's curvature is $\ddot g = +9.1$ (joint 4's
jerk; its bound is 30), with the limit violated on both sides. A touch is a local
maximum of $g$: $g(t^*) = 0$, $\dot g(t^*) = 0$, and with the continuous control of the
jerk-cost policy $\ddot g(t^*) \le 0$. A touch with $\ddot g > 0$ has become an arc, as
in Bryson–Denham at $\ell = 1/6$ (Bonnans & Hermant 2008: a touch becomes an arc).

`_activate` offered the violation three structures, in order:

| structure | result |
|---|---|
| one arc over the violated span | does not propagate ($7 \times 10^6$) |
| the touch split into two touches | solves; touches at 0.57797 and 0.58218, the jump halved, $\ddot g = +0.05$ at both, the limit up to $2.8 \times 10^{-8}$ above its level between them |
| a short arc about the touch, the touch's jump as its entry jump | solves: [0.57798, 0.58219], multiplier $2.3 \times 10^{-4}$ falling to 0 |

The route took the second: an arc in disguise, its violation below the route's
threshold. The short arc solves from every width tried ($2 \times 10^{-4}$ to
$2 \times 10^{-2}$ of $T$) with the jump on its entry; seeds without the jump do not.

*Rule R35* (`Homotopy.touch_curvature`). The curvature at each order-2 touch is the
slope of $\dot g$ (exact at each state: at order 2 the control does not enter it)
within $0.002\,T$ of the touch. A touch whose curvature is positive (above $10^{-3}$
of $\max|\dot g|/T$) becomes a short arc about it, inside its neighbours' events
(half-width at most a quarter of the distance to the nearest junction, Lemma D), with
the touch's jump as its entry jump. A violation beside such a touch is offered the
short arc first.

*Result.* Rate cases on the first route alone, 60 s cap (R1, R2, R16, R20, R21, R25,
R26, R27, R30–R34 on): 9 of 45 with and without R35, the same cases. R35 fires in
44 of the 45, and arcs of negative width fall from 10.8 to 5.6 thousand evaluations.
Goal 0 at 80% then gets close to the end of its offsets (to $|o| = 0.036$ in 172 s)
before it stalls on another event: the exit of `w3_min` (order 2) and the entry of
`tau5_max` (order 1) swap at every step (R30 fires 452 times), and R2's width steps,
made for order-1 births, predict route parameters of $-0.36$ and $0.32$.

### Order-2 multipliers that rise: Rule R36, tried

With every set of limits ridden at run time (PR #73), goal 0 at 80% reaches the end of
its offsets on the first route. Its exact solve converges (residual $6 \times 10^{-11}$,
violation $10^{-12}$) and is rejected: `mu negative`, on `w5_max` and `w2_min`. Lemma H2
says why that is right: on an order-2 arc $\mu$ is convex and ends at zero with
$\dot\mu \le 0$, so $\pi_2 = \mu(t_{\rm in}^+) \ge 0$ and $\mu$ never rises. Along the
route, $\mu$ on `w5_max` first rises inside the arc (at about $|o| = 1.25$), then dips
below zero there, and only then does $\pi_2$ turn negative. The rise of `w2_min` is
exactly the stretch where `tau5_max` is ridden with it. The direct transcription (120
intervals, for comparison only) has `w2_min` and `w5_max` held only briefly, before
`tau5_max` enters, and `w3_min` in three pieces.

*Rule R36* (`Homotopy.order2_rise`). Acceptance checks the rise at order 2 as at order
1, and R5 splits an order-2 arc where its multiplier rises (R33 picks the arc). The
second piece enters with $(-\dot\mu, \mu)$ at its entry, the jump that continues the
arc's costate there, and also with $(0, \mu)$ and $(\pi_1, \mu)$. Each piece is also
offered alone: the arc cut before the rise, or started after it. A rise within a tenth
of its stretch of an end trims the arc there. No new junction is put on the rise's own
ends, which are often another limit's junctions (Lemma D): it goes 2%, half or twice
the stretch past them.

*Result.* The plain splits do not propagate. Riding fails just after the second piece
re-enters, whatever its jump, which suggests the edge of the holdable set (R8 treats it
at order 1 only). The one-piece offers do work. On the first route alone, goal 0 at 80%
solves exactly with `w2_min` and `w5_max` as touches before `tau5_max`, as the direct
answer has them, but in 136 s. Rate cases on the first route, 60 s cap (R1, R2, R16, R20,
R21, R25, R26, R27, R30–R35 on): 14 of 45 with and without R36, the same cases. R36 fires
in three cases (goal 0 at 80% and 70%, rest 2), each of which reaches the rise only
near its cap. The two rejections for `mu negative` are gone. What these cases need now
is a faster route to the rise.

*R36's seeds, fixed (2026-11-19).* The splits with a gap never propagated because the
seed was NaN: the second piece's jump $(-\dot\mu, \mu)$ took $\dot\mu$ by a difference over
the arc's samples, and the trajectory samples each junction from both sides, so a time
repeats and the difference divides by zero. With $\dot\mu$ taken on the distinct times no
offer fails to ride. Rate cases on the first route, 60 s cap (R35, R36, R38 on): 16 of 45
before and after, the same cases; the two rejections for `mu negative` are gone.

*Lemma H2's other half, tried (not kept).* $\mu$ is convex on an order-2 arc, so the exact
$\dot\mu$ (the multiplier's gradient along the riding flow) must not fall. It falls a
great deal: by 60 to 180% of $\max|\dot\mu|$ on the long rate arcs of goals 0 and 1 at 80%,
on several arcs over one stretch, at another limit's entry (`w2_min`'s at 0.5894 in goal
0). The violation is real, but splitting the arcs that show it, over that stretch, made
the routes worse (goal 0 at 80% stops at $|o| = 1.25$ instead of reaching the end; goal 1
at 0.13): the stretch is the entering limit's junction, and it is the entry, not the
arcs beside it, that the lemma faults.

### Where the rate-limited routes stall, and coincident junctions: Rule R38, tried

On the first route alone (R35 and R36 on, 60 s cap), 29 of the 45 rate cases fail. Each
was traced, and the route after its last progress (its least $|o|$ within 1%) was
classified:

| what the route does | cases |
|---|---|
| still progressing when the time runs out | 11: goals 0, 2 at 70%; 3, 4, 5 at 80% or 70%; rests 1, 3, 9, 10, 11 |
| R30 reorders the same pair of junctions back and forth | 10: goals 1 at 80% and 70%, 4 and 5 at 90%, 9 at 70%; rests 6, 8, 12, 13, 14 |
| crawls: most solves fail, at steps near $10^{-4}$ | 5: rests 2, 4, 5; goals 3 and 7 at 70% |
| reaches the end of its offsets, the exact solve rejected | goal 0 at 80% (goals 1 at 80% and 5 at 90% also reach $|o| = 0$) |

The reorders are coincident events, which Lemma D's genericity does not cover:

- **By symmetry.** Rest 8 ends with five rate touches at one instant, 0.63184: with no
  torque arc left, the motion is time-symmetric and every rate peak falls at $T/2$.
  R30 saw crossings below $10^{-9}$ of $T$, rounding, and reordered for them.
- **Forced.** In goal 1 at 80% the entries of `tau5_min` and `w3_max` swap back and forth
  by about $7 \times 10^{-5}$ of $T$, each order's solution pushing them past each other;
  in goal 9 at 70% `tau2_min` leaves where `tau6_min` enters; in rest 2 `tau1_min` leaves
  where `w4_min` enters, the same instant to five digits at every step. A hand-over
  of this kind (likely a forced exit, Lemma K) needs one junction for the two events,
  which the shooting does not have.

*Rule R38* (`Homotopy.coincident_junctions`). Junctions of different limits that cross by
less than `coincide_tol` ($10^{-8}$) of $T$ are coincident, in either order: R30 does not
reorder them, `_retire` does not re-sort them, and acceptance takes them. Below the
junction times' own resolution the two orders give the same motion. R35's short arc no
longer takes its width from events coincident with the touch, which had made it zero
(rest 8: R35 then fired 140 times, each arc retired back to a touch).

*Result.* Rate cases on the first route, 60 s cap (R1, R2, R16, R20, R21, R25–R27,
R30–R36 on): 16 of 45 with and without R38, the same cases; R38 fires in 15, and the
common solves take 12% less time (430 s to 377 s). Rest 8, alone on the machine, now
reaches $|o| = 0.094$ with a symmetric structure (`w5_max` and `w4_min` on the same
span about $T/2$), still progressing at the cap. The forced hand-overs are untouched.

### Hand-overs: Rule R39, tried

A forced coincidence (R38's second kind) is a hand-over: one limit leaves where another
enters, and each order's solution pushes the two junctions past each other. In rest 2
`tau1_min` leaves where `w4_min` enters, the same instant to five digits over 45 s of
route; in goal 9 at 70% `tau2_min` leaves where `tau6_min` enters; in goal 1 at 80% the
entries of `tau5_min` and `w3_max` swap by about $7 \times 10^{-5}$ of $T$. The likely
mechanism is Lemma K's forced exit: past the instant, the two cannot be held together
(the edge of the holdable set), so the leaving limit's multiplier need not reach zero.

*Rule R39* (`Homotopy.handover_ties`). The two events are one junction. The earlier
event's own row (rate, multiplier or $H$ continuity, never a tangency row) gives way to
a tie, $t_{\rm next} - t = 10^{-12}\,T$: a hair apart, so the segment between them has
positive width and integrates as any other. The unknowns are unchanged. A tie is made
when R30's reorder crosses back, or when a step fails while two adjacent junctions of
different limits (neither a touch) lie within $10^{-5}$ of $T$; it holds wherever the
two events are adjacent, for the rest of the solve, and acceptance judges the result
(the exiting limit's $\mu \ge 0$, and Lemmas H and H2).

*Result.* The ties take hold: rest 2 moves past its locked pair (from $|o| = 21.9$ to
$20.2$, with `tau1_min`, `w4_min` and `w5_min` then at one instant), and goal 1 at 80%
reaches its exact solve at 16 s instead of churning at the end. That solve converges
(residual $5 \times 10^{-13}$) and is rejected by Lemma H2: $\mu$ rises by 0.8% and
1.5% of its largest value on `w5_max` and `w2_max`. Rate cases on the first route, 60 s
cap (R38 on as well): 16 of 45 with and without R39, the same cases; R39 fires in 8.

*Why it cannot be right (found after the census).* Every row a tie can replace is a
necessary condition. Goal 1 at 80%'s tie replaced the rate row of `tau5_min`'s entry
(the arrival is tangential), and its rejected answer has $\mu$ of `w5_max` and `w2_max`
jumping up by about 1% at that instant, 0.65902. Lemma H2 makes an order-2
multiplier continuous ($\mu(t) = \nu_{\rm out}(t_{\rm out} - t) + \int_t (s - t)\, \eta$),
and Lemma J makes every junction tangential, bounds or not, so the rate rows and the
multiplier rows both carry the control's continuity there; $H$ continuity is necessary
too. A tie that drops one lets the control jump at the junction, and the multipliers of
the limits ridden beside it jump with it. Rest 2's progress under R39 came the same way:
keeping its multiplier rows (a variant tried and dropped), it does not move. Two
junctions of different limits that keep all their conditions coincide only for a
reason, as symmetry (R38) or Lemma K's corner, where the control is at the box's corner
and the rate row is the margin; R39 does not check for either. It stays off, as a
record. Acceptance caught every answer it produced.

On the census page's accepted rate answers, which run with R36 off, Lemma H2 holds:
re-solving the 21 on the first route alone, the 15 that solve there have no rise on any
order-2 arc above rounding ($10^{-16}$ of the arc's largest $\mu$).

### Events at one instant, to rounding (fixes, not rules)

By symmetry, a rest-to-rest motion with no torque arc left puts its rate peaks at
$T/2$: rest 14 ends with six rate arcs entering at one instant and leaving at another,
and on the way five rate touches at $T/2$. The route then lost every step to three
floating-point effects, each independent of the limits and their lemmas:

- **Sampling a segment one float wide.** Newton leaves two coincident junctions crossed
  or apart by an ulp ($7.7 \times 10^{-17}$ of $T$). `np.linspace` over such a segment
  repeats a time, and the integrator refuses times that are not strictly monotone, so
  the violation check after every converged step raised. Rest 14 lost 424 converged
  steps this way. A segment narrower than its samples is now sampled at its two ends
  (`shooting.sample_segment`), and at zero width it is its one point.
- **Ordering junctions closer than the integration resolves.** Acceptance and `_retire`
  require the junctions in the structure's order. Without R38 the slack was zero, so a
  crossing of one ulp rejected an exact solution that passed every other check (rest
  14: residual $1.3 \times 10^{-13}$, violation $2 \times 10^{-14}$, no $\mu$ negative
  or rising). Junctions crossed by $d$ leave a segment of width $-d$, integrated
  backward. That moves the motion by $O(d\,|f|)$, which is below the integration's own
  error when $d < \mathrm{rtol}\, T$, so the two orders cannot be told apart. The
  tolerance is now never less than the route's `rtol` ($10^{-12}$), with R38's
  `coincide_tol` above it when R38 is on.
- **A segment of zero width.** Touches seeded at the same sample, the peak at $T/2$,
  have equal times, and `_segment` refused any segment that was not strictly
  increasing. The seed then never propagated (residual $7 \times 10^{6}$ for every
  step). The flow over no time is the identity, and its derivatives in the end times
  are $\mp F$ there, the limit of the positive widths. The residual is therefore
  continuous through the coincidence, and Newton separates the events if their
  conditions do.

Rest 14 on the first route (release rules) solves in 13.5 s with the three fixes,
where it stalled at 17 s or 19 s before (by sampling, then by the zero width). The other
twelve structural stalls of the census page, on the first route with 120 s, are where
they were: none of them was held by these.

### Where a route's time goes, and Newton's contraction as the step: Rule R37

*Where the time goes.* Rest 11 on the first route (release rules) solves in 227 s. The
integrator is not the cost: 266,607 arc integrations average 0.58 ms each. What costs is
their number. The route makes 621 path solves and 20,986 evaluations of the shooting
function, and 233 of the solves fail (78 s). MINPACK, the fallback after a chord solve
ends close, uses about 66 evaluations a solve, 166 s in all. Most failures sit on stretches
where the structure does not change: on one with 20 junctions, steps of 3 to 5% of the leg
alternate between converging and failing for 30 s of route. The step control halved after a
failure and grew by 1.5 after a success, so it settled into that sawtooth, and each
failure cost a full solve.

*The quantity.* Newton's first contraction from a step's predictor,
$\Theta = \|J^{-1} R(v_1)\| / \|J^{-1} R(v_0)\|$ with the Jacobian at the predictor,
estimates half the Kantorovich quantity $\omega \|\Delta v_0\|$ (Deuflhard 2004, sec. 2.1,
affine-covariant form). The predictor's error grows like $h^p$ in the step: $p = 2$ for
the secant predictor and $p = 1$ from the last solution. So $\Theta \approx c\,h^p$, and the
step that gives a target contraction $\Theta^*$ is $h\,(\Theta^*/\Theta)^{1/p}$ (Deuflhard's
adaptive pathfollowing, sec. 5.1).

*Rule R37* (`Homotopy.contraction_steps`). The chord solve reports $\Theta$. Above
$\Theta = 1/2$ the step is too long for Newton: the solve ends after one Jacobian and one
residual, no fallback solver is run, and the step is cut to $h\,(1/4\,/\,\Theta)^{1/p}$,
clipped to $[0.1h, 0.5h]$. After a success that kept its structure, the next step is
$h\,(1/4\,/\,\Theta)^{1/p}$, clipped to $[0.5h, 2h]$. It only steers: every step is
still solved to the path tolerance and settled as before.

*Two corrections, found on the first prototype.*
- After a predictor that is nearly exact, both corrections are at the integration's noise.
  Their ratio is then about 1 whatever the step, and the rule shrank good steps (goal 5
  at 70%: 581 steps instead of 98). When the first step already reaches the tolerance,
  $\Theta$ is now the ratio of the residuals, which bounds its order.
- The one-limit-at-a-time legs end at $\mathrm{d}s_{\min} = 10^{-3}$. With the rule's floor
  there too, a rejected step went straight to the leg's end without the fallback that
  used to rescue it (rest 1 stalled at $|o| = 18.9$). The rule now stops steering within
  three halvings of the leg's $\mathrm{d}s_{\min}$ ($8\,\mathrm{d}s_{\min}$), so the old
  behaviour finishes each leg.

*Result, first route.* Release rules plus the rounding fixes above, 300 s, four cases
at once:

| case | without R37 | with R37 |
|---|---|---|
| goal 2 at 70% | 60.2 s | 54.8 s |
| goal 4 at 80% | unsolved ($\lvert o\rvert = 0.61$) | 55.1 s |
| goal 5 at 70% | unsolved (0.05) | 141.4 s |
| rest 3 | unsolved (8.1) | 147.5 s |
| goal 7 at 70% | 287.6 s | unsolved (2.18) |
| goal 9 at 80% | 251.1 s | unsolved (0.73) |
| rest 1, rest 2, rest 8, rest 11 | unsolved | unsolved |

Steps that do not converge fall from 1,472 to 273 over the ten cases. Where a case is
decided, R37 wins four and loses two. Single runs move cases on and off by chance (rest 11
solved in 77 s with R37 and 193 s without it on the code before the rounding fixes), so
the census page decides.

*Census page (release configuration plus R37, 60 s cap): 67 of 90, against 69 without it.*
Goal 4 at 80% is gained (57.5 s). Rests 2, 9 and 14 are lost, each at the cap, and the 66
common solves take 7% longer (956 s to 1021 s). R37 stays off.

*Why it cannot be right here.* The theory needs a residual with a Lipschitz Jacobian, and
the route's residual is only piecewise smooth: the held set on a riding arc changes at
events (R22). On rest 14, $\Theta$ is about $10^{-5}$ for steps of $1.2 \times 10^{-3}$ and
20 to 80 for steps of $2.4 \times 10^{-3}$ from the same point, with a start residual of
about $10^{-2}$. That is a kink crossed, not the $h^2$ growth of a smooth predictor. R37
reads it as distance and cuts the step by ten, so rest 14 takes 351 steps instead of 65
(16.8 s against 11.9 s alone, past the cap on the page). Where the residual is smooth
the rule does what the theory says (rest 11). A version for this residual would need to
tell a kink from distance, for instance from where the held-set events fall in the step,
and that is not done.

### Crossed junctions at the exact solve: Rule R41

After the rounding fixes, rest 8's route reaches the end of its offsets, and its exact
solve converges (residual $9.6 \times 10^{-11}$, violation $10^{-11}$, no $\mu$ negative
or rising). It is rejected because junctions are out of order: the exits of `w1_max` and
`w2_max` are crossed by $2.6 \times 10^{-8}$ of $T$, and those of `w5_max` and `w4_min` by
$1.3 \times 10^{-7}$. Both crossings are well above the integration's resolution, so they
are real. Below the route's own tolerance, the path structure was in order. The exact
problem, without the regularization, moves the junctions far enough to pass each other.

*Rule R41* (`Homotopy.exact_reorders`). When an exact solution converges with junctions
crossed past the coincidence tolerance, the structure with the events in the order of
their times is solved from it, at most twice. Lemma E5 says two limits' junctions cross
transversally, so the swapped order is the structure beyond the crossing. The swapped
seed has the crossed segment at width $|d|$ instead of $-|d|$. Acceptance judges the
result as any other.

*Result, first route.* Rest 8 (release rules plus R41, 120 s) is reordered twice. The
first reorder still has `w1_max` and `w5_max` crossed by $8 \times 10^{-8}$, and the second
is accepted (residual $6 \times 10^{-12}$). On the census pages since the rounding fixes, this
rejection is the only one among the unsolved cases besides goal 0 at 90%'s violations:
rest 8 on each, and rest 14 on the page with R37.

*Census page (release configuration plus R32 and R41, 60 s cap): 71 of 90, against 69.*
Rest 8 is new (46.3 s, R41), and so is goal 0 at 90% (14.1 s, R32: its exact solution
violated a limit off its arcs by $2.5 \times 10^{-7}$, above the check's $10^{-7}$). Nothing
is lost, and the 69 common solves take the same time (1071 s and 1076 s). Both rules join
the release configuration.

### Order-2 junctions: Lemma H2's rows in place of H continuity (Rule R43)

*Where the time goes inside a solve.* On rest 1 (first route, release rules) a path solve
that converges takes a median of 5 fresh Jacobians and 9 residuals, and its residual falls
by about 4 per Jacobian: 1e-1, 1e-4, 3e-5, 8e-6, 2e-6, 5e-7. Newton converges like that,
linearly, either on a wrong Jacobian or on a multiple root. Against central differences at a
converged path solution, every segment's STM is right to $10^{-7}$ (free arcs in closed
form, riding arcs with their held-set events, a set of two limits ridden at once), and so
are the gradients of every set's $H$. The rows whose Jacobian disagrees, by up to 28%, are
all one kind: $H$ continuity at the exit of an order-2 (rate) arc.

*Why.* At a tangential junction the control is continuous (Lemma J), and
$H(z, u_{\rm arc}) - H(z, u_{\rm free}) \approx \tfrac12 (u_{\rm arc} - u_{\rm free})^\top
R\,(u_{\rm arc} - u_{\rm free})$ is quadratic in the mismatch. That is a double root. Newton on
it converges linearly, and differences across it are dominated by the curvature. The
shooting already replaces $H$ continuity at order-1 junctions by simple-root rows (R7:
$\mu = 0$ at an exit, $\mu = \pi$ at an entry), and kept it at every order-2 junction.

*Rule R43* (`MultiArcShooting.order2_rows_default`). An order-2 arc's junctions take
Lemma H2's rows: $\mu(t_{\rm out}^-) = 0$ at the exit and $\mu(t_{\rm in}^+) = \pi_2$ at the
entry, with $\pi_2$ the coefficient of $\nabla\dot g$ in the entry's jump. H2 proves both
for the shooting's formulation (indirect form, $\lambda$ continuous at the exit), with no
corner exception. They replace one row each, so the count is unchanged. Acceptance is
unchanged too: the exact solution is judged as before.

*Result.* The same path solves now take one Jacobian and one to three residuals, the
residual falling by $10^3$ to $10^4$ a step. On the first route (release rules plus R43,
300 s, four cases at once):

| case | before | with R43 |
|---|---|---|
| goal 2 at 70% | 60 s | 28.9 s |
| rest 11 | 193 to 227 s | 41.8 s |
| goal 4 at 80% | 88 s, or unsolved | 46.9 s |
| rest 3 | 144 s, or unsolved | 65.4 s |
| rest 1 | 222 s | 81.1 s |
| goal 5 at 70% | 105 s, or unsolved | 232.7 s |
| goal 9 at 70% | unsolved | 213.5 s |
| rest 12 | unsolved | 235.3 s |

Goals 9 at 80%, 0, 1 and 4 at 70%, 3 at 80%, and rests 2, 10 and 13 still stall. The jerk axis
riding its rate limit gives the same extremal with either row (final time to $10^{-10}$,
initial costates to $10^{-8}$), with symbolic arcs and with arcs ridden at run time.

*Census page (release configuration with R32, R41 and R43, 60 s cap): 80 of 90, against
71.* Nine cases are new: goals 5 and 9 at 70% (16.5 s, 18.6 s), rest 13 (21.8 s), goal 2 at
70% (22.4 s), rest 3 (25.5 s), goal 9 at 80% (29.5 s), goal 1 at 70% (35.4 s), rest 11
(38.3 s) and goal 4 at 80% (45.2 s). None is lost. The 71 common solves take 826 s instead of
1137 s, and the median falls from 13.8 s to 10.4 s. Goal 1 at 70% is among them, the case
whose entries of `tau5_min` and `w3_max` swapped back and forth on its first route (why
it now solves is not yet traced). Ten stay unsolved: goals 0, 3 and 4 at 70%, goal 3 at 80%, and rests 1, 4, 5,
6, 10 and 12. R43 joins the release configuration.

## Rules these license

| Rule | What the solver does | Needs |
|---|---|---|
| R1 | Start the offsets at $o_c(0) = p_c (1 + U_c) + \epsilon_c$ with random $U_c$, so that births do not coincide | Lemma D: C1-C5, plus its hypothesis on $\mathcal{E}_2$, which rests on Lemma E and the local checks |
| R2 | At a stalled birth or death, step in the young arc's width $w$ with $s$ solved for | Lemma E: C2-C4, interior jerks at the event; L1-L3 not checked (R2 only steers) |
| R3 | A nested birth is R2 with the curvature in L1 taken along the riding motion of the arcs already active | as R2, with $S \cap B = \emptyset$ at the event (Lemma B) |
| R4 | When a solve fails, solve the reversed problem and map its extremal back exactly | Lemma F: C6 |

When a hypothesis fails the rule stays off, and the solver records which
hypothesis failed and where. A failure of L1 means a degenerate touch (a
higher-order contact); of L2, a turning point (pseudo-arclength continuation
is the tool there, and it is not part of this note); of L3, an offset path
tangent to the event; of $S \cap B = \emptyset$, the corner case below.

## How a rule is specified

A lemma proves something about the answers (a necessary condition) or about
the solution set along the route (regularity, symmetry). A rule changes the
solver. Soundness, that no invalid answer gets through, is the easy part,
because acceptance checks every condition. What a lemma about the answers does
not give is a reason the rule should *help*: it says what the answer looks
like, not how Newton or the route gets there. So each rule now states:

| Field | Content |
|---|---|
| Hook | where in the route it acts: route start, newborn arc, seed failed, consistent step, step stalled, junction rows, evaluation, solve failed |
| Trigger | the observable event, decided from the solver's state |
| Hypotheses | certificates (exact or by construction) and local conditions checked at the event |
| Action | the one thing it changes: acceptance, proposed structures, seed, rows, or the parameter of continuation |
| Soundness | what it preserves, from which lemma |
| Effectiveness | a lemma about the solver's own problem (regularity, convergence from the seed, a predictor's error), or "heuristic" |
| Prediction | the failure class it should reduce (`Diagnostics`), written down before the run |

The rules as they stand, in `auto_optimal_control.rules`:

| Rule | Hook | Lemmas | Effectiveness | Targets | Measured | Default |
|---|---|---|---|---|---|---|
| R1 | route start | C, D | Lemma D | coincident births | 22 → 25 solved | opt-in |
| R2 | step stalled | E | Lemma E | stalled steps | 27 → 26; predictor missing | off |
| R4 | solve failed | F | heuristic | not exact | 25 → 26 | on |
| R5 | consistent step | H | heuristic | mu rises | 26 → 29 | on |
| R6 | newborn arc | I | heuristic | seed failed: riding | removed rest 4's crashes; 29 → 28 | on |
| R7 | junction rows | J | heuristic | (rows degenerate at bounds) | 28 → 27; rest 0 solved | on |
| R8 | seed failed | I, J | heuristic | seed failed: riding | no gain | off |
| R9 | evaluation | K | heuristic | riding failed | 27 → 24 | off |

Only R1 and R2 have an effectiveness lemma. The two most useful rules, R5 and
the Lemma H acceptance check, worked without one. That is not a reason to skip
effectiveness lemmas. It says that a necessary condition is most valuable in
acceptance and in pruning structures, and that rules which change the route's
geometry need a lemma about the route.

*The tools.* `HomotopyResult.diagnostics` counts failed evaluations by class,
steps by outcome, rejections by check and rules fired.
`Homotopy.snapshot_dir` saves failed steps, and `diagnostics.replay` reruns
one in seconds. `lemmas.check` runs the oracles for Lemmas H, I, J and E on a
solution, and `Homotopy.check_lemmas` runs them on every path step.
`lemmas.LimitObjects` gives the lemmas' quantities ($a$, $d$, the margin,
$\alpha$, the free rate) from the kernels and the expressions of C1 and C2.

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
