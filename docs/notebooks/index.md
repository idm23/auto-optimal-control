# Lectures

A short course on the indirect (Pontryagin) method, one notebook per lecture.
Every lecture uses the same plant, the cart-pole swing-up from Kelly (2017),
and each one adds one complication to the problem before it. Theory comes
first, written as lecture notes, with SymPy doing the algebra in view. The
application lectures then solve the problem with this package and show all of
the code that does it, and check the answer independently.

The plant, fixed throughout: cart mass $m_1 = 1$ kg, pole mass $m_2 = 0.3$ kg,
pole length $l = 0.5$ m, $g = 9.81$ m/s², from rest hanging down to rest
upright, a distance $d = 1$ m down the track.

| # | Lecture | Problem | New ideas |
|---|---|---|---|
| 1 | {doc}`01_necessary_conditions` | minimum effort, $T = 2$ s | the Hamiltonian and costate equations from a first variation; the boundary value problem; the minimum principle, the clipped and bang-bang laws and the switching function; the state transition matrix and its jump at a switch |
| 2 | {doc}`02_shooting_the_swing_up` | minimum effort, $\lvert u \rvert \le 20$ N | single shooting and why it is sensitive; random multistart; a direct collocation solution for comparison; several extremals; independent verification |
| 3 | A motor with a limit *(in preparation)* | the force limit tightened until it binds | the clipped law in action; continuation in the limit; kinks in the dynamics |
| 4 | As fast as possible *(in preparation)* | minimum time | bang-bang control; free final time; seeding a bang-bang solve by smoothing; singular arcs |
| 5 | A track with ends *(in preparation)* | $\lvert q_1 \rvert \le x_{\max}$ | state constraints and their order; exterior penalties; boundary arcs, touch points and costate jumps |
| 6 | Everything at once *(in preparation)* | minimum time on a short track | combining bang-bang control with a state limit; what is still open |

To run the notebooks yourself, clone the repository and run
`uv sync --group docs`, then open them in Jupyter. They need a C compiler,
like the package.

```{toctree}
:maxdepth: 1
:hidden:

01_necessary_conditions
02_shooting_the_swing_up
```

## Sources

The lectures follow the order and notation of these courses and texts:

- MIT 16.323 *Principles of Optimal Control* (J. How, Spring 2008), lecture
  notes 6 (calculus of variations applied to optimal control), 8 (properties
  of the solution) and 9 (constrained optimal control: the minimum principle,
  minimum time, fuel and energy).
  [MIT OpenCourseWare](https://ocw.mit.edu/courses/16-323-principles-of-optimal-control-spring-2008/).
- Stanford AA203 *Optimal and Learning-based Control* (Spring 2026),
  lectures 4 (necessary conditions via the calculus of variations) and 5 (the
  minimum principle, minimum time, fuel and energy problems, indirect methods
  in practice). [Course site](https://stanfordasl.github.io/aa203/).
- D. Liberzon, *Calculus of Variations and Optimal Control Theory*
  (Princeton, 2012), the text of UIUC ECE 553, chapter 4: the maximum
  principle, bang-bang and singular controls.
  [Course page](https://liberzon.csl.illinois.edu/20ECE553.html).
- R. Tedrake, *Underactuated Robotics* (MIT 6.832), the cart-pole chapter.
  [Online notes](https://underactuated.mit.edu/).
- M. Kelly, "An Introduction to Trajectory Optimization: How to Do Your Own
  Direct Collocation", *SIAM Review* 59(4), 2017: the cart-pole problem, its
  published solution, and the direct method Lecture 2 compares with.
- D. E. Kirk, *Optimal Control Theory: An Introduction* (Dover, 2004),
  chapter 5; A. E. Bryson and Y.-C. Ho, *Applied Optimal Control* (1975),
  chapters 2 and 3, including state-variable inequality constraints.
- R. F. Hartl, S. P. Sethi and R. G. Vickson, "A Survey of the Maximum
  Principles for Optimal Control Problems with State Constraints",
  *SIAM Review* 37(2), 1995.
