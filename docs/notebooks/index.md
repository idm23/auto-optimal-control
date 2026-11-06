# Lectures

A course on optimal control by the indirect (Pontryagin) method, one
notebook per lecture, in six parts, with a prerequisite lecture before them. It covers what a first graduate course
covers, from the brachistochrone to state constraints, and every mechanism
the package is built on. Each lecture opens with a challenge the previous
ones cannot meet, tells the story of the people who met it, develops the
theory with SymPy doing the algebra in view, and then solves real problems
with the package, showing all of the code and checking the answers
independently. A reader who works through all twenty can use every part
of the package; Lecture 17 maps each module to the lectures that teach it.

Diagrams drawn to illustrate an idea, rather than to solve something, have their
plotting code folded away: click "Show code cell source" above a figure to see
it. Everything else shows its code.

## Prerequisites

Single-variable calculus and matrix multiplication are all this assumes. Skip it
if gradients, Jacobians and Hessians are familiar.

| # | Lecture | The challenge | New ideas |
|---|---|---|---|
| P1 | {doc}`p1_jacobians_and_hessians` | the words every lecture uses | derivatives as linear approximations; the gradient; the Jacobian and its determinant; the chain rule; Newton's method and how it fails; the Hessian and the test for a minimum; finite differences, symbolic and automatic differentiation |

## Part I. Formulating optimal control

| # | Lecture | The challenge | New ideas |
|---|---|---|---|
| 1 | {doc}`01_brachistochrone` | optimize a curve, not a number | Bolza, Lagrange and Mayer forms; states, controls, boundary conditions; posing problems with `OptimalControlProblem` |
| 2 | {doc}`02_calculus_of_variations` | a method, not a trick | the first variation; Euler–Lagrange; natural boundary conditions; costates as multipliers; transversality; what `derive` returns |
| 3 | {doc}`03_hamilton_cart_pole` | a real, nonlinear system | Hamilton's canonical equations; cyclic coordinates and constant $H$; the cart-pole's necessary conditions by hand and by `derive`; a first look at the STM and its jump |
| 4 | {doc}`04_minimum_principle` | bounded controls | Pontryagin's minimum principle and needle variations; saturated, bang-bang and unit-vector laws; switching functions |
| 5 | {doc}`05_dynamic_programming` | which extremal is optimal, and what is $\lambda$? | the value function and HJB; $\lambda = \nabla V$; the linear-quadratic regulator and Riccati; the curse of dimensionality |

## Part II. Computing extremals

| # | Lecture | The challenge | New ideas |
|---|---|---|---|
| 6 | {doc}`06_compiled_kernels` | thousands of accurate integrations | adaptive Runge–Kutta and DOP853; code generation, compilation and the cache; the forward map `propagate`, `trajectory`, `propagate_batch` |
| 7 | {doc}`07_switches_as_events` | discontinuous dynamics | the extremal as a hybrid system; switches and saturation kinks as events; status codes |
| 8 | {doc}`08_state_transition_matrix` | exact derivatives | the variational equation; the saltation jump; parameter and final-time sensitivities; symplecticity |
| 9 | {doc}`09_shooting` | turning misses into corrections | single shooting on the cart-pole; random multistart; several extremals; a direct method for comparison; independent verification |
| 10 | {doc}`10_continuation` | problems shooting cannot start | homotopy; smoothing bang-bang controls; continuation in a physical parameter; when paths break |

## Part III. Bang-bang and singular control

| # | Lecture | The challenge | New ideas |
|---|---|---|---|
| 11 | {doc}`11_minimum_time` | as fast as possible | free final time and $H = 0$; the switching curve; Feldbaum's theorem; the COPS robot arm |
| 12 | {doc}`12_singular_arcs` | when the switching function vanishes | singular control by differentiation; order; Legendre–Clebsch; Goddard's rocket; multi-arc shooting with singular arcs |

## Part IV. State constraints

| # | Lecture | The challenge | New ideas |
|---|---|---|---|
| 13 | {doc}`13_path_constraints` | limits on the state | order of a constraint; exterior penalties; crossings as events; termination |
| 14 | {doc}`14_boundary_arcs` | the exact constrained optimum | boundary arcs, touch points and costate jumps; seeding from penalties; limits with bang-bang and singular control; order three and chattering |

## Part V. Putting it together

| # | Lecture | The challenge | New ideas |
|---|---|---|---|
| 15 | {doc}`15_automatic_homotopy` | the structure is unknown | the automatic homotopy: regularize, continue in phases, read the structure, jump, accept |
| 16 | {doc}`16_spacecraft` | real orbit transfers | the primer vector; orbit raising; a fuel-optimal Earth–Mars rendezvous |
| 17 | {doc}`17_practice` | trusting and extending a result | independent checks; benchmarks and animations; the map of the package; open problems |

## Part VI. Robot motion

| # | Lecture | The challenge | New ideas |
|---|---|---|---|
| 18 | {doc}`18_minimum_time_several_axes` | the fastest motion of several axes | non-unique optima; vanishing costates and singular Jacobians; a secondary cost, a two-stage solve, a fixed path |
| 19 | {doc}`19_choosing_among_equal_motions` | a two-link arm in minimum time, with bounded jerks and torque limits | non-uniqueness read from the statement; selectable policies; riding a limit with bounded controls |
| 20 | {doc}`20_six_joints` | what breaks at six joints? | the bottlenecks, measured; the recursive Newton–Euler algorithm; tapes; arcs solved at run time; a survey of random motions |

To run the notebooks yourself, clone the repository and run
`uv sync --group docs`, then open them in Jupyter. They need a C compiler,
like the package. Each compiles the kernels it uses the first time, which
takes a few seconds per problem; later runs load them from the cache.

```{toctree}
:maxdepth: 1
:hidden:

p1_jacobians_and_hessians
01_brachistochrone
02_calculus_of_variations
03_hamilton_cart_pole
04_minimum_principle
05_dynamic_programming
06_compiled_kernels
07_switches_as_events
08_state_transition_matrix
09_shooting
10_continuation
11_minimum_time
12_singular_arcs
13_path_constraints
14_boundary_arcs
15_automatic_homotopy
16_spacecraft
17_practice
18_minimum_time_several_axes
19_choosing_among_equal_motions
20_six_joints
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
  (Princeton, 2012), the text of UIUC ECE 553.
  [Course page](https://liberzon.csl.illinois.edu/20ECE553.html).
- D. E. Kirk, *Optimal Control Theory: An Introduction* (Dover, 2004);
  A. E. Bryson and Y.-C. Ho, *Applied Optimal Control* (1975).
- R. Tedrake, *Underactuated Robotics* (MIT 6.832), the cart-pole chapter.
  [Online notes](https://underactuated.mit.edu/).
- M. Kelly, "An Introduction to Trajectory Optimization: How to Do Your Own
  Direct Collocation", *SIAM Review* 59(4), 2017: the cart-pole problem, its
  published solution, and the direct method Lecture 9 compares with.
- R. F. Hartl, S. P. Sethi and R. G. Vickson, "A Survey of the Maximum
  Principles for Optimal Control Problems with State Constraints",
  *SIAM Review* 37(2), 1995.
- H. J. Sussmann and J. C. Willems, "300 years of optimal control: from the
  brachistochrone to the maximum principle", *IEEE Control Systems Magazine*
  17(3), 1997, for the history.
