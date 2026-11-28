# Orbit transfer census

Low-thrust transfers around the Earth (`auto_optimal_control.orbits.low_thrust_transfer`):
minimum-time, with the thrust always on at full strength and its direction a unit vector
the solver steers, and minimum-fuel, where the solver also switches the engine on and
off. The dynamics come from `orbitprop`, the package's symbolic
propagator: Gauss's variational equations in modified equinoctial elements
`p, f, g, h, k, L`, with the Earth's zonal harmonics from EGM2008 as a perturbation in
the radial / transverse / normal frame. Everything is a SymPy expression, so the costate
equations and the control law are derived, not hand-written, and adding a force (J3,
drag, solar radiation pressure) is adding a term.

Units are canonical: lengths in Earth radii (6378.1363 km), times in
`sqrt(R^3 / mu)` (806.8 s), so one circular orbit at the surface takes 2 pi. The thrust
`T` is the initial thrust acceleration in units of surface gravity (9.798 m/s²); the
engine's exhaust speed is that of a 3000 s ion engine throughout, and the mass starts
at 1.

There are three families, each a ladder of falling thrust, so of more and more
revolutions, and each case again with J2 (`j` in front of its name):

- **`raise<i>`**: a coplanar raise between circular equatorial orbits, from 1.1 to 1.3
  Earth radii (638 to 1913 km up), at `T` = 0.05 down to 0.0001 g (half a revolution
  to about ninety).
- **`incl<i>`**: the same raise with a 10° plane change, at `T` = 0.03 down to
  0.0003 g.
- **`gto<i>`**: from a geostationary transfer orbit (perigee 250 km, apogee
  35,786 km, 7°) to the geostationary orbit, at `T` = 0.01 down to 0.0003 g.
- **`fuel<i>`**: the raise again at minimum propellant, at 0.05, 0.01 and 0.002 g (`raise0`,
  `raise2`, `raise4`), with the throttle free in [0, 1] and `t_f` fixed at 1.2, 1.5 and
  2 times that rung's minimum time.

Each case starts at the perigee of its orbit (the longitude `L` = 0) and ends anywhere
on the target orbit (`L` free). The final time is searched in 0.3 to 4 times Edelbaum's
estimate `dv / T`, and the costates in a box of half-width `1 / T`: a minimum-time
costate says how the arrival time moves with the elements, so it grows as the thrust
falls. A minimum-fuel case's costates are in a box of half-width `1 / c`: its costate says
how the propellant spent moves with the elements, and the thrust costs `T delta / c`.
The rules are the arm census's of 2026-11-25 (before R90), with R91 on for minimum fuel,
and a cap of 300 s a case.

## Results

The census of 2026-11-26 on homebox-node-1 (six cores, `--parallel-routes`, a 300 s
cap) solves 31 of the 36 cases, 22 of them within 60 s; the mean over the solved is
51.7 s. Every answer ends on its target orbit to 5e-11 or better.

- **`raise`: 17 of 18**, down to 0.0001 g (0.98 mm/s²): 92.5 revolutions over 7.1 days
  in 24 s (`raise8`), 92.8 with J2 in 242 s (`jraise8`). J2 lengthens the transfer by 0.1
  to 0.3%. Edelbaum's estimate is within 1% from 0.005 g down. `raise7` (46
  revolutions) fails at the random draws; an earlier run of the census solved it in 46 s,
  before a change to the thrust direction's law that left its value the same but its
  arithmetic not. The draws decide it.
- **`incl`: 6 of 10.** Down to 0.003 g (9.7 revolutions), two-body and J2. The four
  failures (`incl3`, `incl4`, `jincl3`, `jincl4`, 30 and 120 revolutions) are all at the
  first stage, the random draws of the costates and the final time: none solved the
  regularized problem within the cap. Before the cap held (see below) a probe solved
  `incl3` in 1407 s, so these fail on the budget, not for want of an answer. `incl2`
  solved here in 294 s and failed in the earlier run: the same luck of the draws.
- **`gto`: all 8**, down to 0.0003 g (2.9 mm/s²): 14.7 revolutions over 8.8 days. Two of
  the two-body answers are local extrema, not the minimum: `gto1` and `gto2` take 126
  and 361 time units where their J2 twins take 91 and 275, in both runs. The product
  `T t_f` (the delta-v a constant-thrust transfer spends) is 0.27 to 0.29 for the other
  six and 0.36 to 0.38 for these two. Each has more revolutions than its twin (2.2
  against 1.6, 7.7 against 3.3), and J2 is far too weak to explain the gap.
  Edelbaum's estimate, taken from the perigee, is twice the answer here; the final-time
  box still holds it.
- **`fuel`: 17 of 18** (2026-11-27, node-1, one case at a time, no parallel routes):
  all nine two-body cases and eight of the nine with J2, 10 under 60 s, a mean of 70.6 s.
  Every answer ends on its target to 4e-8 or better. A J2 case starts from its two-body
  twin's answer (`Homotopy.solve(start=)`), continued in the parameters with the
  twin's switches held; where J2 changes the switching, that fails in a few steps and
  the case is solved from the costate draws as before.
  - **Propellant.** Given time to spare, the raise spends 2.029 to 2.033% of the mass at
    every thrust, against 4.8% at minimum time at 0.05 g and 2.06 to 2.09% at the lower
    thrusts. The impulsive (Hohmann) transfer, a delta-v of 0.0763, spends 2.028%: the
    low-thrust answers come within 0.005% of the mass of it. At 0.05 g and 1.2 times the
    minimum time there is not yet room for that: 2.28%.
  - **Switches.** The engine burns near the perigee and the apogee of each
    revolution: 2 switches at 0.05 g and 1.2 times, up to 18 at 0.002 g and 2 times
    (9.4 revolutions).
  - **From the twin.** `jfuel7` and `jfuel8` (7 and 9.4 revolutions), at the cap from
    the draws, solve from their twins in 24.5 and 31.2 s, and `jfuel0` in 0.013 s. The
    others' switching differs from their twins' (3 switches against 4 at 0.01 g).
  - **Unsolved: `jfuel6`** (0.002 g, 1.2 times, 5.7 revolutions), at the cap. Its twin
    has 6 switches, and with J2 the continuation from the draws reads 6 but none of its
    exact solves converges.

The census found these bugs on its way:

- **The cap did not hold.** `Homotopy.solve_seconds` did not bound the random draws
  (`SingleShooting.multistart`), so a case could overrun its cap by hours (8549 s and
  24341 s for `jincl3` and `incl4` in a probe). Then, with the draws bounded, one draw's
  root-find could still carry a case far past it (551 s and 761 s under a 300 s cap in
  this census, which ran before the fix). Now `Homotopy.solve` sets
  `SingleShooting.deadline` and every propagation past it fails at once.
- **`trajectory()` was not the extremal solved.** `SingleShooting.trajectory` integrated
  at the kernels' default tolerances, not the shooting's own. Replayed that way, a
  minimum-fuel raise with 8 switches gained a ninth and missed its target by 0.018. The
  page draws every answer at the tolerances it was solved at.
- **`derive()` on the fuel problem.** The problem's size broke it three times. The
  throttle cancels out of the thrust direction `-b/|b|`, but only `simplify()` cancelled
  it, and the law is too large to simplify. So the direction carried the throttle's whole
  law, and the regularized throttle's law carried the throttle itself; SymPy's `Max` and
  `Min` then compared those laws for hours. Now the direction's common factors and the
  minimizer of a quadratic are taken out exactly past `SIMPLIFY_LIMIT_OPS`: the fuel
  problem derives in 3 s two-body and 30 s with J2, regularized, and every other problem
  derives as before.
- **The smoothing started too high for a fuel cost** (Lemma ZC, Observation O14). The
  throttle's smoothing started at `eps = 1`, a hundred times the fuel cost's own scale
  `T / c`. Where a constant throttle of about one half reaches the target, the costates
  then shrank to zero and the continuation stalled: 7 of the first 9 fuel cases. It now
  starts at `0.06 |a| half` for a control whose running cost is `a u`, and at 1 as before
  for every other control.
- **A burn inside one integration step was missed.** On a coast the elements barely move,
  so the integrator's steps grew past a time unit, and the switching function was tested
  only at their ends: a whole burn could fit inside one step. The solve's own
  propagations carried the STM, whose shorter steps found the burns, so three answers were
  accepted that replay never reached (`fuel2` replayed spent no propellant at all). The
  kernels now test the switching functions inside every step, and the verdict replays an
  answer without the STM before accepting it.

What comes next, in the order the results ask for it:

- **Costates from the rung above.** Down a ladder the answers change smoothly with the
  thrust, and a minimum-time costate scales roughly as `1 / T`; seeding each case from
  the one above would replace the random draws that fail on `incl`.
- **Telling a local extremum from the minimum** (`gto1`, `gto2`): the number of
  revolutions is the discrete choice, and a second solve from the J2 twin's answer, or
  from one revolution fewer, would show it.
- **Faster minimum fuel.** Below `eps = 1e-5` a continuation step takes 10 to 75 s, in
  MINPACK solves of about 60 propagations each, and the exact structure comes near
  `eps = 1e-7`. `solve(start=)` from a neighbour skips all of that when the switching
  carries over: from the shorter `t_f`, or from the rung's minimum-time answer (the fuel
  answer at `t_f = t_min`), a continuation that lets switch pairs be born would do it for
  every case.
- **More of minimum fuel:** the plane change and GTO to GEO.
- **More forces** in `orbitprop`: higher zonals are a parameter away (`zonal_degree`),
  then drag, solar radiation pressure and third bodies.

Pick a case to watch it. The spiral runs from blue at the start to gold at the end,
dashed where the engine is off, with the start orbit solid and the target dashed; the
red arrow is the thrust. The panel beside it shows the semi-major axis, eccentricity,
inclination and mass over the transfer, the steering (the thrust's angle in the orbit
plane from the transverse direction towards the radial, and out of it) and, for
minimum fuel, the throttle. The page is drawn from the data in
`_static/orbit_census/data`:

```bash
python benchmarks/orbit_census.py orbit_census.jsonl --cap 300 --parallel-routes \
    --export docs/_static/orbit_census/data --cases raise0 ...   # the minimum-time cases
python benchmarks/orbit_census.py fuel_census.jsonl --cap 300 \
    --export docs/_static/orbit_census/data --cases fuel0 ...    # and the fuel cases
```

<iframe src="_static/orbit_census/index.html" title="Orbit transfer census" loading="lazy"
        style="width: 100%; height: 1500px; border: 0;"></iframe>

[Open the census on its own page](_static/orbit_census/index.html).
