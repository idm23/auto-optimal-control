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

The census of 2026-11-27 on homebox-node-1 (six cores, `--parallel-routes`, a 300 s cap)
solves 53 of the 54 cases, 46 within 60 s; the mean over the solved is 26.3 s. Every
answer ends on its target orbit to 1.2e-8 or better (2e-10 at minimum time). Each case
starts from its neighbours' answers (`Homotopy.solve(start=)`, `neighbours` in the
script): a J2 case from its two-body twin, every case from the one before it on its
ladder. The answer is continued in the parameters, a secant predicting each step; where
that fails (the structure changes on the way), the case is solved from the costate
draws. `--no-seed` solves every case from the draws, as the census of 2026-11-26 did:
31 of the 36 minimum-time cases, in a mean of 51.7 s.

- **Minimum time: all 36**, a mean of 14.7 s, 25 under 5 s.
  - **`raise`** down to 0.0001 g (0.98 mm/s²): 92.5 revolutions over 7.1 days in 82 s
    (`raise8`), 92.8 with J2 in 6.7 s (`jraise8`, from its twin in one step; 242 s from
    the draws). J2 lengthens the transfer by 0.1 to 0.3%. Edelbaum's estimate is within
    1% from 0.005 g down.
  - **`incl`** down to 0.0003 g: 98 revolutions over 7.7 days (`incl4`, 263 s, in 7
    steps from `incl3`). The draws solved none of the four cases from 30 revolutions
    down.
  - **`gto`** down to 0.0003 g: 11.3 revolutions over 8.5 days.
- **The extremum depends on the path.** Many extremals reach the target, one per
  number of revolutions and more, and a case lands on the one its route reaches. From
  the draws, `gto1` took 125.7 time units and `gto2` 361.0. Seeded, they take 89.8 and
  263.9, and `gto3` 904.7 against 945.3. A seeded answer is not always the least either:
  without the secant predictor, `gto3` reached 892.2, `jgto1` 91.1 (94.3 here) and
  `incl2` 82.59 (82.83 here).
- **Minimum fuel: 17 of 18**, a mean of 51.0 s.
  - **Propellant.** Given time to spare, the raise spends 2.029 to 2.033% of the mass at
    every thrust, against 4.8% at minimum time at 0.05 g and 2.06 to 2.09% at the lower
    thrusts. The impulsive (Hohmann) transfer, a delta-v of 0.0763, spends 2.028%: the
    low-thrust answers come within 0.005% of the mass of it. At 0.05 g and 1.2 times the
    minimum time there is not yet room for that: 2.28%.
  - **Switches.** The engine burns near the perigee and the apogee of each
    revolution: 2 switches at 0.05 g and 1.2 times, up to 18 at 0.002 g and 2 times
    (9.4 revolutions).
  - **From the neighbours.** Eight of the seventeen come from a neighbour, in 0.01 to 6.3 s:
    the longer `t_f`s from the shorter at the same thrust, and `jfuel7`, `jfuel8` (at the
    cap from the draws) from their twins. The others' switching differs from their
    neighbours' (3 switches against 4 at 0.01 g with J2), and they come from the draws,
    in up to 215 s.
  - **Unsolved: `jfuel6`** (0.002 g, 1.2 times, 5.7 revolutions), at the cap. Its twin
    has 6 switches; with J2 the continuation from the draws reads 6 too, but none of its
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

- **Choosing among the extrema.** Seeding reaches better extrema than the draws, but which
  one depends on the path (`gto3`: 892.2 or 904.7). Trying every neighbour and keeping
  the least, or a second solve from one revolution fewer, would choose rather than
  inherit.
- **Switches born on the way.** A continuation from a neighbour holds its structure, so
  a fuel case whose switches differ from its neighbour's goes back to the draws (up to
  215 s). Following the switching function through a birth, as the active-set route
  follows a limit's, would let every case start from its neighbour, and from its rung's
  minimum-time answer (the fuel answer at `t_f = t_min`).
- **Faster minimum fuel from the draws.** Below `eps = 1e-5` a continuation step takes
  10 to 75 s, in MINPACK solves of about 60 propagations each, and the exact structure
  comes near `eps = 1e-7`.
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
    --export docs/_static/orbit_census/data
```

<iframe src="_static/orbit_census/index.html" title="Orbit transfer census" loading="lazy"
        style="width: 100%; height: 1500px; border: 0;"></iframe>

[Open the census on its own page](_static/orbit_census/index.html).
