"""Extremals generated from one end, and the embedding that answers queries from them.

A single jerk-limited axis in minimum time with acceleration limits (order 1):
as stated its jerk is bang-bang, so arcs are entered at a corner; with the
secondary jerk cost the control law is continuous, so entries are tangential.
Both ends are parameters, so one set of kernels generates and polishes.
"""

import warnings

import numpy as np
import pytest
import sympy as sp

from auto_optimal_control.embedding import EndpointIndex, ExtremalSet, Polisher
from auto_optimal_control.examples import jerk_limited_min_time
from auto_optimal_control.generation import ExtremalGenerator, Rejected
from auto_optimal_control.homotopy import Homotopy
from auto_optimal_control.policy import SecondaryCost
from auto_optimal_control.problem import Box, OptimalControlProblem, PathConstraint
from auto_optimal_control.shooting import MultiArcShooting


def axis_between():
    """``x''' = j``, ``|j| <= J``, ``|a| <= A``, in minimum time from ``(x0, v0, a0)`` to ``(xf, vf, af)``."""
    t, x, v, a, j = sp.symbols("t x v a j", real=True)
    J, A = sp.symbols("J A", positive=True)
    e0, ef = sp.symbols("x0 v0 a0", real=True), sp.symbols("xf vf af", real=True)
    return OptimalControlProblem(
        name="axis_between", time=t, states=[x, v, a], controls=[j], params=[*e0, *ef, J, A],
        dynamics=[v, a, j], running_cost=1, initial_state=dict(zip([x, v, a], e0)),
        final_state=dict(zip([x, v, a], ef)), control_sets=[Box(j, -J, J)],
        path_constraints=[PathConstraint(a - A, "a_max"), PathConstraint(-a - A, "a_min")])


def params(x0, xf):
    return [*x0, *xf, 1.0, 1.0]


@pytest.fixture(scope="module")
def bang(tmp_path_factory):
    return Homotopy(axis_between(), cache_dir=tmp_path_factory.mktemp("k"), policy="as_stated")


@pytest.fixture(scope="module")
def smooth(tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Homotopy(axis_between(), cache_dir=tmp_path_factory.mktemp("k"),
                        policy=SecondaryCost())


def generate(h, n, seed=0, moving=False, **kwargs):
    """``n`` draws from rest at 0 (or, ``moving``, from a random speed and acceleration)."""
    gen = ExtremalGenerator.from_homotopy(h, params(np.zeros(3), np.zeros(3)), **kwargs)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x0 = np.array([0.0, *rng.uniform(-0.5, 0.5, 2)]) if moving else np.zeros(3)
        try:
            out.append(gen.generate(x0, rng.normal(size=3), rng.uniform(0.5, 6), rng=rng))
        except Rejected:
            pass
    return gen, out


@pytest.fixture(scope="module")
def bang_set(bang):
    gen, rest = generate(bang, 80)
    _, moving = generate(bang, 80, seed=1, moving=True)
    return gen, rest + moving


def assert_valid(samples):
    for e in samples:
        assert e.junction_residual < 1e-9
        assert e.checks["violation"] < 1e-8
        assert e.checks["multiplier_min"] > -1e-8
        assert e.checks["pi_min"] > -1e-8
        assert e.checks["hamiltonian"] < 1e-7      # H = 0 all along (free final time)


def test_generated_extremals_with_corner_entries_meet_every_condition(bang_set):
    gen, samples = bang_set
    assert not gen.regular
    assert len(samples) >= 120
    assert_valid(samples)
    with_arcs = [e for e in samples if e.arcs]
    assert len(with_arcs) >= 10
    assert any(len(e.arcs) >= 4 for e in with_arcs)            # two arcs, both limits


def test_generated_extremals_with_tangential_entries_meet_every_condition(smooth):
    gen, samples = generate(smooth, 40, seed=1)
    assert gen.regular
    assert_valid(samples)
    k, p = gen.kernels, gen.params
    entered = 0
    for e in samples:
        shoot, v = gen.shooting_of(e)
        _, _, segs = shoot._evaluate(v, jac=False)
        for (kind, c), (t_in, _, pi) in zip([a for a in e.arcs if a[0] == "enter"], e.values):
            # dg/dt = 0 at the end of the free arc before the entry: the arrival is tangential.
            before = next(sg for sg in segs if sg[0] is None and abs(sg[3] - t_in) < 1e-12)
            rate = np.atleast_1d(k.constraint_rate(t_in, before[-1]["z"], p))[c]
            assert abs(rate) < 1e-7
            assert pi >= -1e-12
            entered += 1
    assert entered >= 3


def test_a_generated_extremal_solves_its_own_boundary_value_problem(bang, bang_set):
    gen, samples = bang_set
    pol = Polisher(bang, params)
    for e in [s for s in samples if s.arcs][:4]:
        res = pol.solve(e.x0, e.xf, [e])
        assert res.accepted, res.message
        assert abs(res.tf - e.tf) < 1e-8
        assert res.arcs == e.arcs


def test_an_arc_open_at_the_end_has_an_exact_jacobian(bang, bang_set):
    gen, samples = bang_set
    open_end = [e for e in samples if e.arcs and e.arcs[-1][0] == "enter"]
    if not open_end:
        pytest.skip("no sample ends on an arc")
    e = open_end[0]
    shoot, v = gen.shooting_of(e)
    v = v + 1e-3 * np.random.default_rng(0).normal(size=len(v)) * np.abs(v).clip(1e-3)
    R, J = shoot.residual_jacobian(v)
    FD = np.zeros_like(J)
    for i in range(len(v)):
        h = 1e-6 * max(1.0, abs(v[i]))
        vp, vm = v.copy(), v.copy()
        vp[i] += h
        vm[i] -= h
        FD[:, i] = (shoot.residual(vp) - shoot.residual(vm)) / (2 * h)
    assert np.allclose(J, FD, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("D", [0.5, 2.0, 4.0, 8.0])
def test_rest_to_rest_from_the_embedding_matches_the_s_curve(bang, bang_set, D):
    gen, samples = bang_set
    data = ExtremalSet.from_extremals(samples)
    index = EndpointIndex(data, invariant=[0])
    pol = Polisher(bang, params, generator=gen)
    x0, xf = np.zeros(3), np.array([D, 0.0, 0.0])
    ids, _ = index.nearest(x0, xf, k=24)
    res = pol.solve(x0, xf, [index.shifted(i, x0) for i in ids])
    assert res.accepted, res.message
    tf, _, _ = jerk_limited_min_time(1.0, 1.0, 1e9, D)
    assert abs(res.tf - tf) < 1e-8


def test_the_structure_at_a_query_is_voted_by_its_neighbours(bang_set):
    _, samples = bang_set
    data = ExtremalSet.from_extremals(samples)
    held, rest = data.extremal(0), ExtremalSet.from_extremals(samples[1:])
    index = EndpointIndex(rest, invariant=[0])
    ranked = index.structures(held.x0, held.xf)
    assert abs(sum(p for _, p in ranked) - 1.0) < 1e-12
    assert all(a[1] >= b[1] for a, b in zip(ranked, ranked[1:]))
    hits = 0
    for i in range(1, 41):                     # leave one out over 40 samples
        e = data.extremal(i)
        others = ExtremalSet.from_extremals(samples[:i] + samples[i + 1:])
        hits += EndpointIndex(others, invariant=[0]).structures(e.x0, e.xf)[0][0] == tuple(e.arcs)
    assert hits >= 28
    arcs = ranked[0][0]
    assert all(tuple(rest.arcs[j]) == arcs for j in index.nearest_with(held.x0, held.xf, arcs, k=3))


def test_a_set_of_extremals_saves_and_loads(bang_set, tmp_path):
    _, samples = bang_set
    data = ExtremalSet.from_extremals(samples)
    data.save(tmp_path / "set.npz")
    back = ExtremalSet.load(tmp_path / "set.npz")
    assert len(back) == len(data)
    assert np.array_equal(back.lam0, data.lam0) and back.arcs == data.arcs
    assert back.values == data.values
    assert (data + back).x0.shape == (2 * len(data), 3)


def test_an_arc_left_open_needs_open_end(bang):
    bang.params = np.array(params(np.zeros(3), np.array([4.0, 0, 0])))
    kx, px = bang.exact_kernels, bang.exact_params()
    with pytest.raises(ValueError, match="never left"):
        MultiArcShooting(kx, px, [("enter", 0)])
    shoot = MultiArcShooting(kx, px, [("enter", 0)], open_end=True)
    assert shoot.n_unknowns == shoot.n_base + 2          # t_in and pi, no exit


def test_learned_proposals_train():
    torch = pytest.importorskip("torch")
    from auto_optimal_control import learn

    rng = np.random.default_rng(0)
    C = rng.normal(size=(400, 3))
    Y = np.hstack([np.sin(C[:, :1]) + 0.05 * rng.normal(size=(400, 1)), C[:, 1:2] ** 2])
    flow, sc, sy = learn.train_flow(C, Y, epochs=40, batch=100)
    with torch.no_grad():
        nll = -flow.log_prob(torch.as_tensor(sy(Y), dtype=torch.float32),
                             torch.as_tensor(sc(C), dtype=torch.float32)).mean()
    assert float(nll) < 1.0                    # well below a standard normal's 2.84
    ys = learn.flow_proposals(flow, sc, sy, C[0], 16, seed=0)
    assert ys.shape == (16, 2)
    model, ss, sy2, se = learn.train_jepa(C, Y, Y[:, :1] + C[:, :1], epochs=20, batch=100)
    y, err = learn.jepa_refine(model, ss, sy2, se, C[0], Y[0, :1] + C[0, :1], Y[:4], steps=20)
    assert y.shape == (4, 2) and np.all(np.diff(err) >= 0)


def test_anchored_extremals_start_on_an_exact_entry(smooth):
    gen = ExtremalGenerator.from_homotopy(smooth, params(np.zeros(3), np.zeros(3)))
    k, p = gen.kernels, gen.params
    rng = np.random.default_rng(2)
    kept = []
    for _ in range(40):
        z = np.r_[0.0, rng.uniform(-0.5, 0.5, 2), rng.normal(size=3)]
        T = rng.uniform(1, 5)
        try:
            kept.append(gen.generate_anchored(int(rng.integers(2)), z, 0.4 * T, 0.6 * T, rng=rng))
        except Rejected:
            pass
    assert len(kept) >= 15
    assert_valid(kept)
    anchored = [e for e in kept if e.checks["anchored"]]
    assert len(anchored) >= 10
    for e in anchored:
        shoot, v = gen.shooting_of(e)
        _, _, segs = shoot._evaluate(v, jac=False)
        c = k.CONSTRAINT_NAMES.index(e.checks["anchored"])
        entries = [(cc, val[0]) for (kind, cc), val in zip([a for a in e.arcs if a[0] == "enter"], e.values)]
        t_in = next(t for cc, t in entries if cc == c)
        before = next(sg for sg in segs if sg[0] is None and abs(sg[3] - t_in) < 1e-12)
        assert abs(np.atleast_1d(k.constraint_rate(t_in, before[-1]["z"], p))[c]) < 1e-7


def test_a_constructed_anchor_is_a_peak_of_the_limit(smooth):
    # Drawn costates saturate the free jerk, and half the draws would bend the wrong
    # way; the anchor is built so the free motion meets the limit from inside.
    gen = ExtremalGenerator.from_homotopy(smooth, params(np.zeros(3), np.zeros(3)))
    k, p = gen.kernels, gen.params
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = int(rng.integers(2))
        z = gen._construct_anchor(c, 0.0, np.r_[0.0, rng.uniform(-0.5, 0.5, 2), rng.normal(size=3)])
        assert abs(k.constraint(0.0, z, p)[c]) < 1e-9
        assert abs(np.atleast_1d(k.constraint_rate(0.0, z, p))[c]) < 1e-9
        assert abs(k.hamiltonian(0.0, z, p)) < 1e-9
        assert gen._curvature(c, 0.0, z) < 0


def test_an_integration_past_its_evaluation_budget_rejects_the_draw(smooth):
    # A stiff or stalled integration must end the draw, not run on.
    gen = ExtremalGenerator.from_homotopy(smooth, params(np.zeros(3), np.zeros(3)))
    gen.max_nfev = 5
    rng = np.random.default_rng(0)
    for _ in range(5):
        z = np.r_[0.0, rng.uniform(-0.5, 0.5, 2), rng.normal(size=3)]
        with pytest.raises(Rejected):
            gen.generate_anchored(0, z, 1.0, 2.0, rng=rng)


def test_multiple_shooting_has_an_exact_jacobian_and_the_same_answer(smooth):
    gen, samples = generate(smooth, 40, seed=1)
    e = max((s for s in samples if s.arcs and s.arcs[-1][0] == "exit"),
            key=lambda s: min(b - a for a, b, _ in s.values))
    pol = Polisher(smooth, params)
    pol.nodes, pol.continuation = 3, False
    shoot = pol._shooting(e.x0, e.xf, e.arcs)
    v = pol.carried(shoot, e, e.x0, e.xf)
    assert shoot.n_nodes == 2 * (len(e.arcs) + 1) and len(v) == shoot.n_unknowns
    assert np.max(np.abs(shoot.residual(v))) < 1e-7        # nodes on the trajectory: no defects
    w = v + 1e-6 * np.random.default_rng(0).normal(size=len(v)) * np.abs(v).clip(1e-3)
    R, J = shoot.residual_jacobian(w)
    FD = np.zeros_like(J)
    for i in range(len(w)):
        h = 1e-6 * max(1.0, abs(w[i]))
        a, b = w.copy(), w.copy()
        a[i] += h
        b[i] -= h
        FD[:, i] = (shoot.residual(a) - shoot.residual(b)) / (2 * h)
    assert np.allclose(J, FD, rtol=1e-3, atol=2e-4)
    out, msg = pol._attempt(e.x0, e.xf, e, 1)
    assert out is not None, msg
    assert abs(out.tf - e.tf) < 1e-8


def test_an_answer_can_join_the_set_it_came_from(bang, bang_set):
    _, samples = bang_set
    e = next(s for s in samples if s.arcs and s.arcs[-1][0] == "exit")
    pol = Polisher(bang, params)
    res = pol.solve(e.x0, e.xf, [e])
    data = ExtremalSet.from_extremals(samples)
    grown = data + pol.extremal_of(res, e.x0, e.xf)
    assert len(grown) == len(data) + 1 and grown.arcs[-1] == tuple(e.arcs)
    assert abs(grown.tf[-1] - e.tf) < 1e-8


def test_an_anchored_extremal_can_be_made_to_start_at_rest(smooth):
    gen = ExtremalGenerator.from_homotopy(smooth, params(np.zeros(3), np.zeros(3)))
    rng = np.random.default_rng(3)
    kept = []
    for _ in range(40):
        z = np.r_[0.0, rng.uniform(-0.5, 0.5, 2), rng.normal(size=3)]
        T = rng.uniform(1, 5)
        try:
            kept.append(gen.generate_anchored(int(rng.integers(2)), z, 0.4 * T, 0.6 * T, rng=rng,
                                              start={1: 0.0, 2: 0.0}))
        except Rejected:
            pass
    assert kept
    assert_valid(kept)
    assert max(np.abs(e.x0[1:]).max() for e in kept) < 1e-8


def test_the_quasimetric_obeys_the_triangle_inequality_untrained():
    torch = pytest.importorskip("torch")
    from auto_optimal_control import learn

    torch.manual_seed(0)
    model = learn.Quasimetric(2, 1)
    x, y, z = (torch.randn(500, 2) for _ in range(3))
    px, py, pz = (torch.randn(500, 1) for _ in range(3))
    with torch.no_grad():
        dxz = model(x, px, z, pz)
        assert torch.all(dxz <= model(x, px, y, py) + model(y, py, z, pz) + 1e-5)
        assert torch.all(model(x, px, x, px) < 1e-5)
