import numpy as np

from auto_optimal_control.diagnostics import Diagnostics
from auto_optimal_control.pipeline import (PREDICT, STEP_FAILED_SIZE, Pipeline, StepContext,
                                           default_pipelines, is_on)


class _H:
    """The little of a Homotopy that the step's rules read."""

    def __init__(self, **switches):
        from auto_optimal_control.rules import RULES
        for r in RULES:
            name = r.switch.partition("=")[0]
            if not name.startswith("MultiArcShooting.") and r.switch != "always":
                setattr(self, name, False)
        self.portfolio = []
        self.contraction_target = 0.25
        self.diagnostics = Diagnostics()
        for k, v in switches.items():
            setattr(self, k, v)

    def _detail(self, *args):
        pass


def _ctx(h, **kw):
    v0 = np.array([1.0, 2.0])
    arcs = [("boundary", 0)]
    base = dict(frac=0.5, f=0.6, ds=0.1, ds_min=1e-6, v0=v0, arcs=arcs,
                prev=(0.4, np.array([0.9, 1.8]), arcs), prev2=(0.3, np.array([0.8, 1.6]), arcs),
                moves_weight=True)
    base.update(kw)
    return StepContext(h, **base)


def test_the_predictor_composes_its_rules_best_first():
    off, on = _H(), _H(quadratic_predictor=True)
    p = default_pipelines()[PREDICT]
    secant_and_last = p.collect(_ctx(off))
    assert len(secant_and_last) == 2 and np.allclose(secant_and_last[0], [1.1, 2.2])
    quad = p.collect(_ctx(on))
    assert len(quad) == 3 and np.allclose(quad[0], [1.1, 2.2])   # exact on a line
    assert not is_on(p.rules[0], off) and is_on(p.rules[0], on)


def test_a_failed_step_takes_the_first_rule_that_answers():
    h = _H(locate_births=True)
    p = default_pipelines()[STEP_FAILED_SIZE]
    assert p.first(_ctx(h)) == 0.05                       # halving
    assert np.isclose(p.first(_ctx(h, located=0.3)), 0.03)  # R16: the birth's fraction
    assert h.diagnostics.as_dict()["rules"].get("R16") == 1


def test_a_rule_of_ones_own_slots_into_a_hook():
    class Third:
        name, hook, fires = None, STEP_FAILED_SIZE, False

        def act(self, ctx):
            return ctx.ds / 3

    p = Pipeline(STEP_FAILED_SIZE, [Third()] + default_pipelines()[STEP_FAILED_SIZE].rules)
    assert np.isclose(p.first(_ctx(_H())), 0.1 / 3)
