"""The solver's rules, each with its specification (``docs/notes/structure_certificates.md``,
"How a rule is specified").

A rule acts at one *hook* of the active-set route, on a *trigger*. Its *soundness*
is what it provably preserves (always: the answers accepted), from the lemma named.
Its *effectiveness* is either a lemma about the solver's own problem (that the path
is regular in some parameter, that Newton converges from the seed) or the word
``"heuristic"``. ``targets`` names the failure class of
:class:`~auto_optimal_control.diagnostics.Diagnostics` the rule should reduce; it
is judged on that class first and on the number of cases solved second.

``switch`` is where the rule is turned on: an attribute of the
:class:`~auto_optimal_control.homotopy.Homotopy` (``"offset_start=staggered"`` for
a value) or a class default of :class:`~auto_optimal_control.shooting.MultiArcShooting`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    name: str
    hook: str
    trigger: str
    action: str
    lemmas: tuple
    effectiveness: str
    targets: str
    switch: str


RULES = (
    Rule("R1", "route start", "every solve", "stagger the offsets' starting values at random",
         ("C", "D"), "Lemma D (births separate with probability one)", "coincident births",
         "offset_start=staggered"),
    Rule("R2", "step stalled", "a step fails with ds < 1e-3 beside a young order-1 arc",
         "step in the arc's width with the route parameter solved for", ("E",),
         "Lemma E (the branch is smooth in the width)", "stalled",
         "width_steps"),
    Rule("R4", "solve failed", "the forward solve is not exact and C6 holds",
         "solve the time-reversed problem and map its answer back", ("F",), "heuristic",
         "not exact", "reversal"),
    Rule("R5", "consistent step", "an order-1 arc's multiplier rises",
         "split the arc around the rise", ("H",), "heuristic", "mu rises (Lemma H)",
         "split_rising"),
    Rule("R6", "newborn arc", "a violation is to be brought in as an order-1 arc",
         "pull the arc's ends in to the holdable stretch about the peak", ("I",), "heuristic",
         "seed failed: riding_failed", "holdable_births"),
    Rule("R7", "junction rows", "every order-1 junction under C2",
         "use dg/dt = 0 under the far side's control", ("J",), "heuristic",
         "converged off the edge", "MultiArcShooting.rate_junctions_default"),
    Rule("R8", "seed failed", "a seed's riding fails partway",
         "pull the arc's exit in to the edge", ("I", "J"), "heuristic",
         "seed failed: riding_failed", "edge_exits"),
    Rule("R9", "evaluation", "a single limit's riding fails before its exit",
         "continue past the edge on the free law", ("K",), "heuristic", "riding_failed",
         "MultiArcShooting.past_edge_default"),
)

BY_NAME = {r.name: r for r in RULES}


def enabled(h):
    """Which rules are on for Homotopy ``h``: ``{name: bool}``."""
    from .shooting import MultiArcShooting

    out = {}
    for r in RULES:
        name, _, value = r.switch.partition("=")
        if name.startswith("MultiArcShooting."):
            cur = getattr(MultiArcShooting, name.split(".", 1)[1])
        else:
            cur = getattr(h, name)
        out[r.name] = (str(cur) == value) if value else bool(cur)
    return out


def set_rule(h, name, on):
    """Turn rule ``name`` on or off for Homotopy ``h`` (class defaults for the shooting's)."""
    from .shooting import MultiArcShooting

    attr, _, value = BY_NAME[name].switch.partition("=")
    new = (value if on else {"offset_start": "together"}.get(attr, None)) if value else bool(on)
    if attr.startswith("MultiArcShooting."):
        setattr(MultiArcShooting, attr.split(".", 1)[1], new)
    else:
        setattr(h, attr, new)
