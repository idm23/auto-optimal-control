"""What a solve ran into: failures by class, rules fired, answers rejected, and snapshots.

:class:`~auto_optimal_control.homotopy.Homotopy` keeps one :class:`Diagnostics`
per solve and returns it as ``HomotopyResult.diagnostics``. The counters are the
failure taxonomy that solver rules are judged on
(``docs/notes/structure_certificates.md``, "How a rule is specified"):

- ``evaluations``: failed residual evaluations, by
  :attr:`~auto_optimal_control.shooting.ShootingError.failure`;
- ``steps``: the route's steps, by outcome (``"converged"``, or why not);
- ``rejected``: answers that converged but failed an acceptance check, by check;
- ``rules``: how often each rule fired;
- ``oracles``: lemma conclusions found false on a path solution (when
  ``Homotopy.check_lemmas`` is on).

A *snapshot* is the state of the route at a failed step, written when
``Homotopy.snapshot_dir`` is set: enough to rebuild the step's shooting problem
and seed and try a rule on it in seconds (:func:`replay`).
"""

import os
import pickle

import numpy as np


class Diagnostics:
    """Counters of one solve (see the module docstring)."""

    def __init__(self):
        self.evaluations = {}
        self.steps = {}
        self.rejected = {}
        self.rules = {}
        self.oracles = {}
        self.snapshots = []
        self._per_class = {}

    @staticmethod
    def _add(table, key, n=1):
        table[key] = table.get(key, 0) + n

    def step(self, outcome):
        self._add(self.steps, outcome)

    def reject(self, reason):
        self._add(self.rejected, reason)

    def fired(self, rule):
        self._add(self.rules, rule)

    def oracle(self, lemma):
        self._add(self.oracles, lemma)

    def snapshot(self, directory, tag, failure, payload, per_class=2):
        """Write ``payload`` for a failure of class ``failure`` (at most ``per_class``
        per class and solve); returns the path or None."""
        if directory is None or self._per_class.get(failure, 0) >= per_class:
            return None
        self._add(self._per_class, failure)
        os.makedirs(directory, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "@._" else "_" for ch in f"{tag or 'solve'}-{failure}")
        path = os.path.join(directory, f"{safe}-{self._per_class[failure]}.pkl")
        with open(path, "wb") as fo:
            pickle.dump({"failure": failure, "tag": tag, **payload}, fo)
        self.snapshots.append(path)
        return path

    def as_dict(self):
        return {"evaluations": dict(self.evaluations), "steps": dict(self.steps),
                "rejected": dict(self.rejected), "rules": dict(self.rules),
                "oracles": dict(self.oracles), "snapshots": list(self.snapshots)}


def load_snapshot(path):
    with open(path, "rb") as fi:
        return pickle.load(fi)


def rebuild(h, snap, at="attempt"):
    """The shooting problem and seed of a snapshot: ``(shoot, guess, le, off)``.

    ``at="attempt"`` rebuilds the failed step (its structure and seed at the step's
    target), ``at="before"`` the consistent solution the step started from.
    ``h`` is a :class:`~auto_optimal_control.homotopy.Homotopy` for the same problem."""
    h.params = np.asarray(snap["params"], dtype=float)
    side = snap[at]
    arcs, values, base = side["arcs"], side["values"], side["base"]
    le, off = side["le"], np.asarray(side["off"], dtype=float)
    entries, values = h._mas_entries(arcs, values)
    sh = h._smoothed(entries, 10.0 ** le if h.bang_bang else None, off)
    guess = sh.pack_arcs(base, values) if arcs else np.asarray(base, dtype=float)
    return sh, guess, le, off


def replay(h, snap, at="attempt"):
    """Solve a snapshot's step again with ``h``'s current rules: ``(shoot, sol, failures)``,
    with ``failures`` the evaluations that failed, by class."""
    from .shooting import SingleShooting

    h.params = np.asarray(snap["params"], dtype=float)
    side = snap[at]
    sink, SingleShooting.sink = SingleShooting.sink, {}
    try:
        sh, r = h._solve_structure(side["arcs"], side["values"], side["base"], side["le"],
                                   np.asarray(side["off"], dtype=float), snap["path_tol"])
        return sh, r, dict(SingleShooting.sink)
    finally:
        SingleShooting.sink = sink
