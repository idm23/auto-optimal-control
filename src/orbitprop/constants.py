"""Bodies and units.

Units are kilometres and seconds. :class:`CanonicalUnits` scales a problem so that
``mu = 1`` and a chosen length is 1, which keeps an optimizer's unknowns near one.
"""

import math
from dataclasses import dataclass, field

# EGM2008 fully normalized zonal coefficients C_n0 (tide-free), n = 2..6.
_EGM2008_C = {2: -4.84165143790815e-4, 3: 9.57161207093473e-7, 4: 5.39965866638991e-7,
              5: 6.86702913736681e-8, 6: -1.49953927978527e-7}


@dataclass(frozen=True)
class Body:
    """A central body: ``mu`` (km^3/s^2), reference ``radius`` (km), unnormalized zonal
    coefficients ``J`` ({n: J_n}, the potential's ``-J_n (R/r)^n P_n(sin phi)`` terms)
    and ``rotation`` rate (rad/s)."""

    name: str
    mu: float
    radius: float
    J: dict = field(default_factory=dict)
    rotation: float = 0.0

    def zonals(self, degree):
        """``{n: J_n}`` for n = 2..degree."""
        return {n: self.J[n] for n in range(2, degree + 1)}


EARTH = Body(
    name="Earth",
    mu=398600.4418,                 # km^3/s^2 (WGS84 / EGM2008)
    radius=6378.1363,               # km (EGM2008 reference radius)
    # J_n = -C_n0 sqrt(2n + 1) unnormalizes a fully normalized zonal coefficient.
    J={n: -c * math.sqrt(2 * n + 1) for n, c in _EGM2008_C.items()},
    rotation=7.292115e-5,           # rad/s
)


@dataclass(frozen=True)
class CanonicalUnits:
    """Lengths in units of ``length`` km and times in ``sqrt(length^3 / mu)`` s, so that
    ``mu = 1``. Velocities are then in units of the circular speed at ``length``."""

    mu: float
    length: float

    @property
    def time(self):
        return math.sqrt(self.length ** 3 / self.mu)

    @property
    def speed(self):
        return self.length / self.time

    @property
    def acceleration(self):
        return self.length / self.time ** 2

    @classmethod
    def of(cls, body, length=None):
        """Canonical units of ``body``, with ``length`` its radius by default."""
        return cls(body.mu, body.radius if length is None else length)
