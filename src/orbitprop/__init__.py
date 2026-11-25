"""Symbolic orbital dynamics: force models and equations of motion in SymPy.

Everything is a SymPy expression, so it can be differentiated exactly (an indirect
method needs the dynamics' Jacobian for its costate equations, a shooting method the
variational equations) and compiled. Numbers come from the same expressions through
:func:`numeric`, so the model that is optimized and the model that checks the answer
are one.

The package never imports ``auto_optimal_control``: it is a base that the optimal
control problems are built on, and is meant to move to its own repository.

- :mod:`orbitprop.constants`: bodies (Earth: EGM2008 zonal harmonics) and canonical
  units.
- :mod:`orbitprop.elements`: Keplerian, Cartesian and modified equinoctial elements,
  symbolic and numeric.
- :mod:`orbitprop.forces`: accelerations (point mass, zonal harmonics) as SymPy
  vectors, and their projection on the radial / transverse / normal frame.
- :mod:`orbitprop.dynamics`: equations of motion, Cartesian and modified equinoctial
  (Gauss's variational equations), with any extra acceleration (a thrust) added.
"""

from orbitprop.constants import EARTH, Body, CanonicalUnits
from orbitprop.dynamics import cartesian, equinoctial
from orbitprop.elements import (cartesian_to_equinoctial, equinoctial_to_cartesian,
                                equinoctial_to_keplerian, keplerian_to_equinoctial,
                                symbolic_equinoctial_to_cartesian)
from orbitprop.forces import Gravity, numeric, rtn_basis

__all__ = ["EARTH", "Body", "CanonicalUnits", "Gravity", "cartesian", "equinoctial",
           "cartesian_to_equinoctial", "equinoctial_to_cartesian", "equinoctial_to_keplerian",
           "keplerian_to_equinoctial", "symbolic_equinoctial_to_cartesian", "numeric",
           "rtn_basis"]
