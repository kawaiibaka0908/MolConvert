"""
Low-level geometry routines for molecular structure math.

All functions operate on numpy arrays. No dependencies on internal data
classes — this module is intentionally kept pure so it can be tested in
isolation and reused freely.

Conventions
-----------
- Positions are (3,) float64 numpy arrays in Angstroms.
- Angles are returned and accepted in DEGREES unless the function name
  explicitly says _rad.
"""

import numpy as np


# ------------------------------------------------------------------ #
#  Basic vector math                                                   #
# ------------------------------------------------------------------ #

def unit(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of v. Raises ValueError if v is zero-length."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        raise ValueError(f"Cannot normalise a near-zero vector: {v}")
    return v / norm


