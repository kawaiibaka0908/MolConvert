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


def bond_length(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between points a and b (Angstroms)."""
    return float(np.linalg.norm(b - a))


def bond_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle a–b–c in degrees.

    b is the vertex (the middle atom).
    Returns a value in [0, 180].
    """
    ba = unit(a - b)
    bc = unit(c - b)
    # Clamp dot product to [-1, 1] to guard against floating-point overshoot.
    cos_theta = np.clip(np.dot(ba, bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


