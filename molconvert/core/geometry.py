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


def dihedral_deg(a: np.ndarray, b: np.ndarray,
                 c: np.ndarray, d: np.ndarray) -> float:
    """
    Dihedral (torsion) angle a–b–c–d in degrees.

    Uses the atan2 formulation for full [-180, 180] range.
    This is numerically more stable than the acos formulation near 0° and 180°.

    Parameters
    ----------
    a, b, c, d : (3,) arrays — four consecutive bonded atoms.
    """
    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)

    if norm_n1 < 1e-10 or norm_n2 < 1e-10:
        # Atoms are collinear; dihedral is undefined — return 0.0 as convention.
        return 0.0

    n1 = n1 / norm_n1
    n2 = n2 / norm_n2

    m1 = np.cross(n1, unit(b2))

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return float(np.degrees(np.arctan2(y, x)))


# ------------------------------------------------------------------ #
#  Coordinate reconstruction (NeRF — Natural Extension Reference Frame)
# ------------------------------------------------------------------ #

