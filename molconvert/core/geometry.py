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

def place_atom(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond_length: float,
    bond_angle_deg: float,
    dihedral_deg: float,
) -> np.ndarray:
    """
    Place a new atom D given three preceding atoms A, B, C and the IC values
    (bond_length C–D, bond_angle B–C–D, dihedral A–B–C–D).

    This is the NeRF (Natural Extension Reference Frame) algorithm. It builds
    a local coordinate frame from A, B, C and then rotates/translates into
    the global frame.

    Parameters
    ----------
    a, b, c      : Cartesian positions of the three anchor atoms.
    bond_length  : Distance C–D in Angstroms.
    bond_angle_deg : Angle B–C–D in degrees.
    dihedral_deg : Torsion A–B–C–D in degrees.

    Returns
    -------
    d : (3,) array — Cartesian position of the new atom D.
    """
    # Convert angles to radians for numpy trig
    theta = np.radians(bond_angle_deg)   # bond angle
    phi   = np.radians(dihedral_deg)     # dihedral

    # D in the local reference frame (before rotation into global frame).
    #
    # Our rotation matrix M has columns [-bc, nbc, n], so the local x-axis
    # points from C *toward* B (backward direction).  This changes both signs
    # relative to the "forward-x" NeRF convention found in most papers:
    #
    #   x-component: cos(B-C-D) = dot(unit(b-c), unit(d-c)) = d_local[0]/r
    #                → d_local[0] = +r * cos(theta)   [NOT -cos]
    #
    #   z-component: the backward x-axis flips the handedness of the frame,
    #                so phi must be negated to match our dihedral_deg convention.
    #                → d_local[2] = -r * sin(theta) * sin(phi)
    d_local = np.array([
         bond_length * np.cos(theta),
         bond_length * np.sin(theta) * np.cos(phi),
        -bond_length * np.sin(theta) * np.sin(phi),
    ])

    # Build the rotation matrix M that maps local → global frame
    bc = unit(c - b)

    # Normal to the a-b-c plane.
    # If a, b, c are collinear the cross product is zero — fall back to an
    # arbitrary vector perpendicular to bc so reconstruction can still proceed.
    n_raw = np.cross(b - a, bc)
    if np.linalg.norm(n_raw) < 1e-10:
        # Pick a reference vector that is not parallel to bc
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(bc, ref)) > 0.9:   # bc ≈ x-axis → use y-axis instead
            ref = np.array([0.0, 1.0, 0.0])
        n_raw = np.cross(bc, ref)
    n = unit(n_raw)

    nbc = np.cross(n, bc)               # completes right-hand frame

    # Columns of M: [-bc | nbc | n]  (matches the local frame axes above)
    M = np.column_stack([-bc, nbc, n])

    # Rotate and translate
    d = M @ d_local + c
    return d
