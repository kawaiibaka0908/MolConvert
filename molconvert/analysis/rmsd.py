"""
RMSD and structural analysis utilities.

Functions
---------
kabsch_align          : Full Kabsch alignment — returns (R, t, rmsd).
                        coords2 (target) is rotated onto coords1 (reference).
                        H = Y^T @ X convention; matches PyMOL / Open Babel.
kabsch_superpose      : Return aligned copy of coords2 after Kabsch superposition.
kabsch_rmsd           : RMSD after Kabsch optimal superposition.
rmsd                  : Naïve RMSD between two (N, 3) arrays (no superposition).
rmsd_molecules        : RMSD between two MoleculeIC objects.
per_atom_deviation    : Per-atom distance after optional Kabsch superposition.
ic_summary            : Descriptive statistics for IC values in a MoleculeIC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.internal_coords import MoleculeIC


# ------------------------------------------------------------------ #
#  Input validation helper                                             #
# ------------------------------------------------------------------ #

def _validate_coords(coords1: np.ndarray, coords2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cast to float64, check shape (N, 3), N >= 3, and matching sizes."""
    c1 = np.asarray(coords1, dtype=np.float64)
    c2 = np.asarray(coords2, dtype=np.float64)

    if c1.ndim != 2 or c1.shape[1] != 3:
        raise ValueError(
            f"coords1 must be a (N, 3) array, got shape {c1.shape}"
        )
    if c2.ndim != 2 or c2.shape[1] != 3:
        raise ValueError(
            f"coords2 must be a (N, 3) array, got shape {c2.shape}"
        )
    if c1.shape != c2.shape:
        raise ValueError(
            f"Shape mismatch: coords1 {c1.shape} vs coords2 {c2.shape}"
        )
    if len(c1) < 3:
        raise ValueError(
            f"At least 3 point pairs are required for Kabsch alignment, got {len(c1)}"
        )
    return c1, c2


# ------------------------------------------------------------------ #
#  Kabsch alignment — canonical implementation                         #
# ------------------------------------------------------------------ #

def kabsch_align(
    coords1: np.ndarray,
    coords2: np.ndarray,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Align coords2 (target) onto coords1 (reference) via the Kabsch algorithm.

    Steps
    -----
    1. Validate inputs (N >= 3, identical shape, (N, 3) layout).
    2. Compute centroids c1, c2.
    3. Centre: X = coords1 - c1,  Y = coords2 - c2.
    4. Covariance matrix:  H = Y^T @ X.
    5. SVD:  H = U S V^T.
    6. Rotation:  R = V @ U^T.
    7. Reflection correction: if det(R) < 0, flip last column of V and recompute.
    8. Align: Y_aligned = Y @ R + c1.
    9. RMSD = sqrt( mean( sum( (coords1 - Y_aligned)^2, axis=1 ) ) ).
    10. Translation: t = c1 - c2 @ R   (so  coords2 @ R + t  gives aligned coords).

    Parameters
    ----------
    coords1 : (N, 3) reference coordinates.
    coords2 : (N, 3) target coordinates to be moved.
    debug   : If True, print centroids, H, det(R), first 3 aligned coords,
              and RMSD before/after alignment.

    Returns
    -------
    R    : (3, 3) rotation matrix.
    t    : (3,)  translation vector. Apply as: coords2 @ R + t.
    rmsd : float — RMSD in Angstroms after alignment.
    """
    # --- Step 1: Validate ---
    c1, c2 = _validate_coords(coords1, coords2)

    # --- Step 2: Centroids ---
    centroid1 = c1.mean(axis=0)
    centroid2 = c2.mean(axis=0)

    # --- Step 3: Centre ---
    X = c1 - centroid1   # reference centred
    Y = c2 - centroid2   # target centred

    if debug:
        rmsd_before = float(np.sqrt(np.mean(np.sum((c1 - c2) ** 2, axis=1))))
        print(f"[debug] centroid1 (reference) : {centroid1}")
        print(f"[debug] centroid2 (target)    : {centroid2}")
        print(f"[debug] RMSD before alignment : {rmsd_before:.6f} Å")

    # --- Step 4: Covariance matrix  H = X^T @ Y ---
    # NOTE: The spec listed H = Y^T @ X, but that convention pairs with
    # X_aligned = X @ R (rotating the reference onto the target).
    # Because we want Y_aligned = Y @ R (rotating the TARGET onto the
    # reference), the correct cross-covariance is H = X^T @ Y.
    # Using H = Y^T @ X here would produce R^T instead of R, giving
    # wrong aligned coordinates and non-zero RMSD on a pure rotation.
    H = X.T @ Y   # (3, N) @ (N, 3) → (3, 3)

    if debug:
        print(f"[debug] covariance matrix H:\n{H}")

    # --- Step 5: SVD ---
    U, _S, Vt = np.linalg.svd(H)   # H = U @ diag(S) @ Vt

    # --- Step 6: Rotation  R = V @ U^T ---
    R = Vt.T @ U.T

    # --- Step 7: Reflection correction ---
    det = float(np.linalg.det(R))
    if debug:
        print(f"[debug] det(R) before reflection check : {det:.6f}")

    if det < 0:
        # Flip the last column of V (= last row of Vt) to turn
        # an improper rotation (reflection) into a proper one.
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
        if debug:
            print(f"[debug] det(R) after  reflection fix   : {float(np.linalg.det(R)):.6f}")

    # --- Step 8: Apply rotation and translate back to reference frame ---
    Y_aligned = Y @ R + centroid1

    if debug:
        print(f"[debug] first 3 aligned coordinates:\n{Y_aligned[:3]}")

    # --- Step 9: RMSD ---
    rmsd_val = float(np.sqrt(np.mean(np.sum((c1 - Y_aligned) ** 2, axis=1))))

    if debug:
        print(f"[debug] RMSD after  alignment  : {rmsd_val:.6f} Å")

    # --- Step 10: Translation vector ---
    # Y_aligned = (c2 - centroid2) @ R + centroid1
    #           = c2 @ R  +  (centroid1 - centroid2 @ R)
    t = centroid1 - centroid2 @ R

    return R, t, rmsd_val


# ------------------------------------------------------------------ #
#  Convenience wrappers (use kabsch_align internally)                  #
# ------------------------------------------------------------------ #

