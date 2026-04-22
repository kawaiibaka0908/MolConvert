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

def kabsch_superpose(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """
    Return coords2 after optimal superposition onto coords1 (reference).

    Parameters
    ----------
    coords1 : (N, 3) reference.
    coords2 : (N, 3) target — this is the array that gets rotated.

    Returns
    -------
    (N, 3) array — coords2 rotated and translated to best fit coords1.
    """
    _R, _t, _rmsd = kabsch_align(coords1, coords2)
    c2 = np.asarray(coords2, dtype=np.float64)
    centroid2 = c2.mean(axis=0)
    Y = c2 - centroid2
    c1 = np.asarray(coords1, dtype=np.float64)
    centroid1 = c1.mean(axis=0)
    return Y @ _R + centroid1


def kabsch_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    RMSD between two coordinate sets after Kabsch optimal superposition.

    Rotates coords2 to best fit coords1, then returns the RMSD.
    Equivalent to what PyMOL / VMD report after alignment.

    Parameters
    ----------
    coords1 : (N, 3) reference.
    coords2 : (N, 3) target.

    Returns
    -------
    float — RMSD in Angstroms.
    """
    _R, _t, rmsd_val = kabsch_align(coords1, coords2)
    return rmsd_val


# ------------------------------------------------------------------ #
#  Naïve RMSD (no superposition)                                       #
# ------------------------------------------------------------------ #

def rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Naïve RMSD — no superposition. Assumes structures are already aligned.

    Parameters
    ----------
    coords1, coords2 : (N, 3) float arrays — must have the same shape.

    Returns
    -------
    float — RMSD in Angstroms.
    """
    c1 = np.asarray(coords1, dtype=float)
    c2 = np.asarray(coords2, dtype=float)
    if c1.shape != c2.shape:
        raise ValueError(
            f"Shape mismatch: coords1 {c1.shape} vs coords2 {c2.shape}"
        )
    if len(c1) == 0:
        return 0.0
    diff = c1 - c2
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


# ------------------------------------------------------------------ #
#  Molecule-level helpers                                              #
# ------------------------------------------------------------------ #

def rmsd_molecules(
    mol1: MoleculeIC,
    mol2: MoleculeIC,
    atom_filter: Optional[list[str]] = None,
    superpose: bool = True,
) -> float:
    """
    RMSD between two MoleculeIC objects.

    Parameters
    ----------
    mol1, mol2   : Molecules to compare — must have Cartesian positions set.
    atom_filter  : If given, only atoms whose ``atom_name`` is in this list
                   are included.  E.g. ``["CA"]`` gives Cα-only RMSD.
    superpose    : If True (default), apply Kabsch optimal superposition before
                   computing RMSD.

    Returns
    -------
    float — RMSD in Angstroms.
    """
    coords1 = _extract_coords(mol1, atom_filter)
    coords2 = _extract_coords(mol2, atom_filter)

    if len(coords1) != len(coords2):
        label = f" (filter={atom_filter})" if atom_filter else ""
        raise ValueError(
            f"Atom count mismatch{label}: "
            f"{mol1.name} has {len(coords1)}, {mol2.name} has {len(coords2)}"
        )

    if len(coords1) == 0:
        return 0.0

    if superpose:
        if len(coords1) < 3:
            # Fall back to naïve RMSD when too few atoms for Kabsch
            return rmsd(coords1, coords2)
        return kabsch_rmsd(coords1, coords2)
    return rmsd(coords1, coords2)


def per_atom_deviation(
    mol1: MoleculeIC,
    mol2: MoleculeIC,
    atom_filter: Optional[list[str]] = None,
    superpose: bool = True,
) -> list[dict]:
    """
    Per-atom distance between corresponding atoms in two molecules.

    Returns
    -------
    List of dicts, one per atom pair::

        {
            "atom_serial": int,
            "atom_name":   str,
            "residue_name": str,
            "residue_seq":  int,
            "chain_id":     str,
            "deviation":    float,   # Angstroms
        }
    """
    atoms1 = _filter_atoms(mol1, atom_filter)
    atoms2 = _filter_atoms(mol2, atom_filter)

    if len(atoms1) != len(atoms2):
        label = f" (filter={atom_filter})" if atom_filter else ""
        raise ValueError(
            f"Atom count mismatch{label}: "
            f"{mol1.name} has {len(atoms1)}, {mol2.name} has {len(atoms2)}"
        )

    coords1 = np.array([a.position for a in atoms1], dtype=float)
    coords2 = np.array([a.position for a in atoms2], dtype=float)

    if superpose and len(coords1) >= 3:
        coords2 = kabsch_superpose(coords1, coords2)

    results = []
    for i, (a1, a2) in enumerate(zip(atoms1, atoms2)):
        if a1.position is None or a2.position is None:
            raise ValueError(
                f"Atom {a1.atom_serial} ({a1.atom_name}) is missing "
                "Cartesian coordinates in one or both molecules."
            )
        dev = float(np.linalg.norm(coords1[i] - coords2[i]))
        results.append({
            "atom_serial":  a1.atom_serial,
            "atom_name":    a1.atom_name,
            "residue_name": a1.residue_name,
            "residue_seq":  a1.residue_seq,
            "chain_id":     a1.chain_id,
            "deviation":    dev,
        })

    return results


# ------------------------------------------------------------------ #
#  IC statistics                                                       #
# ------------------------------------------------------------------ #

@dataclass
class ICSummary:
    """Descriptive statistics for the IC values in a MoleculeIC."""

    n_atoms: int
    n_anchors: int

    bond_length_mean: float
    bond_length_std:  float
    bond_length_min:  float
    bond_length_max:  float

    bond_angle_mean: float
    bond_angle_std:  float
    bond_angle_min:  float
    bond_angle_max:  float

    dihedral_mean: float
    dihedral_std:  float
    dihedral_min:  float
    dihedral_max:  float

    def __str__(self) -> str:
        lines = [
            f"IC Summary — {self.n_atoms} atoms ({self.n_anchors} anchors)",
            f"  Bond lengths (Å) : mean={self.bond_length_mean:.3f}  "
            f"std={self.bond_length_std:.3f}  "
            f"[{self.bond_length_min:.3f}, {self.bond_length_max:.3f}]",
            f"  Bond angles  (°) : mean={self.bond_angle_mean:.2f}  "
            f"std={self.bond_angle_std:.2f}  "
            f"[{self.bond_angle_min:.2f}, {self.bond_angle_max:.2f}]",
            f"  Dihedrals    (°) : mean={self.dihedral_mean:.2f}  "
            f"std={self.dihedral_std:.2f}  "
            f"[{self.dihedral_min:.2f}, {self.dihedral_max:.2f}]",
        ]
        return "\n".join(lines)


