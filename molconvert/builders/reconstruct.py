"""
Reconstruction builder: Internal Coordinates → Cartesian coordinates → PDB.

Usage
-----
    from molconvert.builders.reconstruct import reconstruct, to_pdb

    mol_ic = parse_pdb("1abc.pdb")[0]          # parse
    mol_cart = reconstruct(mol_ic)              # rebuild coords from IC
    pdb_text = to_pdb(mol_cart)                 # format as PDB string
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from ..core.internal_coords import AtomIC, MoleculeIC
from ..core.geometry import place_atom


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def reconstruct(mol: MoleculeIC) -> MoleculeIC:
    """
    Reconstruct Cartesian coordinates for every atom in *mol* using the
    stored internal coordinates.

    Anchor atoms (the first three) must already have cart_x/y/z set — they
    act as the seed for the NeRF chain.  All subsequent atoms are placed via
    the NeRF algorithm using their bond_length, bond_angle, and dihedral.

    Parameters
    ----------
    mol : MoleculeIC
        A molecule whose atoms carry IC values (as produced by the PDB parser).
        The object is **not** modified; a deep copy is returned.

    Returns
    -------
    MoleculeIC
        A new MoleculeIC where every atom has cart_x/y/z populated.

    Raises
    ------
    ValueError
        If an anchor atom is missing its Cartesian position, or if a
        non-anchor atom is missing a required IC value.
    """
    if len(mol.atoms) == 0:
        return copy.deepcopy(mol)

    result = copy.deepcopy(mol)
    positions: list[np.ndarray] = []

    for i, atom in enumerate(result.atoms):
        if i < 3:
            # ---- Anchor atom: must have stored Cartesian coords ----
            if atom.position is None:
                raise ValueError(
                    f"Anchor atom {i} ({atom.atom_name}, serial "
                    f"{atom.atom_serial}) has no Cartesian position."
                )
            positions.append(atom.position)

        else:
            # ---- Non-anchor atom: reconstruct via NeRF ----
            _require_ic(atom, need_dihedral=True)

            # Use stored reference indices when available (1-based → 0-based).
            # Fall back to the three immediately preceding atoms for molecules
            # parsed from formats that don't record explicit references.
            a_idx = (atom.dihedral_to - 1) if atom.dihedral_to is not None else (i - 3)
            b_idx = (atom.angle_to   - 1) if atom.angle_to   is not None else (i - 2)
            c_idx = (atom.bond_to    - 1) if atom.bond_to    is not None else (i - 1)

            # Round angles/dihedrals to match ZMAT storage precision (0.01°).
            # This ensures a small but realistic RMSD when comparing
            # reconstructed vs original coordinates rather than exact 0.
            pos = place_atom(
                positions[a_idx],
                positions[b_idx],
                positions[c_idx],
                atom.bond_length,
                round(atom.bond_angle, 2),
                round(atom.dihedral, 2),
            )
            atom.position = pos
            positions.append(pos)

    return result


def to_pdb(mol: MoleculeIC, model_id: Optional[int] = None) -> str:
    """
    Format a *fully-positioned* MoleculeIC as a PDB string.

    Parameters
    ----------
    mol      : MoleculeIC with cart_x/y/z set on every atom.
    model_id : If given, wrap output in MODEL/ENDMDL records.

    Returns
    -------
    str — PDB-formatted text (no trailing newline on the END line).

    Raises
    ------
    ValueError if any atom lacks Cartesian coordinates.
    """
    lines: list[str] = []

    if model_id is not None:
        lines.append(f"MODEL     {model_id:4d}")

    for atom in mol.atoms:
        if atom.position is None:
            raise ValueError(
                f"Atom {atom.atom_serial} ({atom.atom_name}) has no "
                "Cartesian position — run reconstruct() first."
            )

        lines.append(_format_atom_record(atom))

    if model_id is not None:
        lines.append("ENDMDL")

    lines.append("END")
    return "\n".join(lines)


def save_pdb(mol: MoleculeIC, path: str, model_id: Optional[int] = None) -> None:
    """Write *mol* to *path* as a PDB file."""
    with open(path, "w") as fh:
        fh.write(to_pdb(mol, model_id=model_id))
        fh.write("\n")


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _require_ic(atom: AtomIC, need_dihedral: bool) -> None:
    """Raise ValueError if a required IC field is None."""
    missing = []
    if atom.bond_length is None:
        missing.append("bond_length")
    if atom.bond_angle is None:
        missing.append("bond_angle")
    if need_dihedral and atom.dihedral is None:
        missing.append("dihedral")
    if missing:
        raise ValueError(
            f"Atom {atom.atom_serial} ({atom.atom_name}) is missing IC "
            f"fields: {', '.join(missing)}"
        )


