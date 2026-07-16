"""
MOL2 -> Internal Coordinates parser.

Strategy
--------
Each ``@<TRIPOS>MOLECULE`` block is parsed independently.  Atoms are
traversed in the order they appear in the ``@<TRIPOS>ATOM`` section.
The same IC scheme as the SDF and PDB parsers is used:

  index 0   -> no IC; store Cartesian position only (1st anchor)
  index 1   -> bond_length only + Cartesian position (2nd anchor)
  index 2   -> bond_length + bond_angle + Cartesian position (3rd anchor)
  index 3+  -> bond_length + bond_angle + dihedral + Cartesian position

Atom names and residue information are read directly from the MOL2 atom
block rather than being synthesised.

Element symbols are derived from the MOL2 ``atom_type`` field by
splitting on ``'.'`` and taking the first part (e.g. ``C.3`` -> ``C``).

Does NOT require RDKit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..core.internal_coords import AtomIC, MoleculeIC
from ..core.geometry import (
    bond_length as _bl,
    bond_angle_deg as _ba,
    dihedral_deg as _di,
)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def parse_mol2(path: str) -> list[MoleculeIC]:
    """
    Parse a MOL2 file and return one MoleculeIC per molecule record.

    Parameters
    ----------
    path : Path to the MOL2 file.

    Returns
    -------
    List of MoleculeIC, one per ``@<TRIPOS>MOLECULE`` block.

    Raises
    ------
    FileNotFoundError if *path* does not exist.
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"MOL2 file not found: {path}")

    stem = Path(path).stem

    text = Path(path).read_text()

    # Split on @<TRIPOS>MOLECULE — first element is empty / header text
    raw_blocks = text.split("@<TRIPOS>MOLECULE")
    mol_blocks = raw_blocks[1:]  # discard everything before the first marker

    molecules: list[MoleculeIC] = []

    for block_idx, block in enumerate(mol_blocks):
        mol_ic = _parse_block(block, stem=stem, block_idx=block_idx)
        if mol_ic is not None:
            molecules.append(mol_ic)

    return molecules


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _parse_block(
    block: str,
    stem: str,
    block_idx: int,
) -> Optional[MoleculeIC]:
    """
    Parse a single molecule block (text after ``@<TRIPOS>MOLECULE``).

    Returns None if the block cannot be parsed (e.g. missing ATOM section).
    """
    lines = block.splitlines()

    # --- molecule name and counts ---
    non_empty = [ln for ln in lines if ln.strip()]
    if len(non_empty) < 2:
        return None

    mol_name = non_empty[0].strip()
    counts_parts = non_empty[1].split()
    n_atoms = int(counts_parts[0])

    # --- locate @<TRIPOS>ATOM section within the block ---
    atom_start: Optional[int] = None
    for i, ln in enumerate(lines):
        if "@<TRIPOS>ATOM" in ln:
            atom_start = i + 1
            break

    if atom_start is None:
        return None

    # --- read n_atoms lines from the ATOM section ---
    atom_lines = lines[atom_start : atom_start + n_atoms]

    unique_name = f"{stem}_{mol_name}_{block_idx}"

    mol_ic = MoleculeIC(
        name=unique_name,
        source_fmt="mol2",
        metadata={"mol_title": mol_name},
    )

    positions: list[np.ndarray] = []

    for i, aline in enumerate(atom_lines):
        parts = aline.split()
        if len(parts) < 6:
            continue
        # MOL2 ATOM line:
        # atom_id  atom_name  x  y  z  atom_type  [res_id  res_name  charge]
        atom_name = parts[1]
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        atom_type = parts[5]
        element = atom_type.split(".")[0]

        # Optional residue fields
        res_id = int(parts[6]) if len(parts) > 6 else 1
        res_name = parts[7] if len(parts) > 7 else "MOL"

        pos = np.array([x, y, z], dtype=float)
        positions.append(pos)

        # IC values — same depth logic as SDF / PDB parser
        bl: Optional[float] = None
        ba: Optional[float] = None
        di: Optional[float] = None

        if i >= 1:
            bl = _bl(positions[i - 1], pos)
        if i >= 2:
            ba = _ba(positions[i - 2], positions[i - 1], pos)
        if i >= 3:
            di = _di(positions[i - 3], positions[i - 2], positions[i - 1], pos)

        ic_atom = AtomIC(
            atom_serial=i + 1,
            atom_name=atom_name,
            residue_name=res_name,
            chain_id="A",
            residue_seq=res_id,
            element=element,
            bond_length=bl,
            bond_angle=ba,
            dihedral=di,
            bond_to=i if i >= 1 else None,
            angle_to=i - 1 if i >= 2 else None,
            dihedral_to=i - 2 if i >= 3 else None,
            cart_x=float(pos[0]),
            cart_y=float(pos[1]),
            cart_z=float(pos[2]),
        )
        mol_ic.atoms.append(ic_atom)

    return mol_ic
