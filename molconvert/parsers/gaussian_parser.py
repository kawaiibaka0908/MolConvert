"""
Gaussian log → Internal Coordinates parser.

Strategy
--------
Gaussian geometry-optimisation log files contain one or more coordinate
blocks headed by ``Standard orientation:`` (or ``Input orientation:``).
Each block is a table of Cartesian coordinates for every atom.

The parser:

1. Scans the file for ``Standard orientation:`` headers (falls back to
   ``Input orientation:`` if none are found).
2. Extracts the atom table from each block (atomic number → element via
   RDKit periodic table, plus X / Y / Z Cartesian coordinates).
3. Applies the ``step`` filter to select which geometry / geometries
   to return.
4. Converts each selected block into a `MoleculeIC` using the same
   sequential IC scheme as the SDF and PDB parsers.

IC depth:
  index 0   → no IC; store Cartesian position only (1st anchor)
  index 1   → bond_length only + Cartesian position (2nd anchor)
  index 2   → bond_length + bond_angle + Cartesian position (3rd anchor)
  index 3+  → bond_length + bond_angle + dihedral + Cartesian position

Because Gaussian log files have no chain / residue concept the following
defaults are used:
  chain_id     : "A"
  residue_name : "MOL"
  residue_seq  : 1

Atom names are constructed as element symbol + 1-based serial
(e.g. "O1", "H2", "H3").

Requires RDKit (already listed in setup.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem

from ..core.internal_coords import AtomIC, MoleculeIC
from ..core.geometry import (
    bond_length as _bl,
    bond_angle_deg as _ba,
    dihedral_deg as _di,
)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def parse_gaussian(
    path: str,
    step: str = "last",
) -> list[MoleculeIC]:
    """
    Parse a Gaussian log file and return one MoleculeIC per geometry step.

    Parameters
    ----------
    path : Path to the Gaussian ``.log`` / ``.out`` file.
    step : Which geometry step(s) to keep.
           ``"last"``  – only the final geometry (default).
           ``"all"``   – every geometry found in the file.
           ``"0"``, ``"1"``, … – a specific step by 0-based index.

    Returns
    -------
    List of MoleculeIC, one per selected geometry block.

    Raises
    ------
    FileNotFoundError if *path* does not exist.
    ValueError        if no coordinate block is found in the file, or if
                      a numeric *step* index is out of range.
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"Gaussian log file not found: {path}")

    stem = Path(path).stem
    lines = Path(path).read_text().splitlines()

    # -------------------------------------------------------------- #
    #  1. Locate orientation blocks                                    #
    # -------------------------------------------------------------- #
    header_indices = _find_header_indices(lines, "Standard orientation:")
    if not header_indices:
        header_indices = _find_header_indices(lines, "Input orientation:")
    if not header_indices:
        raise ValueError("No coordinate block found in Gaussian log file")

    # -------------------------------------------------------------- #
    #  2. Extract atom tables                                          #
    # -------------------------------------------------------------- #
    blocks: list[list[tuple[int, float, float, float]]] = []
    pt = Chem.GetPeriodicTable()

    for hdr_idx in header_indices:
        # Skip 4 lines: dashes, column headers, dashes
        data_start = hdr_idx + 5
        atoms: list[tuple[int, float, float, float]] = []
        for line in lines[data_start:]:
            if line.strip().startswith("---"):
                break
            parts = line.split()
            if len(parts) < 6:
                break
            # center_num, atomic_num, atomic_type, x, y, z
            atomic_num = int(parts[1])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            atoms.append((atomic_num, x, y, z))
        blocks.append(atoms)

    # -------------------------------------------------------------- #
    #  3. Apply step filter                                            #
    # -------------------------------------------------------------- #
    if step == "last":
        selected = [(len(blocks) - 1, blocks[-1])]
    elif step == "all":
        selected = list(enumerate(blocks))
    else:
        idx = int(step)
        if idx < 0 or idx >= len(blocks):
            raise ValueError(
                f"Step index {idx} out of range; file contains "
                f"{len(blocks)} geometry block(s)."
            )
        selected = [(idx, blocks[idx])]

    # -------------------------------------------------------------- #
    #  4. Build MoleculeIC objects                                     #
    # -------------------------------------------------------------- #
    molecules: list[MoleculeIC] = []

    for block_idx, atom_data in selected:
        mol_name = f"{stem}_step{block_idx}"
        mol_ic = MoleculeIC(
            name=mol_name,
            source_fmt="gaussian",
            metadata={"program": "gaussian", "step_index": block_idx},
        )

        positions: list[np.ndarray] = []

        for i, (atomic_num, x, y, z) in enumerate(atom_data):
            pos = np.array([x, y, z], dtype=float)
            positions.append(pos)

            element = pt.GetElementSymbol(atomic_num)
            serial = i + 1
            atom_name = f"{element}{serial}"

            # IC values — same depth logic as SDF / PDB parsers
            bl: Optional[float] = None
            ba: Optional[float] = None
            di: Optional[float] = None

            if i >= 1:
                bl = _bl(positions[i - 1], pos)
            if i >= 2:
                ba = _ba(positions[i - 2], positions[i - 1], pos)
            if i >= 3:
                di = _di(
                    positions[i - 3], positions[i - 2], positions[i - 1], pos
                )

            ic_atom = AtomIC(
                atom_serial=serial,
                atom_name=atom_name,
                residue_name="MOL",
                chain_id="A",
                residue_seq=1,
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

        molecules.append(mol_ic)

    return molecules


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _find_header_indices(lines: list[str], header: str) -> list[int]:
    """Return line indices where *header* appears."""
    return [i for i, line in enumerate(lines) if header in line]
