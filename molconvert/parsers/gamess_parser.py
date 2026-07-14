"""
GAMESS output → Internal Coordinates parser.

Strategy
--------
GAMESS output files contain one or more coordinate blocks headed by the
line ``COORDINATES OF ALL ATOMS ARE (ANGS)`` (or ``(BOHR)``).  Each block
lists atoms with their nuclear charge and Cartesian position.

The ``step`` parameter controls which coordinate block(s) are converted:

  "last"  → keep only the final block  (default, matches optimised geometry)
  "all"   → keep every block
  "0", "1", … → keep the block at that 0-based index

For each kept block the parser builds a MoleculeIC using the same
sequential IC scheme as the SDF and PDB parsers:

  index 0   → no IC; Cartesian position only (1st anchor)
  index 1   → bond_length only               (2nd anchor)
  index 2   → bond_length + bond_angle        (3rd anchor)
  index 3+  → bond_length + bond_angle + dihedral

Requires RDKit (for element-symbol lookup from atomic number).
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
#  Constants                                                           #
# ------------------------------------------------------------------ #

_BOHR_TO_ANGSTROM = 0.529177210903

_COORD_HEADER = "COORDINATES OF ALL ATOMS"


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def parse_gamess(
    path: str,
    step: str = "last",
) -> list[MoleculeIC]:
    """
    Parse a GAMESS output file and return a list of MoleculeIC objects.

    Parameters
    ----------
    path : Path to the GAMESS output file.
    step : Which coordinate block(s) to keep:
           ``"last"``  – only the final block (default).
           ``"all"``   – every block found in the file.
           ``"0"``, ``"1"``, … – the block at that 0-based index.

    Returns
    -------
    List of MoleculeIC, one per kept coordinate block.

    Raises
    ------
    FileNotFoundError if *path* does not exist.
    ValueError        if no coordinate block is found in the file,
                      or if a numeric *step* index is out of range.
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"GAMESS output file not found: {path}")

    stem = Path(path).stem
    lines = Path(path).read_text().splitlines()

    # --- locate coordinate block headers --------------------------------
    header_indices = [
        i for i, line in enumerate(lines) if _COORD_HEADER in line
    ]

    if not header_indices:
        raise ValueError("No coordinate block found in GAMESS output file")

    # --- parse each coordinate block ------------------------------------
    raw_blocks: list[tuple[str, list[tuple[str, float, float, float]]]] = []

    for hdr_idx in header_indices:
        header_line = lines[hdr_idx]
        units = "bohr" if "(BOHR)" in header_line.upper() else "angstroms"

        # Skip column labels + dashes (2 lines after the header)
        data_start = hdr_idx + 3
        atoms: list[tuple[str, float, float, float]] = []

        for j in range(data_start, len(lines)):
            tokens = lines[j].split()
            if len(tokens) < 4:
                break
            label = tokens[0]
            charge_float = tokens[1]
            x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])

            # Convert Bohr → Angstroms if needed
            if units == "bohr":
                x *= _BOHR_TO_ANGSTROM
                y *= _BOHR_TO_ANGSTROM
                z *= _BOHR_TO_ANGSTROM

            # Element symbol from nuclear charge
            atomic_num = int(round(float(charge_float)))
            element = Chem.GetPeriodicTable().GetElementSymbol(atomic_num)

            atoms.append((element, x, y, z))

        raw_blocks.append((units, atoms))

    # --- apply step filter ----------------------------------------------
    if step == "last":
        selected = [(len(raw_blocks) - 1, raw_blocks[-1])]
    elif step == "all":
        selected = list(enumerate(raw_blocks))
    else:
        idx = int(step)
        if idx < 0 or idx >= len(raw_blocks):
            raise ValueError(
                f"Step index {idx} out of range "
                f"(file contains {len(raw_blocks)} coordinate block(s))"
            )
        selected = [(idx, raw_blocks[idx])]

    # --- build MoleculeIC for each kept block ---------------------------
    molecules: list[MoleculeIC] = []

    for block_idx, (units, atom_data) in selected:
        mol_name = f"{stem}_{block_idx}"
        mol_ic = MoleculeIC(
            name=mol_name,
            source_fmt="gamess",
            metadata={
                "program": "gamess",
                "units": units,
                "step_index": block_idx,
            },
        )

        positions: list[np.ndarray] = []

        for i, (element, x, y, z) in enumerate(atom_data):
            pos = np.array([x, y, z], dtype=float)
            positions.append(pos)

            serial = i + 1
            atom_name = f"{element}{serial}"

            # IC values — same depth logic as SDF/PDB parsers
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
