"""
SDF → Internal Coordinates parser.

Strategy
--------
Each molecule record in the SDF file is read with RDKit.  Atoms are
traversed in the order they appear in the atom block.  The same IC
scheme as the PDB parser is used:

  index 0   → no IC; store Cartesian position only (1st anchor)
  index 1   → bond_length only + Cartesian position (2nd anchor)
  index 2   → bond_length + bond_angle + Cartesian position (3rd anchor)
  index 3+  → bond_length + bond_angle + dihedral + Cartesian position

Because SDF has no chain / residue concept, the following defaults are used:
  chain_id     : "A"
  residue_name : molecule name truncated to 3 chars (or "LIG" if unnamed)
  residue_seq  : 1

Atom names are constructed as element symbol + 1-based index (e.g. "C1",
"O2") since SDF atom blocks carry no named-atom field.

Requires RDKit (already listed in setup.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ..core.internal_coords import AtomIC, MoleculeIC
from ..core.geometry import (
    bond_length as _bl,
    bond_angle_deg as _ba,
    dihedral_deg as _di,
)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def parse_sdf(
    path: str,
    remove_hydrogens: bool = False,
    skip_invalid: bool = True,
) -> list[MoleculeIC]:
    """
    Parse an SDF file and return one MoleculeIC per molecule record.

    Parameters
    ----------
    path             : Path to the SDF file.
    remove_hydrogens : If True, hydrogen atoms are stripped before IC
                       computation (useful for heavy-atom-only analysis).
    skip_invalid     : If True, molecule records that RDKit cannot parse
                       are silently skipped.  If False, a ValueError is
                       raised on the first bad record.

    Returns
    -------
    List of MoleculeIC, one per valid molecule record in the file.

    Raises
    ------
    FileNotFoundError if *path* does not exist.
    ValueError        if a molecule fails to parse and skip_invalid=False,
                      or if no 3-D conformer is present.
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"SDF file not found: {path}")

    stem = Path(path).stem

    supplier = Chem.SDMolSupplier(path, removeHs=remove_hydrogens, sanitize=True)

    molecules: list[MoleculeIC] = []

    for mol_idx, rdmol in enumerate(supplier):
        if rdmol is None:
            if skip_invalid:
                continue
            raise ValueError(
                f"Molecule record {mol_idx} in '{path}' could not be parsed."
            )

        # Ensure a 3-D conformer exists
        if rdmol.GetNumConformers() == 0:
            if skip_invalid:
                continue
            raise ValueError(
                f"Molecule record {mol_idx} ('{_mol_name(rdmol, mol_idx)}') "
                "has no 3-D coordinates."
            )

        mol_name = _mol_name(rdmol, mol_idx)
        unique_name = f"{stem}_{mol_name}_{mol_idx}"

        mol_ic = _build_molecule_ic(rdmol, name=unique_name, mol_name=mol_name)
        molecules.append(mol_ic)

    return molecules


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _mol_name(rdmol, fallback_idx: int) -> str:
    """Return the molecule's title line, or 'mol<N>' if blank."""
    name = rdmol.GetProp("_Name").strip() if rdmol.HasProp("_Name") else ""
    return name if name else f"mol{fallback_idx}"


def _residue_name(mol_name: str) -> str:
    """
    Derive a 3-char residue-name-style label from the molecule title.
    Truncates to 3 characters and uppercases.
    """
    cleaned = "".join(c for c in mol_name if c.isalnum())
    label = cleaned[:3].upper()
    return label if label else "LIG"


def _build_molecule_ic(rdmol, name: str, mol_name: str) -> MoleculeIC:
    """
    Convert an RDKit Mol (with a 3-D conformer) into a MoleculeIC.

    Atoms are traversed in RDKit atom-block order.  IC values are
    computed relative to the immediately preceding atoms, matching
    the PDB parser convention.
    """
    conf = rdmol.GetConformer()
    res_name = _residue_name(mol_name)

    mol_ic = MoleculeIC(
        name=name,
        source_fmt="sdf",
        metadata={"mol_title": mol_name},
    )

    positions: list[np.ndarray] = []

    for i, atom in enumerate(rdmol.GetAtoms()):
        rdpos = conf.GetAtomPosition(i)
        pos = np.array([rdpos.x, rdpos.y, rdpos.z], dtype=float)
        positions.append(pos)

        # IC values — same depth logic as PDB parser
        bl: Optional[float] = None
        ba: Optional[float] = None
        di: Optional[float] = None

        if i >= 1:
            bl = _bl(positions[i - 1], pos)
        if i >= 2:
            ba = _ba(positions[i - 2], positions[i - 1], pos)
        if i >= 3:
            di = _di(positions[i - 3], positions[i - 2], positions[i - 1], pos)

        element = atom.GetSymbol()
        atom_name = f"{element}{i + 1}"   # e.g. "C1", "O3"

        ic_atom = AtomIC(
            atom_serial=i + 1,
            atom_name=atom_name,
            residue_name=res_name,
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

    return mol_ic
