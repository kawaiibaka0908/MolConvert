"""
PDB → Internal Coordinates parser.

Strategy
--------
Atoms are traversed in file order (chain → residue sequence → PDB atom order).
For each atom we compute:

  index 0       → no IC; store Cartesian position only (1st anchor)
  index 1       → bond_length only + Cartesian position (2nd anchor)
  index 2       → bond_length + bond_angle + Cartesian position (3rd anchor)
  index 3+      → bond_length + bond_angle + dihedral + Cartesian position

Cartesian positions are stored for all atoms so the JSON is self-contained
and RMSD comparisons can be made without running reconstruction first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from Bio import PDB
from Bio.PDB.Atom import Atom

from ..core.internal_coords import AtomIC, MoleculeIC
from ..core.geometry import (
    bond_length as _bl,
    bond_angle_deg as _ba,
    dihedral_deg as _di,
)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def parse_pdb(
    path: str,
    model_id: int = 0,
    include_hetatm: bool = False,
) -> list[MoleculeIC]:
    """
    Parse a PDB file and return a list of MoleculeIC objects.

    Parameters
    ----------
    path           : Path to the PDB file.
    model_id       : Which MODEL record to use (0 = first model).
    include_hetatm : Also return one MoleculeIC per ligand residue
                     (HETATM records, excluding water).

    Returns
    -------
    List of MoleculeIC.  A single-chain protein returns a list of length 1.
    """
    biopython_parser = PDB.PDBParser(QUIET=True)
    structure = biopython_parser.get_structure(Path(path).stem, path)

    try:
        model = structure[model_id]
    except KeyError:
        raise ValueError(f"Model id {model_id} not found in {path}")

    molecules: list[MoleculeIC] = []
    stem = Path(path).stem

    # --- ATOM records: one MoleculeIC per chain ---
    for chain in model:
        atoms = _collect_atoms(chain, hetatm=False)
        if not atoms:
            continue
        mol = _build_molecule_ic(
            atoms,
            name=f"{stem}_{chain.id}",
            source_fmt="pdb",
        )
        molecules.append(mol)

    # --- HETATM records: one MoleculeIC per unique ligand residue ---
    if include_hetatm:
        for chain in model:
            for residue in chain:
                hetflag = residue.id[0].strip()
                # BioPython marks HETATM as "H_<resname>"; water as "W"
                if not hetflag or hetflag == "W":
                    continue
                lig_atoms = list(residue.get_atoms())
                if not lig_atoms:
                    continue
                mol = _build_molecule_ic(
                    lig_atoms,
                    name=f"{stem}_{residue.resname.strip()}_{residue.id[1]}",
                    source_fmt="pdb",
                )
                molecules.append(mol)

    return molecules


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _collect_atoms(chain, hetatm: bool = False) -> list[Atom]:
    """Flatten a chain into an ordered list of atoms, skipping HETATM if asked."""
    atoms: list[Atom] = []
    for residue in chain:
        hetflag = residue.id[0].strip()
        if hetflag and not hetatm:
            continue   # skip HETATM / water for ATOM-only mode
        atoms.extend(residue.get_atoms())
    return atoms


def _build_molecule_ic(
    atoms: list[Atom],
    name: str,
    source_fmt: str,
) -> MoleculeIC:
    """
    Convert a flat ordered list of BioPython Atoms into a MoleculeIC.

    The IC for each atom is computed relative to its immediate predecessors
    in the list.  Atoms 0–2 act as anchors: their Cartesian positions seed
    the NeRF reconstruction chain.
    """
    mol = MoleculeIC(name=name, source_fmt=source_fmt)
    positions: list[np.ndarray] = []

    for i, atom in enumerate(atoms):
        pos = np.array(atom.get_vector().get_array(), dtype=float)
        positions.append(pos)

        # Compute whichever IC values are available at this depth
        bl: Optional[float] = None
        ba: Optional[float] = None
        di: Optional[float] = None

        if i >= 1:
            bl = _bl(positions[i - 1], pos)
        if i >= 2:
            ba = _ba(positions[i - 2], positions[i - 1], pos)
        if i >= 3:
            di = _di(positions[i - 3], positions[i - 2], positions[i - 1], pos)

        # Resolve element symbol; fall back to first letter of atom name
        element = (atom.element or "").strip()
        if not element:
            element = atom.name.strip().lstrip("0123456789")[0]

        residue = atom.get_parent()

        ic_atom = AtomIC(
            atom_serial=atom.serial_number,
            atom_name=atom.name.strip(),
            residue_name=residue.resname.strip(),
            chain_id=residue.get_parent().id,
            residue_seq=residue.id[1],
            element=element,
            bond_length=bl,
            bond_angle=ba,
            dihedral=di,
            bond_to=i if i >= 1 else None,          # 1-based index of previous atom
            angle_to=i - 1 if i >= 2 else None,     # 1-based index two atoms back
            dihedral_to=i - 2 if i >= 3 else None,  # 1-based index three atoms back
            cart_x=float(pos[0]),
            cart_y=float(pos[1]),
            cart_z=float(pos[2]),
        )
        mol.atoms.append(ic_atom)

    return mol
