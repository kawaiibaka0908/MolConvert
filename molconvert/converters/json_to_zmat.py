"""
MoleculeIC (JSON internal IR) → Z-matrix (ZMAT) external format.

ZMAT is the *external* representation for internal coordinates.
JSON / MoleculeIC is always the *internal* IR — this module is a view layer only.

ZMAT file structure
-------------------
    ZMAT <molecule_name>
    # source_fmt <fmt>
    # n_atoms <N>
    # anchor <1-based-idx> <x> <y> <z>      ← one line per anchor atom (first 3)
    <idx>  <name>  <resname>  <chain>  <resseq>
    <idx>  <name>  <resname>  <chain>  <resseq>  <bond_to>  <bond_len>
    <idx>  <name>  <resname>  <chain>  <resseq>  <bond_to>  <bond_len>  <angle_to>  <angle>
    <idx>  <name>  <resname>  <chain>  <resseq>  <bond_to>  <bond_len>  <angle_to>  <angle>  <dihedral_to>  <dihedral>
    ...
    END

Indices in the data lines are 1-based positions in the atom list (not PDB serials).
Anchor Cartesian coordinates are stored in `# anchor` header lines so the file is
self-contained for a full round-trip back to PDB via zmat_to_json.

Public API
----------
    molecule_to_zmat(mol)        -> str
    save_zmat(mol, path)         -> None
"""

from __future__ import annotations
from ..core.internal_coords import MoleculeIC


