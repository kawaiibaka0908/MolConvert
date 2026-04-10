"""
Z-matrix (ZMAT) external format → MoleculeIC (JSON internal IR).

This is the reverse of json_to_zmat. It parses a .zmat file produced by
json_to_zmat (or any conforming ZMAT) and returns a MoleculeIC ready for
reconstruction or further conversion.

ZMAT file structure expected
-----------------------------
    ZMAT <molecule_name>
    # source_fmt <fmt>
    # n_atoms <N>
    # anchor <1-based-idx> <x> <y> <z>
    <idx>  <name>  <resname>  <chain>  <resseq>
    <idx>  <name>  <resname>  <chain>  <resseq>  <bond_to>  <bond_len>
    <idx>  <name>  <resname>  <chain>  <resseq>  <bond_to>  <bond_len>  <angle_to>  <angle>
    <idx>  <name>  <resname>  <chain>  <resseq>  <bond_to>  <bond_len>  <angle_to>  <angle>  <dihedral_to>  <dihedral>
    ...
    END

Public API
----------
    zmat_to_molecule(text)   -> MoleculeIC   (parse a ZMAT string)
    load_zmat(path)          -> MoleculeIC   (parse a .zmat file)
"""

from __future__ import annotations
from ..core.internal_coords import AtomIC, MoleculeIC


