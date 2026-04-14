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

