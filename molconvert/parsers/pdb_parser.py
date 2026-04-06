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

