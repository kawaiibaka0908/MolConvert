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

