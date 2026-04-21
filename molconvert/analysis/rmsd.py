"""
RMSD and structural analysis utilities.

Functions
---------
kabsch_align          : Full Kabsch alignment — returns (R, t, rmsd).
                        coords2 (target) is rotated onto coords1 (reference).
                        H = Y^T @ X convention; matches PyMOL / Open Babel.
kabsch_superpose      : Return aligned copy of coords2 after Kabsch superposition.
kabsch_rmsd           : RMSD after Kabsch optimal superposition.
rmsd                  : Naïve RMSD between two (N, 3) arrays (no superposition).
rmsd_molecules        : RMSD between two MoleculeIC objects.
per_atom_deviation    : Per-atom distance after optional Kabsch superposition.
ic_summary            : Descriptive statistics for IC values in a MoleculeIC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.internal_coords import MoleculeIC


# ------------------------------------------------------------------ #
#  Input validation helper                                             #
# ------------------------------------------------------------------ #

