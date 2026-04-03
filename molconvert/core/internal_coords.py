"""
Internal Coordinates representation.

This is the central Intermediate Representation (IR) for all format conversions.
Every parser produces a MoleculeIC; every builder consumes one.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import numpy as np


@dataclass
class AtomIC:
    """
    Internal coordinate record for a single atom.

    The first three atoms in a chain are anchor atoms: they store absolute
    Cartesian positions (cart_x/y/z) because there is no prior chain context
    to define bond_length / bond_angle / dihedral.

    All subsequent atoms store:
      - bond_length : distance to atom i-1  (Angstroms)
      - bond_angle  : angle  i-2 -- i-1 -- i  (degrees)
      - dihedral    : torsion i-3 -- i-2 -- i-1 -- i  (degrees)
    """

    # Identity
    atom_serial: int          # PDB ATOM serial number
    atom_name: str            # e.g. "CA", "N", "CB"
    residue_name: str         # e.g. "ALA", "GLY"
    chain_id: str             # e.g. "A"
    residue_seq: int          # Residue sequence number
    element: str              # e.g. "C", "N", "O"

    # Internal coordinates (None for anchor atoms)
    bond_length: Optional[float] = None   # Angstroms
    bond_angle: Optional[float] = None    # Degrees
    dihedral: Optional[float] = None      # Degrees

    # Explicit reference atom indices (1-based position in mol.atoms).
    # Used by the ZMAT converter; None means "use sequential fallback".
    bond_to: Optional[int] = None
    angle_to: Optional[int] = None
    dihedral_to: Optional[int] = None

    # Absolute Cartesian positions (set for anchor atoms, or after reconstruction)
    cart_x: Optional[float] = None
    cart_y: Optional[float] = None
    cart_z: Optional[float] = None

    @property
    def is_anchor(self) -> bool:
        """True if this atom uses absolute Cartesian coords (first 3 atoms)."""
        return self.bond_length is None

    @property
    def position(self) -> Optional[np.ndarray]:
        """Return Cartesian position as a numpy array, or None if unset."""
        if self.cart_x is None:
            return None
        return np.array([self.cart_x, self.cart_y, self.cart_z], dtype=float)

    @position.setter
    def position(self, coords: np.ndarray) -> None:
        self.cart_x = float(coords[0])
        self.cart_y = float(coords[1])
        self.cart_z = float(coords[2])

