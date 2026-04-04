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

    def to_dict(self) -> dict:
        return {
            "atom_serial": self.atom_serial,
            "atom_name": self.atom_name,
            "residue_name": self.residue_name,
            "chain_id": self.chain_id,
            "residue_seq": self.residue_seq,
            "element": self.element,
            "bond_length": self.bond_length,
            "bond_angle": self.bond_angle,
            "dihedral": self.dihedral,
            "bond_to": self.bond_to,
            "angle_to": self.angle_to,
            "dihedral_to": self.dihedral_to,
            "cart_x": self.cart_x,
            "cart_y": self.cart_y,
            "cart_z": self.cart_z,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AtomIC:
        # Merge with defaults for fields added after the original schema.
        data = {
            **d,
            "bond_to": d.get("bond_to"),
            "angle_to": d.get("angle_to"),
            "dihedral_to": d.get("dihedral_to"),
        }
        return cls(**data)


@dataclass
class MoleculeIC:
    """
    Internal coordinate representation of an entire molecule or chain.

    Attributes
    ----------
    name        : Molecule / structure identifier (e.g. PDB ID or filename stem)
    source_fmt  : Original file format ("pdb" | "sdf")
    atoms       : Ordered list of AtomIC records (ordering matters for reconstruction)
    metadata    : Arbitrary key-value pairs (e.g. header info from PDB)
    """

    name: str
    source_fmt: str
    atoms: list[AtomIC] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    #  Convenience accessors                                               #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.atoms)

    def get_positions(self) -> np.ndarray:
        """
        Return Cartesian coordinates as an (N, 3) array.
        Raises ValueError if any atom has no position set.
        """
        positions = []
        for atom in self.atoms:
            if atom.position is None:
                raise ValueError(
                    f"Atom {atom.atom_serial} ({atom.atom_name}) has no "
                    "Cartesian position. Run reconstruction first."
                )
            positions.append(atom.position)
        return np.array(positions, dtype=float)

    def get_atom_names(self) -> list[str]:
        return [a.atom_name for a in self.atoms]

    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source_fmt": self.source_fmt,
            "metadata": self.metadata,
            "atoms": [a.to_dict() for a in self.atoms],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str) -> None:
        with open(path, "w") as fh:
            fh.write(self.to_json())

    @classmethod
    def from_dict(cls, d: dict) -> MoleculeIC:
        atoms = [AtomIC.from_dict(a) for a in d["atoms"]]
        return cls(
            name=d["name"],
            source_fmt=d["source_fmt"],
            atoms=atoms,
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def load_json(cls, path: str) -> MoleculeIC:
        with open(path) as fh:
            return cls.from_dict(json.load(fh))
