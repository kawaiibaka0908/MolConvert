"""
MoleculeIC → SDF V2000 builder.

Bond connectivity is inferred from interatomic distances using standard
covalent radii with a 30 % tolerance. Bond orders are always written as
1 (single) — the IC representation does not carry bond-order information.

Public API
----------
    molecule_to_sdf(mol)              -> str          (one SDF record)
    save_sdf(mol, path)               -> None
    molecules_to_sdf(mols)            -> str          (multi-record SDF)
    save_sdf_multi(mols, path)        -> None
"""

from __future__ import annotations
from ..core.internal_coords import AtomIC, MoleculeIC


# Covalent radii in Ångströms (Alvarez 2008, doi:10.1039/b801115j)
_COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71,
    "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Fe": 1.32, "Co": 1.26,
    "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Br": 1.20, "I": 1.39,
    "Mn": 1.61, "Se": 1.20,
}
_DEFAULT_RADIUS = 0.90   # fallback for unknown elements
_BOND_TOLERANCE = 1.3    # accept distances up to 130 % of covalent-radii sum


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def molecule_to_sdf(mol: MoleculeIC) -> str:
    """
    Convert a fully-positioned MoleculeIC to a single SDF V2000 record.

    Bond connectivity is inferred from interatomic distances.
    All bonds are written as type 1 (single).

    Raises
    ------
    ValueError if any atom lacks Cartesian coordinates.
    """
    _require_positions(mol)

    mol_title = mol.metadata.get("mol_title", mol.name)
    bonds     = _infer_bonds(mol.atoms)
    n_atoms   = len(mol.atoms)
    n_bonds   = len(bonds)

    lines: list[str] = []

    # ---- Header block (3 lines) ----
    lines.append(mol_title)
    lines.append("     molconvert          3D")
    lines.append("")

    # ---- Counts line ----
    lines.append(f"{n_atoms:3d}{n_bonds:3d}  0  0  0  0  0  0  0  0999 V2000")

    # ---- Atom block ----
    for atom in mol.atoms:
        lines.append(_atom_line(atom))

    # ---- Bond block ----
    for a1_idx, a2_idx, btype in bonds:
        lines.append(f"{a1_idx:3d}{a2_idx:3d}{btype:3d}  0")

    lines.append("M  END")
    lines.append("$$$$")

    return "\n".join(lines)


def save_sdf(mol: MoleculeIC, path: str) -> None:
    """Write one MoleculeIC to a .sdf file (single record)."""
    with open(path, "w") as fh:
        fh.write(molecule_to_sdf(mol))
        fh.write("\n")


