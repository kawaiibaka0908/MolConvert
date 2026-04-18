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


def molecules_to_sdf(mols: list[MoleculeIC]) -> str:
    """Convert a list of MoleculeIC objects to a multi-record SDF string."""
    return "\n".join(molecule_to_sdf(m) for m in mols)


def save_sdf_multi(mols: list[MoleculeIC], path: str) -> None:
    """Write multiple MoleculeIC objects to a single multi-record SDF file."""
    with open(path, "w") as fh:
        fh.write(molecules_to_sdf(mols))
        fh.write("\n")


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _require_positions(mol: MoleculeIC) -> None:
    for atom in mol.atoms:
        if atom.cart_x is None:
            raise ValueError(
                f"Atom {atom.atom_serial} ({atom.atom_name}) has no "
                "Cartesian position — run reconstruct() first."
            )


def _atom_line(atom: AtomIC) -> str:
    """Format one AtomIC as a V2000 atom block line."""
    elem = f"{atom.element:<3s}"
    return (
        f"{atom.cart_x:10.4f}"
        f"{atom.cart_y:10.4f}"
        f"{atom.cart_z:10.4f}"
        f" {elem}"
        " 0  0  0  0  0  0  0  0  0  0  0  0"
    )


def _infer_bonds(atoms: list[AtomIC]) -> list[tuple[int, int, int]]:
    """
    Infer bond pairs from interatomic distances.

    Returns list of (atom1_1based, atom2_1based, bond_type=1).
    """
    bonds: list[tuple[int, int, int]] = []
    n = len(atoms)

    # Pre-extract positions and radii to avoid repeated attribute lookups
    xs = [a.cart_x for a in atoms]
    ys = [a.cart_y for a in atoms]
    zs = [a.cart_z for a in atoms]
    radii = [_COVALENT_RADII.get(a.element, _DEFAULT_RADIUS) for a in atoms]

    for i in range(n):
        for j in range(i + 1, n):
            threshold = (radii[i] + radii[j]) * _BOND_TOLERANCE
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dz = zs[i] - zs[j]
            if dx*dx + dy*dy + dz*dz < threshold * threshold:
                bonds.append((i + 1, j + 1, 1))

    return bonds
