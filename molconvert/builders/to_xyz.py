"""
XYZ file builder.

Converts a MoleculeIC intermediate representation into the XYZ text format.
Coordinates are written in Angstroms.
"""

from __future__ import annotations

from ..core.internal_coords import MoleculeIC


def molecule_to_xyz(mol: MoleculeIC) -> str:
    """
    Format a MoleculeIC as an XYZ string.

    The XYZ format is:
        Line 1: number of atoms
        Line 2: comment / title (molecule name)
        Lines 3+: element  x  y  z  (coordinates in Angstroms)

    Raises ValueError if any atom has no Cartesian position.
    """
    lines: list[str] = []

    # Line 1: atom count
    lines.append(str(len(mol.atoms)))

    # Line 2: title / comment
    lines.append(mol.name)

    # Coordinate lines
    for atom in mol.atoms:
        if atom.cart_x is None or atom.cart_y is None or atom.cart_z is None:
            raise ValueError(
                f"Atom {atom.atom_serial} ({atom.atom_name}) has no Cartesian "
                "position. Run reconstruction first."
            )
        lines.append(
            f"{atom.element:<2s}  {atom.cart_x:12.6f}  {atom.cart_y:12.6f}  {atom.cart_z:12.6f}"
        )

    return "\n".join(lines)


def save_xyz(mol: MoleculeIC, path: str) -> None:
    """Write mol to path as an XYZ file."""
    with open(path, "w") as fh:
        fh.write(molecule_to_xyz(mol))
        fh.write("\n")
