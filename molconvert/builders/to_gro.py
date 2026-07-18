"""
GROMACS GRO file builder.

Converts a MoleculeIC intermediate representation into the GRO text format.
Coordinates are written in nanometers (Angstroms / 10.0).
"""

from __future__ import annotations

from ..core.internal_coords import MoleculeIC


def molecule_to_gro(
    mol: MoleculeIC,
    box: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """
    Format a MoleculeIC as a GROMACS GRO string.

    GRO format (fixed-width columns):
        Line 1: title (molecule name)
        Line 2: number of atoms
        Atom lines: resSeq resName atomName atomSerial x y z  (coords in nm)
        Last line: box vectors

    IMPORTANT: GRO uses nanometers, not Angstroms. All coordinates are
    divided by 10.0 before formatting.

    Raises ValueError if any atom has no Cartesian position.
    """
    lines: list[str] = []

    # Line 1: title
    lines.append(mol.name)

    # Line 2: atom count (right-aligned in 5 chars)
    lines.append(f"{len(mol.atoms):>5d}")

    # Atom lines (fixed-width GRO format)
    for atom in mol.atoms:
        if atom.cart_x is None or atom.cart_y is None or atom.cart_z is None:
            raise ValueError(
                f"Atom {atom.atom_serial} ({atom.atom_name}) has no Cartesian "
                "position. Run reconstruction first."
            )

        residue_seq = atom.residue_seq % 100000
        atom_serial = atom.atom_serial % 100000

        # Convert Angstroms to nanometers
        x_nm = atom.cart_x / 10.0
        y_nm = atom.cart_y / 10.0
        z_nm = atom.cart_z / 10.0

        # GRO fixed-width format:
        # %5d%-5s%5s%5d%8.3f%8.3f%8.3f
        line = (
            f"{residue_seq:5d}"
            f"{atom.residue_name:<5s}"
            f"{atom.atom_name:>5s}"
            f"{atom_serial:5d}"
            f"{x_nm:8.3f}"
            f"{y_nm:8.3f}"
            f"{z_nm:8.3f}"
        )
        lines.append(line)

    # Box vector line
    lines.append(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}")

    return "\n".join(lines)


def save_gro(
    mol: MoleculeIC,
    path: str,
    box: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Write mol to path as a GRO file."""
    with open(path, "w") as fh:
        fh.write(molecule_to_gro(mol, box=box))
        fh.write("\n")
