"""
MoleculeIC -> Gaussian input file builder (.gjf/.com).

Public API
----------
    molecule_to_gaussian(mol, ...)  -> str
    save_gaussian(mol, path, ...)   -> None
"""

from __future__ import annotations
from typing import Optional
from ..core.internal_coords import MoleculeIC


def molecule_to_gaussian(
    mol: MoleculeIC,
    route: str = "# HF/6-31G(d) opt",
    charge: int = 0,
    multiplicity: int = 1,
    title: str | None = None,
) -> str:
    """
    Format a MoleculeIC as a Gaussian input file string (.gjf/.com).

    Raises ValueError if any atom has no Cartesian position.
    """
    _require_positions(mol)

    title_str = title or mol.name or "Gaussian Input"

    lines: list[str] = []

    # Route section
    lines.append(route)
    lines.append("")  # blank line after route

    # Title section
    lines.append(title_str)
    lines.append("")  # blank line after title

    # Charge and multiplicity
    lines.append(f"{charge} {multiplicity}")

    # Atom coordinate lines
    for atom in mol.atoms:
        lines.append(
            f"{atom.element:<2s}    {atom.cart_x:14.8f}  {atom.cart_y:14.8f}  {atom.cart_z:14.8f}"
        )

    # Trailing blank line (Gaussian spec requires the file to end with a blank line)
    lines.append("")
    lines.append("")

    return "\n".join(lines)


def save_gaussian(mol: MoleculeIC, path: str, **kwargs) -> None:
    """Write mol to path as a Gaussian input file."""
    with open(path, "w") as fh:
        fh.write(molecule_to_gaussian(mol, **kwargs))


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _require_positions(mol: MoleculeIC) -> None:
    for atom in mol.atoms:
        if atom.cart_x is None or atom.cart_y is None or atom.cart_z is None:
            raise ValueError(
                f"Atom {atom.atom_serial} ({atom.atom_name}) has no "
                "Cartesian position. Run reconstruction first."
            )
