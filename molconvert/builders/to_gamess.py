"""
MoleculeIC -> GAMESS input file builder (.inp).

Public API
----------
    molecule_to_gamess(mol, ...)  -> str
    save_gamess(mol, path, ...)   -> None
"""

from __future__ import annotations
from typing import Optional
from rdkit import Chem
from ..core.internal_coords import MoleculeIC


# Cache the periodic table instance
_PT = Chem.GetPeriodicTable()


def molecule_to_gamess(
    mol: MoleculeIC,
    contrl: str = "SCFTYP=RHF RUNTYP=ENERGY",
    basis: str = "GBASIS=STO NGAUSS=3",
    charge: int = 0,
    multiplicity: int = 1,
    title: str | None = None,
) -> str:
    """
    Format a MoleculeIC as a GAMESS input file string (.inp).

    Needs atomic numbers -- use RDKit: Chem.GetPeriodicTable().GetAtomicNumber(element)

    Raises ValueError if any atom has no Cartesian position.
    """
    _require_positions(mol)

    title_str = title or mol.name or "GAMESS Job"

    lines: list[str] = []

    # $CONTRL group -- $ must be in column 2 (preceded by a single space)
    lines.append(
        f" $CONTRL {contrl} ICHARG={charge} MULT={multiplicity} UNITS=ANGS $END"
    )

    # $BASIS group
    lines.append(f" $BASIS {basis} $END")

    # $DATA group
    lines.append(" $DATA")
    lines.append(title_str)
    lines.append("C1")
    lines.append("")  # blank line between C1 and first atom

    # Atom coordinate lines
    for atom in mol.atoms:
        atomic_num = float(_PT.GetAtomicNumber(atom.element))
        lines.append(
            f"{atom.element:<2s}  {atomic_num:.1f}  {atom.cart_x:14.8f}  {atom.cart_y:14.8f}  {atom.cart_z:14.8f}"
        )

    # Close $DATA group
    lines.append(" $END")

    return "\n".join(lines)


def save_gamess(mol: MoleculeIC, path: str, **kwargs) -> None:
    """Write mol to path as a GAMESS input file."""
    with open(path, "w") as fh:
        fh.write(molecule_to_gamess(mol, **kwargs))


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
