"""
MoleculeIC -> MOL2 (Tripos) builder.

Bond connectivity and bond orders are obtained from RDKit after
converting the MoleculeIC to an RWMol via ``molecule_to_rdmol``.
Atom types are mapped to Tripos SYBYL types via ``tripos_atom_type``.

Public API
----------
    molecule_to_mol2(mol)             -> str          (one MOL2 block)
    save_mol2(mol, path)              -> None
    molecules_to_mol2(mols)           -> str          (multi-molecule MOL2)
    save_mol2_multi(mols, path)       -> None
"""

from __future__ import annotations

from rdkit import Chem

from ..core.internal_coords import MoleculeIC
from ..core.rdkit_bridge import molecule_to_rdmol, tripos_atom_type


# ------------------------------------------------------------------ #
#  Bond-type mapping: RDKit BondType -> MOL2 bond type string          #
# ------------------------------------------------------------------ #

_BOND_TYPE_MAP = {
    Chem.BondType.SINGLE: "1",
    Chem.BondType.DOUBLE: "2",
    Chem.BondType.TRIPLE: "3",
    Chem.BondType.AROMATIC: "ar",
}


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def molecule_to_mol2(mol: MoleculeIC) -> str:
    """
    Convert a fully-positioned MoleculeIC to a single MOL2 block.

    Parameters
    ----------
    mol : MoleculeIC
        Must have ``cart_x``, ``cart_y``, ``cart_z`` set on every atom.

    Returns
    -------
    str
        A complete MOL2 block including MOLECULE, ATOM, and BOND sections.

    Raises
    ------
    ValueError
        If any atom in *mol* lacks Cartesian coordinates.
    """
    # 1. Require all atoms to have Cartesian positions
    _require_positions(mol)

    # 2. Convert to RDKit molecule (perceives bonds + bond orders)
    rdmol = molecule_to_rdmol(mol)

    # 3. Extract bonds with orders
    bonds: list[tuple[int, int, str]] = []
    for bond in rdmol.GetBonds():
        bt = bond.GetBondType()
        bt_str = _BOND_TYPE_MAP.get(bt, "1")
        # RDKit uses 0-based indices; MOL2 uses 1-based
        begin = bond.GetBeginAtomIdx() + 1
        end = bond.GetEndAtomIdx() + 1
        bonds.append((begin, end, bt_str))

    # 4. Map each atom to a Tripos atom type
    atom_types: list[str] = []
    for rd_atom in rdmol.GetAtoms():
        atom_types.append(tripos_atom_type(rd_atom))

    # 5. Format MOL2 sections
    mol_name = mol.metadata.get("mol_title", mol.name)
    n_atoms = len(mol.atoms)
    n_bonds = len(bonds)

    lines: list[str] = []

    # --- @<TRIPOS>MOLECULE ---
    lines.append("@<TRIPOS>MOLECULE")
    lines.append(mol_name)
    lines.append(f" {n_atoms} {n_bonds} 0 0 0")
    lines.append("SMALL")
    lines.append("NO_CHARGES")
    lines.append("")

    # --- @<TRIPOS>ATOM ---
    lines.append("@<TRIPOS>ATOM")
    for i, atom in enumerate(mol.atoms):
        atom_id = i + 1
        atom_name = atom.atom_name
        x = atom.cart_x
        y = atom.cart_y
        z = atom.cart_z
        tripos_type = atom_types[i]
        res_id = atom.residue_seq
        res_name = atom.residue_name

        line = (
            f"{atom_id:>7d} {atom_name:<8s} "
            f"{x:10.4f} {y:10.4f} {z:10.4f} "
            f"{tripos_type:<8s} {res_id:>3d} {res_name:<6s} {0.0:8.4f}"
        )
        lines.append(line)

    # --- @<TRIPOS>BOND ---
    lines.append("@<TRIPOS>BOND")
    for bond_id, (begin_atom, end_atom, bond_type_str) in enumerate(bonds, start=1):
        line = f"{bond_id:>6d} {begin_atom:>5d} {end_atom:>5d} {bond_type_str}"
        lines.append(line)

    return "\n".join(lines)


def save_mol2(mol: MoleculeIC, path: str) -> None:
    """Write one MoleculeIC to a .mol2 file (single molecule)."""
    with open(path, "w") as fh:
        fh.write(molecule_to_mol2(mol))
        fh.write("\n")


def molecules_to_mol2(mols: list[MoleculeIC]) -> str:
    """Convert a list of MoleculeIC objects to a multi-molecule MOL2 string."""
    return "\n".join(molecule_to_mol2(m) for m in mols)


def save_mol2_multi(mols: list[MoleculeIC], path: str) -> None:
    """Write multiple MoleculeIC objects to a single multi-molecule MOL2 file."""
    with open(path, "w") as fh:
        fh.write(molecules_to_mol2(mols))
        fh.write("\n")


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _require_positions(mol: MoleculeIC) -> None:
    """Raise ValueError if any atom lacks Cartesian coordinates."""
    for atom in mol.atoms:
        if atom.cart_x is None or atom.cart_y is None or atom.cart_z is None:
            raise ValueError(
                f"Atom {atom.atom_serial} ({atom.atom_name}) has no "
                "Cartesian position -- run reconstruct() first."
            )
