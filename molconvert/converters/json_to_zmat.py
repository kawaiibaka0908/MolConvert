"""
MoleculeIC (JSON internal IR) → Z-matrix (ZMAT) external format.

ZMAT is the *external* representation for internal coordinates.
JSON / MoleculeIC is always the *internal* IR — this module is a view layer only.

ZMAT file structure
-------------------
    ZMAT <molecule_name>
    # source_fmt <fmt>
    # n_atoms <N>
    # anchor <1-based-idx> <x> <y> <z>      ← one line per anchor atom (first 3)
    <idx>  <name>  <resname>  <chain>  <resseq>
    <idx>  <name>  <resname>  <chain>  <resseq>  <bond_to>  <bond_len>
    <idx>  <name>  <resname>  <chain>  <resseq>  <bond_to>  <bond_len>  <angle_to>  <angle>
    <idx>  <name>  <resname>  <chain>  <resseq>  <bond_to>  <bond_len>  <angle_to>  <angle>  <dihedral_to>  <dihedral>
    ...
    END

Indices in the data lines are 1-based positions in the atom list (not PDB serials).
Anchor Cartesian coordinates are stored in `# anchor` header lines so the file is
self-contained for a full round-trip back to PDB via zmat_to_json.

Public API
----------
    molecule_to_zmat(mol)        -> str
    save_zmat(mol, path)         -> None
"""

from __future__ import annotations
from ..core.internal_coords import MoleculeIC


def molecule_to_zmat(mol: MoleculeIC) -> str:
    """
    Convert a MoleculeIC to a ZMAT-format string.

    Reference indices (bond_to / angle_to / dihedral_to) stored on each
    AtomIC are used when present; otherwise sequential indices are used as
    the fallback (atom i bonds to i-1, angle to i-2, dihedral to i-3).
    """
    lines: list[str] = []

    # ---- Header ----
    lines.append(f"ZMAT {mol.name}")
    lines.append(f"# source_fmt {mol.source_fmt}")
    for key, val in mol.metadata.items():
        lines.append(f"# meta {key} {val}")
    lines.append(f"# n_atoms {len(mol.atoms)}")

    # ---- Anchor Cartesian positions (first 3 atoms) ----
    # Stored as comments so the file is self-contained for reconstruction.
    for i, atom in enumerate(mol.atoms[:3]):
        if atom.cart_x is not None:
            lines.append(
                f"# anchor {i + 1}"
                f"  {atom.cart_x:>12.6f}"
                f"  {atom.cart_y:>12.6f}"
                f"  {atom.cart_z:>12.6f}"
            )

    # ---- Atom data lines ----
    for i, atom in enumerate(mol.atoms):
        idx = i + 1  # 1-based ZMAT position

        row = (
            f"{idx:>4}  "
            f"{atom.atom_name:<4}  "
            f"{atom.residue_name:<3}  "
            f"{atom.chain_id}  "
            f"{atom.residue_seq:>4}"
        )

        if atom.bond_length is not None:
            # 1-based reference index: use stored value or fall back to i (previous atom)
            bond_ref = atom.bond_to if atom.bond_to is not None else i
            row += f"  {bond_ref:>4}  {atom.bond_length:>10.6f}"

            if atom.bond_angle is not None:
                angle_ref = atom.angle_to if atom.angle_to is not None else i - 1
                row += f"  {angle_ref:>4}  {atom.bond_angle:>9.3f}"

                if atom.dihedral is not None:
                    dihedral_ref = atom.dihedral_to if atom.dihedral_to is not None else i - 2
                    row += f"  {dihedral_ref:>4}  {atom.dihedral:>10.3f}"

        lines.append(row)

    lines.append("END")
    return "\n".join(lines)


