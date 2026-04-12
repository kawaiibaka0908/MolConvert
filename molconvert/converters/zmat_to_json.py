"""
Z-matrix (ZMAT) external format → MoleculeIC (JSON internal IR).

This is the reverse of json_to_zmat. It parses a .zmat file produced by
json_to_zmat (or any conforming ZMAT) and returns a MoleculeIC ready for
reconstruction or further conversion.

ZMAT file structure expected
-----------------------------
    ZMAT <molecule_name>
    # source_fmt <fmt>
    # n_atoms <N>
    # anchor <1-based-idx> <x> <y> <z>
    <idx>  <name>  <resname>  <chain>  <resseq>
    <idx>  <name>  <resname>  <chain>  <resseq>  <bond_to>  <bond_len>
    <idx>  <name>  <resname>  <chain>  <resseq>  <bond_to>  <bond_len>  <angle_to>  <angle>
    <idx>  <name>  <resname>  <chain>  <resseq>  <bond_to>  <bond_len>  <angle_to>  <angle>  <dihedral_to>  <dihedral>
    ...
    END

Public API
----------
    zmat_to_molecule(text)   -> MoleculeIC   (parse a ZMAT string)
    load_zmat(path)          -> MoleculeIC   (parse a .zmat file)
"""

from __future__ import annotations
from ..core.internal_coords import AtomIC, MoleculeIC


def zmat_to_molecule(text: str) -> MoleculeIC:
    """
    Parse a ZMAT-format string and return a MoleculeIC.

    Anchor Cartesian positions are restored from `# anchor` header lines.
    Non-anchor atoms will have cart_x/y/z = None until reconstruct() is called.
    """
    name = "unknown"
    source_fmt = "zmat"
    metadata: dict = {}
    anchors: dict[int, tuple[float, float, float]] = {}
    atom_rows: list[list[str]] = []

    for raw in text.splitlines():
        line = raw.strip()

        if not line or line == "END":
            continue

        if line.startswith("ZMAT "):
            name = line[5:].strip()

        elif line.startswith("# source_fmt "):
            source_fmt = line[13:].strip()

        elif line.startswith("# meta "):
            parts = line[7:].split(None, 1)
            if len(parts) == 2:
                metadata[parts[0]] = parts[1]

        elif line.startswith("# anchor "):
            parts = line[9:].split()
            if len(parts) >= 4:
                idx = int(parts[0])
                anchors[idx] = (float(parts[1]), float(parts[2]), float(parts[3]))

        elif line.startswith("#"):
            continue  # skip other comment lines

        else:
            atom_rows.append(line.split())

    atoms: list[AtomIC] = []

    for parts in atom_rows:
        idx          = int(parts[0])    # 1-based ZMAT position
        atom_name    = parts[1]
        residue_name = parts[2]
        chain_id     = parts[3]
        residue_seq  = int(parts[4])

        bond_to = bond_length = angle_to = bond_angle = dihedral_to = dihedral = None

        if len(parts) >= 7:
            bond_to      = int(parts[5])
            bond_length  = float(parts[6])
        if len(parts) >= 9:
            angle_to     = int(parts[7])
            bond_angle   = float(parts[8])
        if len(parts) >= 11:
            dihedral_to  = int(parts[9])
            dihedral     = float(parts[10])

        # Restore anchor Cartesian positions; non-anchors stay None until reconstruct().
        cart_x = cart_y = cart_z = None
        if idx in anchors:
            cart_x, cart_y, cart_z = anchors[idx]

        atoms.append(AtomIC(
            atom_serial  = idx,
            atom_name    = atom_name,
            residue_name = residue_name,
            chain_id     = chain_id,
            residue_seq  = residue_seq,
            element      = _element_from_name(atom_name),
            bond_length  = bond_length,
            bond_angle   = bond_angle,
            dihedral     = dihedral,
            bond_to      = bond_to,
            angle_to     = angle_to,
            dihedral_to  = dihedral_to,
            cart_x       = cart_x,
            cart_y       = cart_y,
            cart_z       = cart_z,
        ))

    return MoleculeIC(name=name, source_fmt=source_fmt, atoms=atoms, metadata=metadata)


def load_zmat(path: str) -> MoleculeIC:
    """Read a .zmat file and return a MoleculeIC."""
    with open(path) as fh:
        return zmat_to_molecule(fh.read())


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _element_from_name(atom_name: str) -> str:
    """
    Derive an element symbol from an atom name.

    Examples: 'N' → 'N', 'CA' → 'C', 'OD1' → 'O', 'HG11' → 'H'
    Strips leading digits first to handle names like '1HB'.
    """
    stripped = atom_name.lstrip("0123456789")
    return (stripped[0] if stripped else atom_name[0]).upper()
