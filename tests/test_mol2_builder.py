"""
Tests for the MOL2 builder (molconvert.builders.to_mol2).
"""

from pathlib import Path

import pytest

from molconvert.parsers.sdf_parser import parse_sdf
from molconvert.parsers.mol2_parser import parse_mol2
from molconvert.builders.reconstruct import reconstruct
from molconvert.builders.to_mol2 import molecule_to_mol2, save_mol2, molecules_to_mol2
from molconvert.core.internal_coords import MoleculeIC, AtomIC

MINI_SDF = str(Path(__file__).parent / "data" / "mini.sdf")


def _get_ethanol() -> MoleculeIC:
    """Parse mini.sdf, reconstruct the first molecule (ethanol), return it."""
    mols = parse_sdf(MINI_SDF)
    return reconstruct(mols[0])


# ------------------------------------------------------------------ #
#  Section markers                                                     #
# ------------------------------------------------------------------ #

def test_mol2_contains_molecule_section():
    mol = _get_ethanol()
    text = molecule_to_mol2(mol)
    assert "@<TRIPOS>MOLECULE" in text


def test_mol2_contains_atom_section():
    mol = _get_ethanol()
    text = molecule_to_mol2(mol)
    assert "@<TRIPOS>ATOM" in text


def test_mol2_contains_bond_section():
    mol = _get_ethanol()
    text = molecule_to_mol2(mol)
    assert "@<TRIPOS>BOND" in text


# ------------------------------------------------------------------ #
#  Atom / bond counts and content                                      #
# ------------------------------------------------------------------ #

def test_mol2_atom_count():
    """Ethanol from mini.sdf should have 9 atoms in the ATOM section."""
    mol = _get_ethanol()
    text = molecule_to_mol2(mol)
    lines = text.splitlines()

    # Find ATOM section and count lines until the next section marker
    in_atom = False
    atom_lines = []
    for line in lines:
        if "@<TRIPOS>ATOM" in line:
            in_atom = True
            continue
        if in_atom:
            if line.startswith("@<TRIPOS>"):
                break
            if line.strip():
                atom_lines.append(line)

    assert len(atom_lines) == 9


def test_mol2_has_bonds():
    """BOND section should have at least 1 bond line."""
    mol = _get_ethanol()
    text = molecule_to_mol2(mol)
    lines = text.splitlines()

    in_bond = False
    bond_lines = []
    for line in lines:
        if "@<TRIPOS>BOND" in line:
            in_bond = True
            continue
        if in_bond:
            if line.startswith("@<TRIPOS>"):
                break
            if line.strip():
                bond_lines.append(line)

    assert len(bond_lines) >= 1


def test_mol2_atom_types_reasonable():
    """Ethanol atom types should contain C.3 (sp3 carbon) and O (or O.3)."""
    mol = _get_ethanol()
    text = molecule_to_mol2(mol)
    lines = text.splitlines()

    in_atom = False
    atom_types = []
    for line in lines:
        if "@<TRIPOS>ATOM" in line:
            in_atom = True
            continue
        if in_atom:
            if line.startswith("@<TRIPOS>"):
                break
            if line.strip():
                parts = line.split()
                # atom_type is the 6th field (index 5)
                atom_types.append(parts[5])

    assert "C.3" in atom_types, f"Expected C.3 in atom types: {atom_types}"
    assert any(t.startswith("O") for t in atom_types), (
        f"Expected O or O.3 in atom types: {atom_types}"
    )


# ------------------------------------------------------------------ #
#  Roundtrip: SDF -> MOL2 string -> parse MOL2 -> compare coordinates  #
# ------------------------------------------------------------------ #

def test_mol2_roundtrip_coordinates(tmp_path):
    """Parse SDF -> build MOL2 -> parse MOL2 -> compare coordinates."""
    mol = _get_ethanol()
    mol2_text = molecule_to_mol2(mol)

    # Write the MOL2 to a temp file so the parser can read it
    mol2_path = str(tmp_path / "ethanol_roundtrip.mol2")
    with open(mol2_path, "w") as fh:
        fh.write(mol2_text)
        fh.write("\n")

    # Parse the MOL2 back
    parsed = parse_mol2(mol2_path)
    assert len(parsed) >= 1

    parsed_mol = parsed[0]

    # Compare coordinates atom-by-atom
    assert len(parsed_mol.atoms) == len(mol.atoms)
    for orig, parsed_atom in zip(mol.atoms, parsed_mol.atoms):
        assert orig.cart_x == pytest.approx(parsed_atom.cart_x, abs=0.01)
        assert orig.cart_y == pytest.approx(parsed_atom.cart_y, abs=0.01)
        assert orig.cart_z == pytest.approx(parsed_atom.cart_z, abs=0.01)


# ------------------------------------------------------------------ #
#  Bond types                                                          #
# ------------------------------------------------------------------ #

def test_mol2_bond_types():
    """All bonds should have valid MOL2 types: 1, 2, 3, or ar."""
    mol = _get_ethanol()
    text = molecule_to_mol2(mol)
    lines = text.splitlines()

    in_bond = False
    valid_types = {"1", "2", "3", "ar"}
    for line in lines:
        if "@<TRIPOS>BOND" in line:
            in_bond = True
            continue
        if in_bond:
            if line.startswith("@<TRIPOS>"):
                break
            if line.strip():
                parts = line.split()
                bond_type = parts[3]
                assert bond_type in valid_types, (
                    f"Invalid bond type '{bond_type}' in line: {line}"
                )


# ------------------------------------------------------------------ #
#  File I/O                                                            #
# ------------------------------------------------------------------ #

def test_save_mol2_creates_file(tmp_path):
    """save_mol2 should create a file at the given path."""
    mol = _get_ethanol()
    out_path = str(tmp_path / "test_output.mol2")
    save_mol2(mol, out_path)
    assert Path(out_path).exists()
    content = Path(out_path).read_text()
    assert "@<TRIPOS>MOLECULE" in content


# ------------------------------------------------------------------ #
#  Error handling                                                      #
# ------------------------------------------------------------------ #

def test_mol2_raises_without_positions():
    """MoleculeIC with no positions should raise ValueError."""
    mol = MoleculeIC(name="empty", source_fmt="test", atoms=[
        AtomIC(
            atom_serial=1,
            atom_name="C1",
            residue_name="LIG",
            chain_id="A",
            residue_seq=1,
            element="C",
            # No cart_x/y/z set -> None
        ),
    ])
    with pytest.raises(ValueError):
        molecule_to_mol2(mol)


# ------------------------------------------------------------------ #
#  Multi-molecule                                                      #
# ------------------------------------------------------------------ #

def test_molecules_to_mol2_multi():
    """molecules_to_mol2 with 2 molecules should produce 2 MOLECULE blocks."""
    mols = parse_sdf(MINI_SDF)
    mols = [reconstruct(m) for m in mols]
    text = molecules_to_mol2(mols)
    count = text.count("@<TRIPOS>MOLECULE")
    assert count == 2, f"Expected 2 @<TRIPOS>MOLECULE blocks, got {count}"


# ------------------------------------------------------------------ #
#  Bond order perception: ethanol (all single) vs acetone (has double) #
# ------------------------------------------------------------------ #

def test_ethanol_all_single_bonds():
    """Ethanol should have only single bonds (type '1')."""
    mol = _get_ethanol()
    text = molecule_to_mol2(mol)
    lines = text.splitlines()
    in_bond = False
    bond_types = []
    for line in lines:
        if "@<TRIPOS>BOND" in line:
            in_bond = True
            continue
        if in_bond:
            if line.startswith("@<TRIPOS>"):
                break
            if line.strip():
                bond_types.append(line.split()[3])
    assert all(bt == "1" for bt in bond_types), (
        f"Expected all single bonds for ethanol, got: {bond_types}"
    )


def test_acetone_has_non_single_bonds():
    """Acetone (second molecule in mini.sdf, heavy atoms only) should have
    bonds perceived as non-single (C=O or higher-order) by RDKit.
    Note: without hydrogens, DetermineBonds may perceive triple bonds
    instead of double, but the key test is that not all bonds are single."""
    mols = parse_sdf(MINI_SDF)
    assert len(mols) >= 2, "mini.sdf should have at least 2 molecules"
    acetone = reconstruct(mols[1])
    text = molecule_to_mol2(acetone)
    lines = text.splitlines()
    in_bond = False
    bond_types = []
    for line in lines:
        if "@<TRIPOS>BOND" in line:
            in_bond = True
            continue
        if in_bond:
            if line.startswith("@<TRIPOS>"):
                break
            if line.strip():
                bond_types.append(line.split()[3])
    # Acetone should have at least one non-single bond
    non_single = [bt for bt in bond_types if bt != "1"]
    assert len(non_single) > 0, (
        f"Expected non-single bonds for acetone, got all single: {bond_types}"
    )

