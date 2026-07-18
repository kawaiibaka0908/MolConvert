"""
Tests for XYZ and GRO builders (Phase 3).
"""

from pathlib import Path

import numpy as np
import pytest

from molconvert.parsers.pdb_parser import parse_pdb
from molconvert.builders.reconstruct import reconstruct
from molconvert.builders.to_xyz import molecule_to_xyz, save_xyz
from molconvert.builders.to_gro import molecule_to_gro, save_gro
from molconvert.core.internal_coords import MoleculeIC, AtomIC

MINI_PDB = str(Path(__file__).parent / "data" / "mini.pdb")


@pytest.fixture
def mini_mol():
    """Parse and reconstruct mini.pdb for builder tests."""
    mols = parse_pdb(MINI_PDB)
    return reconstruct(mols[0])


# Expected coordinates from mini.pdb (Angstroms)
EXPECTED_COORDS = np.array([
    [0.000, 0.000, 0.000],
    [1.458, 0.000, 0.000],
    [2.009, 1.420, 0.000],
    [1.251, 2.395, 0.000],
    [3.349, 1.636, 0.000],
])

EXPECTED_ELEMENTS = ["N", "C", "C", "O", "N"]


# ------------------------------------------------------------------ #
#  XYZ builder tests                                                   #
# ------------------------------------------------------------------ #


def test_xyz_first_line_is_atom_count(mini_mol):
    """First line of XYZ output is the atom count."""
    xyz_str = molecule_to_xyz(mini_mol)
    first_line = xyz_str.split("\n")[0]
    assert first_line.strip() == "5"


def test_xyz_second_line_is_name(mini_mol):
    """Second line of XYZ output contains the molecule name."""
    xyz_str = molecule_to_xyz(mini_mol)
    second_line = xyz_str.split("\n")[1]
    assert mini_mol.name in second_line


def test_xyz_correct_element_symbols(mini_mol):
    """Element symbols N, C, C, O, N appear in the coordinate lines."""
    xyz_str = molecule_to_xyz(mini_mol)
    lines = xyz_str.split("\n")
    coord_lines = lines[2:]  # Skip header and name
    elements = [line.split()[0] for line in coord_lines if line.strip()]
    assert elements == EXPECTED_ELEMENTS


def test_xyz_coordinate_values(mini_mol):
    """Parsed XYZ coordinates match expected values within tolerance."""
    xyz_str = molecule_to_xyz(mini_mol)
    lines = xyz_str.split("\n")
    coord_lines = lines[2:]  # Skip header and name

    for i, line in enumerate(coord_lines):
        if not line.strip():
            continue
        parts = line.split()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        np.testing.assert_allclose(
            [x, y, z], EXPECTED_COORDS[i], atol=0.01,
            err_msg=f"Coordinate mismatch at atom {i}"
        )


def test_xyz_raises_without_positions():
    """ValueError is raised when atoms have no Cartesian positions."""
    mol = MoleculeIC(name="test", source_fmt="test", atoms=[
        AtomIC(
            atom_serial=1, atom_name="N", residue_name="ALA",
            chain_id="A", residue_seq=1, element="N",
            # No cart_x/y/z set
        ),
    ])
    with pytest.raises(ValueError, match="no Cartesian position"):
        molecule_to_xyz(mol)


def test_save_xyz_creates_file(mini_mol, tmp_path):
    """save_xyz creates a file on disk."""
    out_path = str(tmp_path / "test.xyz")
    save_xyz(mini_mol, out_path)
    assert Path(out_path).exists()
    content = Path(out_path).read_text()
    assert content.startswith("5")


def test_xyz_line_count(mini_mol):
    """XYZ output has n_atoms + 2 lines (header + name + coordinates)."""
    xyz_str = molecule_to_xyz(mini_mol)
    lines = xyz_str.split("\n")
    # molecule_to_xyz returns joined lines without trailing newline
    # So line count should be exactly n_atoms + 2
    assert len(lines) == len(mini_mol.atoms) + 2


# ------------------------------------------------------------------ #
#  GRO builder tests                                                   #
# ------------------------------------------------------------------ #


def test_gro_title_line(mini_mol):
    """First line of GRO output contains the molecule name."""
    gro_str = molecule_to_gro(mini_mol)
    first_line = gro_str.split("\n")[0]
    assert mini_mol.name in first_line


def test_gro_atom_count_line(mini_mol):
    """Second line of GRO output is the atom count."""
    gro_str = molecule_to_gro(mini_mol)
    second_line = gro_str.split("\n")[1]
    assert second_line.strip() == "5"


def test_gro_coordinates_in_nanometers(mini_mol):
    """GRO coordinates are Angstroms / 10 (nanometers) within tolerance."""
    gro_str = molecule_to_gro(mini_mol)
    lines = gro_str.split("\n")
    # Atom lines are lines[2] through lines[2 + n_atoms - 1]
    atom_lines = lines[2:2 + len(mini_mol.atoms)]

    for i, line in enumerate(atom_lines):
        # GRO fixed-width: positions are at columns 20-44 (3 x 8-char fields)
        x_nm = float(line[20:28])
        y_nm = float(line[28:36])
        z_nm = float(line[36:44])

        expected_nm = EXPECTED_COORDS[i] / 10.0
        np.testing.assert_allclose(
            [x_nm, y_nm, z_nm], expected_nm, atol=0.001,
            err_msg=f"GRO coordinate mismatch at atom {i}"
        )


def test_gro_box_vector_line(mini_mol):
    """Last line has default box vector values (0.0, 0.0, 0.0)."""
    gro_str = molecule_to_gro(mini_mol)
    last_line = gro_str.split("\n")[-1]
    values = [float(v) for v in last_line.split()]
    np.testing.assert_allclose(values, [0.0, 0.0, 0.0], atol=1e-5)


def test_gro_custom_box(mini_mol):
    """Custom box vector (1.0, 2.0, 3.0) appears in the last line."""
    gro_str = molecule_to_gro(mini_mol, box=(1.0, 2.0, 3.0))
    last_line = gro_str.split("\n")[-1]
    values = [float(v) for v in last_line.split()]
    np.testing.assert_allclose(values, [1.0, 2.0, 3.0], atol=1e-5)


def test_gro_raises_without_positions():
    """ValueError is raised when atoms have no Cartesian positions."""
    mol = MoleculeIC(name="test", source_fmt="test", atoms=[
        AtomIC(
            atom_serial=1, atom_name="N", residue_name="ALA",
            chain_id="A", residue_seq=1, element="N",
            # No cart_x/y/z set
        ),
    ])
    with pytest.raises(ValueError, match="no Cartesian position"):
        molecule_to_gro(mol)


def test_save_gro_creates_file(mini_mol, tmp_path):
    """save_gro creates a file on disk."""
    out_path = str(tmp_path / "test.gro")
    save_gro(mini_mol, out_path)
    assert Path(out_path).exists()
    content = Path(out_path).read_text()
    assert mini_mol.name in content


def test_gro_line_count(mini_mol):
    """GRO output has n_atoms + 3 lines (title + count + coords + box)."""
    gro_str = molecule_to_gro(mini_mol)
    lines = gro_str.split("\n")
    # title + atom_count + n_atoms coord lines + box line
    assert len(lines) == len(mini_mol.atoms) + 3
