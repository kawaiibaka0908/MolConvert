"""Tests for parsers/mol2_parser.py."""

from pathlib import Path

import numpy as np
import pytest

from molconvert.parsers.mol2_parser import parse_mol2

DATA_DIR = Path(__file__).parent / "data"
ETHANOL_MOL2 = str(DATA_DIR / "ethanol.mol2")

# Known ethanol coordinates (same as mini.sdf ethanol)
EXPECTED_COORDS = [
    (0.0000, 0.0000, 0.0000),     # C
    (1.5400, 0.0000, 0.0000),     # C
    (2.0600, 1.0300, 0.0000),     # O
    (-0.3900, 1.0300, 0.0000),    # H
    (-0.3900, -0.5150, -0.8910),  # H
    (-0.3900, -0.5150, 0.8910),   # H
    (1.9300, -0.5150, 0.8910),    # H
    (1.9300, -0.5150, -0.8910),   # H
    (1.9600, 1.9200, 0.0000),     # H
]


# ------------------------------------------------------------------ #
#  Basic parsing                                                       #
# ------------------------------------------------------------------ #

def test_parse_returns_list():
    result = parse_mol2(ETHANOL_MOL2)
    assert isinstance(result, list)


def test_parse_atom_count():
    mol = parse_mol2(ETHANOL_MOL2)[0]
    assert len(mol) == 9


def test_parse_source_fmt():
    mol = parse_mol2(ETHANOL_MOL2)[0]
    assert mol.source_fmt == "mol2"


def test_parse_metadata_has_title():
    mol = parse_mol2(ETHANOL_MOL2)[0]
    assert "mol_title" in mol.metadata
    assert mol.metadata["mol_title"] == "ethanol"


# ------------------------------------------------------------------ #
#  Coordinate checks                                                   #
# ------------------------------------------------------------------ #

def test_parse_coordinates():
    mol = parse_mol2(ETHANOL_MOL2)[0]
    for i, (ex, ey, ez) in enumerate(EXPECTED_COORDS):
        atom = mol.atoms[i]
        assert atom.cart_x == pytest.approx(ex, abs=0.001)
        assert atom.cart_y == pytest.approx(ey, abs=0.001)
        assert atom.cart_z == pytest.approx(ez, abs=0.001)


def test_cartesian_stored():
    mol = parse_mol2(ETHANOL_MOL2)[0]
    for atom in mol.atoms:
        assert atom.cart_x is not None
        assert atom.cart_y is not None
        assert atom.cart_z is not None


# ------------------------------------------------------------------ #
#  Element and atom identity                                           #
# ------------------------------------------------------------------ #

def test_parse_elements():
    mol = parse_mol2(ETHANOL_MOL2)[0]
    elements = [a.element for a in mol.atoms]
    assert elements == ["C", "C", "O", "H", "H", "H", "H", "H", "H"]


def test_atom_names_from_mol2():
    mol = parse_mol2(ETHANOL_MOL2)[0]
    names = [a.atom_name for a in mol.atoms]
    assert names == ["C1", "C2", "O1", "H1", "H2", "H3", "H4", "H5", "H6"]


def test_residue_name():
    mol = parse_mol2(ETHANOL_MOL2)[0]
    for atom in mol.atoms:
        assert atom.residue_name == "LIG1"


# ------------------------------------------------------------------ #
#  Internal coordinates                                                #
# ------------------------------------------------------------------ #

def test_first_atom_is_anchor():
    mol = parse_mol2(ETHANOL_MOL2)[0]
    a = mol.atoms[0]
    assert a.bond_length is None
    assert a.bond_angle is None
    assert a.dihedral is None


def test_ic_computed():
    mol = parse_mol2(ETHANOL_MOL2)[0]
    for atom in mol.atoms[1:]:
        assert atom.bond_length is not None


def test_second_atom_has_bond_length_only():
    mol = parse_mol2(ETHANOL_MOL2)[0]
    a = mol.atoms[1]
    assert a.bond_length is not None
    assert a.bond_angle is None
    assert a.dihedral is None


def test_third_atom_has_bond_length_and_angle():
    mol = parse_mol2(ETHANOL_MOL2)[0]
    a = mol.atoms[2]
    assert a.bond_length is not None
    assert a.bond_angle is not None
    assert a.dihedral is None


def test_fourth_atom_has_all_ic():
    mol = parse_mol2(ETHANOL_MOL2)[0]
    a = mol.atoms[3]
    assert a.bond_length is not None
    assert a.bond_angle is not None
    assert a.dihedral is not None


# ------------------------------------------------------------------ #
#  Error handling                                                      #
# ------------------------------------------------------------------ #

def test_invalid_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_mol2("nonexistent.mol2")
