"""Tests for parsers/gaussian_parser.py."""

from pathlib import Path

import numpy as np
import pytest

from molconvert.parsers.gaussian_parser import parse_gaussian

DATA_DIR = Path(__file__).parent / "data"
WATER_GAUSSIAN = str(DATA_DIR / "water_gaussian.log")


# ------------------------------------------------------------------ #
#  Basic parsing                                                       #
# ------------------------------------------------------------------ #

def test_parse_returns_list():
    result = parse_gaussian(WATER_GAUSSIAN)
    assert isinstance(result, list)


def test_parse_atom_count():
    mols = parse_gaussian(WATER_GAUSSIAN)
    assert len(mols[0]) == 3


def test_parse_elements():
    mol = parse_gaussian(WATER_GAUSSIAN)[0]
    elements = [a.element for a in mol.atoms]
    assert elements == ["O", "H", "H"]


def test_parse_source_fmt():
    for mol in parse_gaussian(WATER_GAUSSIAN, step="all"):
        assert mol.source_fmt == "gaussian"


# ------------------------------------------------------------------ #
#  Coordinate checks                                                   #
# ------------------------------------------------------------------ #

def test_parse_coordinates():
    """Last block should have the shifted x=0.01 coordinates."""
    mol = parse_gaussian(WATER_GAUSSIAN, step="last")[0]
    a0 = mol.atoms[0]
    assert a0.cart_x == pytest.approx(0.010000, abs=0.001)
    assert a0.cart_y == pytest.approx(0.000000, abs=0.001)
    assert a0.cart_z == pytest.approx(0.117790, abs=0.001)


# ------------------------------------------------------------------ #
#  Step parameter                                                      #
# ------------------------------------------------------------------ #

def test_parse_step_last():
    mols = parse_gaussian(WATER_GAUSSIAN, step="last")
    assert len(mols) == 1


def test_parse_step_all():
    mols = parse_gaussian(WATER_GAUSSIAN, step="all")
    assert len(mols) == 2


def test_parse_step_index():
    mols = parse_gaussian(WATER_GAUSSIAN, step="0")
    assert len(mols) == 1
    # First block has x=0.0 for oxygen
    a0 = mols[0].atoms[0]
    assert a0.cart_x == pytest.approx(0.0, abs=0.001)


# ------------------------------------------------------------------ #
#  Error handling                                                      #
# ------------------------------------------------------------------ #

def test_parse_invalid_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_gaussian("nonexistent_file.log")


def test_parse_no_orientation_raises(tmp_path):
    bad_log = tmp_path / "empty.log"
    bad_log.write_text("Some random text\nwithout any orientation blocks\n")
    with pytest.raises(ValueError, match="No coordinate block found"):
        parse_gaussian(str(bad_log))


# ------------------------------------------------------------------ #
#  IC structure checks                                                 #
# ------------------------------------------------------------------ #

def test_first_atom_is_anchor():
    mol = parse_gaussian(WATER_GAUSSIAN)[0]
    a = mol.atoms[0]
    assert a.bond_length is None
    assert a.bond_angle is None
    assert a.dihedral is None


def test_cartesian_stored():
    mol = parse_gaussian(WATER_GAUSSIAN)[0]
    for atom in mol.atoms:
        assert atom.cart_x is not None
        assert atom.cart_y is not None
        assert atom.cart_z is not None


def test_ic_computed():
    mol = parse_gaussian(WATER_GAUSSIAN)[0]
    for atom in mol.atoms[1:]:
        assert atom.bond_length is not None
