"""Tests for parsers/gamess_parser.py."""

from pathlib import Path

import numpy as np
import pytest

from molconvert.parsers.gamess_parser import parse_gamess

DATA_DIR = Path(__file__).parent / "data"
WATER_GAMESS = str(DATA_DIR / "water_gamess.out")


# ------------------------------------------------------------------ #
#  Basic parsing                                                       #
# ------------------------------------------------------------------ #

def test_parse_returns_list():
    result = parse_gamess(WATER_GAMESS)
    assert isinstance(result, list)


def test_parse_atom_count():
    mols = parse_gamess(WATER_GAMESS)
    assert len(mols) == 1
    assert len(mols[0]) == 3


def test_parse_elements():
    mol = parse_gamess(WATER_GAMESS)[0]
    elements = [a.element for a in mol.atoms]
    assert elements == ["O", "H", "H"]


def test_parse_source_fmt():
    mol = parse_gamess(WATER_GAMESS)[0]
    assert mol.source_fmt == "gamess"


def test_parse_coordinates():
    mol = parse_gamess(WATER_GAMESS)[0]
    expected = np.array([
        [0.0000000000,  0.0000000000,  0.1177900000],
        [0.0000000000,  0.7579130000, -0.4711610000],
        [0.0000000000, -0.7579130000, -0.4711610000],
    ])
    actual = mol.get_positions()
    np.testing.assert_allclose(actual, expected, atol=0.001)


def test_parse_units_angstroms():
    mol = parse_gamess(WATER_GAMESS)[0]
    assert mol.metadata["units"] == "angstroms"


# ------------------------------------------------------------------ #
#  Error handling                                                      #
# ------------------------------------------------------------------ #

def test_parse_invalid_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_gamess("nonexistent_file.out")


def test_parse_no_coords_raises(tmp_path):
    bad_file = tmp_path / "empty.out"
    bad_file.write_text("GAMESS output with no coordinates\nJust some text.\n")
    with pytest.raises(ValueError, match="No coordinate block"):
        parse_gamess(str(bad_file))


# ------------------------------------------------------------------ #
#  Cartesian and IC checks                                             #
# ------------------------------------------------------------------ #

def test_cartesian_stored():
    mol = parse_gamess(WATER_GAMESS)[0]
    for atom in mol.atoms:
        assert atom.cart_x is not None
        assert atom.cart_y is not None
        assert atom.cart_z is not None


def test_first_atom_is_anchor():
    mol = parse_gamess(WATER_GAMESS)[0]
    a0 = mol.atoms[0]
    assert a0.bond_length is None
    assert a0.bond_angle is None
    assert a0.dihedral is None


def test_ic_computed():
    mol = parse_gamess(WATER_GAMESS)[0]
    # Atom at index 1 (H) should have bond_length set
    assert mol.atoms[1].bond_length is not None
    # Atom at index 2 (H) should have bond_length and bond_angle set
    assert mol.atoms[2].bond_length is not None
    assert mol.atoms[2].bond_angle is not None


# ------------------------------------------------------------------ #
#  Step selection                                                      #
# ------------------------------------------------------------------ #

def test_parse_step_last():
    mols = parse_gamess(WATER_GAMESS, step="last")
    assert len(mols) == 1


# ------------------------------------------------------------------ #
#  Metadata                                                            #
# ------------------------------------------------------------------ #

def test_metadata_has_program():
    mol = parse_gamess(WATER_GAMESS)[0]
    assert mol.metadata["program"] == "gamess"


# ------------------------------------------------------------------ #
#  Bohr → Angstrom conversion                                         #
# ------------------------------------------------------------------ #

_BOHR_GAMESS_FILE = """\
          ----- RESULTS FROM SUCCESSFUL RHF OPTIMIZATION -----

 COORDINATES OF ALL ATOMS ARE (BOHR)
   ATOM   CHARGE       X              Y              Z
 -------- ------   ----------     ----------     ----------
 O         8.0     0.0000000000   0.0000000000   0.2225920000
 H         1.0     0.0000000000   1.4324410000  -0.8903680000
 H         1.0     0.0000000000  -1.4324410000  -0.8903680000
"""

_BOHR = 0.529177210903


def test_bohr_to_angstrom_conversion(tmp_path):
    """GAMESS output with BOHR coordinates should auto-convert to Angstroms."""
    bohr_file = tmp_path / "water_bohr.out"
    bohr_file.write_text(_BOHR_GAMESS_FILE)

    mol = parse_gamess(str(bohr_file))[0]
    assert mol.metadata["units"] == "bohr"

    # The first atom (O) has BOHR z=0.2225920000, in Angstroms ~0.11778
    assert mol.atoms[0].cart_z == pytest.approx(0.2225920000 * _BOHR, abs=0.001)
    # H y in Bohr=1.4324410000, in Angstroms ~0.75791
    assert mol.atoms[1].cart_y == pytest.approx(1.4324410000 * _BOHR, abs=0.001)

