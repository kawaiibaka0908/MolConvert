"""Tests for parsers/sdf_parser.py."""

from pathlib import Path

import numpy as np
import pytest

from molconvert.parsers.sdf_parser import parse_sdf
from molconvert.builders.reconstruct import reconstruct
from molconvert.analysis import rmsd_molecules

MINI_SDF = str(Path(__file__).parent / "data" / "mini.sdf")


# ------------------------------------------------------------------ #
#  Basic parsing                                                       #
# ------------------------------------------------------------------ #

def test_parse_sdf_returns_list():
    result = parse_sdf(MINI_SDF)
    assert isinstance(result, list)


def test_parse_sdf_molecule_count():
    # mini.sdf contains 2 records: ethanol and acetone
    result = parse_sdf(MINI_SDF)
    assert len(result) == 2


def test_parse_sdf_source_fmt():
    for mol in parse_sdf(MINI_SDF):
        assert mol.source_fmt == "sdf"


def test_parse_sdf_molecule_names_unique():
    mols = parse_sdf(MINI_SDF)
    names = [m.name for m in mols]
    assert len(names) == len(set(names))


def test_parse_sdf_metadata_has_mol_title():
    mols = parse_sdf(MINI_SDF)
    for mol in mols:
        assert "mol_title" in mol.metadata


# ------------------------------------------------------------------ #
#  Atom-level checks                                                   #
# ------------------------------------------------------------------ #

def test_ethanol_atom_count():
    mol = parse_sdf(MINI_SDF)[0]   # ethanol: C, C, O, 6H = 9 atoms
    assert len(mol) == 9


def test_acetone_atom_count():
    mol = parse_sdf(MINI_SDF)[1]   # acetone: C, C, C, O = 4 atoms
    assert len(mol) == 4


def test_atom_serials_one_based():
    mol = parse_sdf(MINI_SDF)[0]
    serials = [a.atom_serial for a in mol.atoms]
    assert serials == list(range(1, len(mol) + 1))


def test_atom_names_include_element():
    mol = parse_sdf(MINI_SDF)[0]
    for atom in mol.atoms:
        assert atom.element in atom.atom_name


def test_atom_chain_id_default():
    for mol in parse_sdf(MINI_SDF):
        assert all(a.chain_id == "A" for a in mol.atoms)


def test_atom_residue_seq_default():
    for mol in parse_sdf(MINI_SDF):
        assert all(a.residue_seq == 1 for a in mol.atoms)


def test_cartesian_positions_stored():
    for mol in parse_sdf(MINI_SDF):
        for atom in mol.atoms:
            assert atom.cart_x is not None
            assert atom.cart_y is not None
            assert atom.cart_z is not None


def test_first_atom_is_anchor():
    for mol in parse_sdf(MINI_SDF):
        assert mol.atoms[0].bond_length is None
        assert mol.atoms[0].bond_angle  is None
        assert mol.atoms[0].dihedral    is None


def test_second_atom_has_bond_length_only():
    for mol in parse_sdf(MINI_SDF):
        a = mol.atoms[1]
        assert a.bond_length is not None
        assert a.bond_angle  is None
        assert a.dihedral    is None


def test_third_atom_has_bond_length_and_angle():
    for mol in parse_sdf(MINI_SDF):
        a = mol.atoms[2]
        assert a.bond_length is not None
        assert a.bond_angle  is not None
        assert a.dihedral    is None


def test_fourth_atom_has_all_ic():
    mol = parse_sdf(MINI_SDF)[0]   # ethanol has >= 4 atoms
    a = mol.atoms[3]
    assert a.bond_length is not None
    assert a.bond_angle  is not None
    assert a.dihedral    is not None


def test_ethanol_first_atom_at_origin():
    mol = parse_sdf(MINI_SDF)[0]
    a = mol.atoms[0]
    assert a.cart_x == pytest.approx(0.0)
    assert a.cart_y == pytest.approx(0.0)
    assert a.cart_z == pytest.approx(0.0)


# ------------------------------------------------------------------ #
#  Elements                                                            #
# ------------------------------------------------------------------ #

def test_ethanol_elements():
    mol = parse_sdf(MINI_SDF)[0]
    elements = [a.element for a in mol.atoms]
    assert elements[0] == "C"
    assert elements[1] == "C"
    assert elements[2] == "O"
    assert all(e == "H" for e in elements[3:])


def test_acetone_elements():
    mol = parse_sdf(MINI_SDF)[1]
    elements = [a.element for a in mol.atoms]
    assert elements.count("C") == 3
    assert elements.count("O") == 1


# ------------------------------------------------------------------ #
#  Round-trip reconstruction                                           #
# ------------------------------------------------------------------ #

