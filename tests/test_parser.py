"""Tests for parsers/pdb_parser.py."""

import math
from pathlib import Path

import pytest

from molconvert.parsers.pdb_parser import parse_pdb
from molconvert.core.internal_coords import MoleculeIC, AtomIC

MINI_PDB = str(Path(__file__).parent / "data" / "mini.pdb")


# ------------------------------------------------------------------ #
#  Basic parsing                                                       #
# ------------------------------------------------------------------ #

def test_parse_returns_list_of_molecules():
    result = parse_pdb(MINI_PDB)
    assert isinstance(result, list)
    assert len(result) == 1


def test_parse_returns_molecule_ic():
    mol = parse_pdb(MINI_PDB)[0]
    assert isinstance(mol, MoleculeIC)


def test_parse_atom_count():
    mol = parse_pdb(MINI_PDB)[0]
    assert len(mol) == 5


def test_parse_molecule_name_contains_chain():
    mol = parse_pdb(MINI_PDB)[0]
    assert mol.name.endswith("_A")


def test_parse_source_fmt():
    mol = parse_pdb(MINI_PDB)[0]
    assert mol.source_fmt == "pdb"


# ------------------------------------------------------------------ #
#  Atom identity fields                                                #
# ------------------------------------------------------------------ #

def test_atom_names():
    mol = parse_pdb(MINI_PDB)[0]
    names = [a.atom_name for a in mol.atoms]
    assert names == ["N", "CA", "C", "O", "N"]


def test_atom_serials():
    mol = parse_pdb(MINI_PDB)[0]
    serials = [a.atom_serial for a in mol.atoms]
    assert serials == [1, 2, 3, 4, 5]


def test_atom_residue_names():
    mol = parse_pdb(MINI_PDB)[0]
    res_names = [a.residue_name for a in mol.atoms]
    assert all(r == "ALA" for r in res_names)


def test_atom_chain_ids():
    mol = parse_pdb(MINI_PDB)[0]
    assert all(a.chain_id == "A" for a in mol.atoms)


def test_atom_elements():
    mol = parse_pdb(MINI_PDB)[0]
    elements = [a.element for a in mol.atoms]
    assert elements == ["N", "C", "C", "O", "N"]


# ------------------------------------------------------------------ #
#  Internal coordinate values                                          #
# ------------------------------------------------------------------ #

def test_first_atom_is_anchor():
    mol = parse_pdb(MINI_PDB)[0]
    atom0 = mol.atoms[0]
    assert atom0.bond_length is None
    assert atom0.bond_angle  is None
    assert atom0.dihedral    is None
    assert atom0.is_anchor


def test_second_atom_has_bond_length_only():
    mol = parse_pdb(MINI_PDB)[0]
    atom1 = mol.atoms[1]
    assert atom1.bond_length is not None
    assert atom1.bond_angle  is None
    assert atom1.dihedral    is None


def test_third_atom_has_bond_length_and_angle():
    mol = parse_pdb(MINI_PDB)[0]
    atom2 = mol.atoms[2]
    assert atom2.bond_length is not None
    assert atom2.bond_angle  is not None
    assert atom2.dihedral    is None


def test_fourth_atom_has_all_ic():
    mol = parse_pdb(MINI_PDB)[0]
    atom3 = mol.atoms[3]
    assert atom3.bond_length is not None
    assert atom3.bond_angle  is not None
    assert atom3.dihedral    is not None


def test_bond_length_ca_n_approx():
    # N→CA in mini.pdb is exactly 1.458 Å
    mol = parse_pdb(MINI_PDB)[0]
    bl = mol.atoms[1].bond_length
    assert abs(bl - 1.458) < 0.001


def test_cartesian_positions_stored():
    mol = parse_pdb(MINI_PDB)[0]
    for atom in mol.atoms:
        assert atom.cart_x is not None
        assert atom.cart_y is not None
        assert atom.cart_z is not None


