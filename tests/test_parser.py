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


