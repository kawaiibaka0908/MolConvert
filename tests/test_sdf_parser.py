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


