"""
Tests for the ZMAT converter layer.

Covers:
  - json_to_zmat  : MoleculeIC → ZMAT string / file
  - zmat_to_json  : ZMAT string / file → MoleculeIC
  - round-trip accuracy (PDB → ZMAT → PDB RMSD near zero)
  - CLI paths: --to zmat, .zmat --to pdb, .zmat --to internal
  - backward-compat: old -f/--format flag still works
"""

import json
import math
from pathlib import Path

import pytest

from molconvert.parsers.pdb_parser import parse_pdb
from molconvert.converters.json_to_zmat import molecule_to_zmat, save_zmat
from molconvert.converters.zmat_to_json import zmat_to_molecule, load_zmat
from molconvert.builders.reconstruct import reconstruct
from molconvert.analysis import rmsd_molecules
from molconvert.cli.main import run_convert

MINI_PDB = str(Path(__file__).parent / "data" / "mini.pdb")


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def mini_mol():
    return parse_pdb(MINI_PDB)[0]


@pytest.fixture
def mini_zmat(mini_mol):
    return molecule_to_zmat(mini_mol)


# ------------------------------------------------------------------ #
#  json_to_zmat — structure                                            #
# ------------------------------------------------------------------ #

def test_zmat_starts_with_zmat_header(mini_zmat):
    assert mini_zmat.startswith("ZMAT ")


def test_zmat_header_contains_molecule_name(mini_mol, mini_zmat):
    assert mini_mol.name in mini_zmat.splitlines()[0]


def test_zmat_ends_with_end(mini_zmat):
    assert mini_zmat.strip().splitlines()[-1] == "END"


def test_zmat_has_correct_atom_count(mini_zmat):
    data_lines = [l for l in mini_zmat.splitlines()
                  if l.strip() and not l.startswith("#") and not l.startswith("ZMAT") and l.strip() != "END"]
    assert len(data_lines) == 5


def test_zmat_contains_source_fmt(mini_zmat):
    assert "# source_fmt pdb" in mini_zmat


def test_zmat_has_n_atoms_comment(mini_zmat):
    assert "# n_atoms 5" in mini_zmat


# ------------------------------------------------------------------ #
#  json_to_zmat — anchor lines                                         #
# ------------------------------------------------------------------ #

