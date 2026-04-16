"""Tests for builders/reconstruct.py."""

import copy
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from molconvert.parsers.pdb_parser import parse_pdb
from molconvert.builders.reconstruct import reconstruct, to_pdb, save_pdb
from molconvert.core.internal_coords import MoleculeIC, AtomIC

MINI_PDB = str(Path(__file__).parent / "data" / "mini.pdb")


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def mol_orig():
    return parse_pdb(MINI_PDB)[0]


@pytest.fixture
def mol_recon(mol_orig):
    return reconstruct(mol_orig)


# ------------------------------------------------------------------ #
#  reconstruct — basic behaviour                                       #
# ------------------------------------------------------------------ #

def test_reconstruct_returns_molecule_ic(mol_orig):
    result = reconstruct(mol_orig)
    assert isinstance(result, MoleculeIC)


def test_reconstruct_does_not_mutate_input(mol_orig):
    original_positions = [(a.cart_x, a.cart_y, a.cart_z) for a in mol_orig.atoms]
    reconstruct(mol_orig)
    for atom, (x, y, z) in zip(mol_orig.atoms, original_positions):
        assert atom.cart_x == x
        assert atom.cart_y == y
        assert atom.cart_z == z


def test_reconstruct_same_atom_count(mol_orig, mol_recon):
    assert len(mol_recon) == len(mol_orig)


def test_reconstruct_all_positions_set(mol_recon):
    for atom in mol_recon.atoms:
        assert atom.cart_x is not None
        assert atom.cart_y is not None
        assert atom.cart_z is not None


# ------------------------------------------------------------------ #
#  Numerical accuracy                                                  #
# ------------------------------------------------------------------ #

