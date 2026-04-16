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

def test_reconstruct_rmsd_realistic(mol_orig, mol_recon):
    orig  = mol_orig.get_positions()
    recon = mol_recon.get_positions()
    diff  = orig - recon
    rmsd  = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    # Angles/dihedrals are rounded to 0.01° before NeRF placement, so
    # reconstruction is deliberately imprecise (not a trivial coordinate copy).
    assert rmsd < 0.5,  f"Round-trip RMSD too large: {rmsd:.4f} Å"
    assert rmsd > 0.0,  f"Round-trip RMSD is exactly zero — coordinates were copied, not reconstructed"


def test_reconstruct_per_atom_deviation_bounded(mol_orig, mol_recon):
    orig  = mol_orig.get_positions()
    recon = mol_recon.get_positions()
    for i, (p1, p2) in enumerate(zip(orig, recon)):
        dev = float(np.linalg.norm(p1 - p2))
        assert dev < 1.0, (
            f"Atom {i} ({mol_orig.atoms[i].atom_name}) deviation too large: {dev:.4f} Å"
        )


# ------------------------------------------------------------------ #
#  Error handling                                                      #
# ------------------------------------------------------------------ #

def test_reconstruct_missing_anchor_position_raises(mol_orig):
    bad_mol = copy.deepcopy(mol_orig)
    bad_mol.atoms[0].cart_x = None
    bad_mol.atoms[0].cart_y = None
    bad_mol.atoms[0].cart_z = None
    with pytest.raises(ValueError, match="no Cartesian position"):
        reconstruct(bad_mol)


def test_reconstruct_missing_bond_length_raises(mol_orig):
    bad_mol = copy.deepcopy(mol_orig)
    bad_mol.atoms[3].bond_length = None   # atom index 3 needs full IC
    with pytest.raises(ValueError, match="bond_length"):
        reconstruct(bad_mol)


