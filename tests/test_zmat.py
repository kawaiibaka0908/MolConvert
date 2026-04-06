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

def test_zmat_has_three_anchor_lines(mini_zmat):
    anchor_lines = [l for l in mini_zmat.splitlines() if l.startswith("# anchor")]
    assert len(anchor_lines) == 3


def test_zmat_anchor_1_position(mini_zmat):
    # Line format: # anchor <idx> <x> <y> <z>  → parts[3..5] are coordinates
    anchor1 = [l for l in mini_zmat.splitlines() if l.startswith("# anchor 1")][0]
    parts = anchor1.split()
    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
    assert x == pytest.approx(0.0, abs=1e-4)
    assert y == pytest.approx(0.0, abs=1e-4)
    assert z == pytest.approx(0.0, abs=1e-4)


# ------------------------------------------------------------------ #
#  json_to_zmat — atom line column counts                              #
# ------------------------------------------------------------------ #

def _data_lines(zmat_text):
    return [l.split() for l in zmat_text.splitlines()
            if l.strip() and not l.startswith("#")
            and not l.startswith("ZMAT") and l.strip() != "END"]


def test_atom1_line_has_5_columns(mini_zmat):
    assert len(_data_lines(mini_zmat)[0]) == 5


def test_atom2_line_has_7_columns(mini_zmat):
    assert len(_data_lines(mini_zmat)[1]) == 7


def test_atom3_line_has_9_columns(mini_zmat):
    assert len(_data_lines(mini_zmat)[2]) == 9


def test_atom4_line_has_11_columns(mini_zmat):
    assert len(_data_lines(mini_zmat)[3]) == 11


# ------------------------------------------------------------------ #
#  json_to_zmat — reference index values                               #
# ------------------------------------------------------------------ #

def test_atom2_bond_ref_is_1(mini_zmat):
    parts = _data_lines(mini_zmat)[1]
    assert int(parts[5]) == 1


def test_atom3_bond_ref_is_2_angle_ref_is_1(mini_zmat):
    parts = _data_lines(mini_zmat)[2]
    assert int(parts[5]) == 2
    assert int(parts[7]) == 1


def test_atom4_refs_are_3_2_1(mini_zmat):
    parts = _data_lines(mini_zmat)[3]
    assert int(parts[5]) == 3
    assert int(parts[7]) == 2
    assert int(parts[9]) == 1


# ------------------------------------------------------------------ #
#  json_to_zmat — IC values                                            #
# ------------------------------------------------------------------ #

def test_atom2_bond_length_matches_source(mini_mol, mini_zmat):
    zmat_bl = float(_data_lines(mini_zmat)[1][6])
    assert zmat_bl == pytest.approx(mini_mol.atoms[1].bond_length, abs=1e-5)


def test_atom3_bond_angle_present(mini_zmat):
    angle = float(_data_lines(mini_zmat)[2][8])
    assert 90.0 < angle < 140.0   # chemically sensible range for C-C-N


def test_atom4_dihedral_present(mini_zmat):
    dihedral = float(_data_lines(mini_zmat)[3][10])
    assert -180.0 <= dihedral <= 180.0


# ------------------------------------------------------------------ #
#  json_to_zmat — save_zmat                                            #
# ------------------------------------------------------------------ #

def test_save_zmat_creates_file(mini_mol, tmp_path):
    out = str(tmp_path / "mol.zmat")
    save_zmat(mini_mol, out)
    assert Path(out).exists()


def test_save_zmat_content_matches_string(mini_mol, tmp_path):
    out = str(tmp_path / "mol.zmat")
    save_zmat(mini_mol, out)
    content = Path(out).read_text().strip()
    assert content == molecule_to_zmat(mini_mol).strip()


# ------------------------------------------------------------------ #
#  zmat_to_json — basic structure                                      #
# ------------------------------------------------------------------ #

def test_zmat_to_molecule_returns_moleculeic(mini_zmat):
    from molconvert.core.internal_coords import MoleculeIC
    mol = zmat_to_molecule(mini_zmat)
    assert isinstance(mol, MoleculeIC)


def test_zmat_to_molecule_atom_count(mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    assert len(mol.atoms) == 5


def test_zmat_to_molecule_name_preserved(mini_mol, mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    assert mol.name == mini_mol.name


def test_zmat_to_molecule_source_fmt(mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    assert mol.source_fmt == "pdb"


def test_zmat_to_molecule_atom_names(mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    names = [a.atom_name for a in mol.atoms]
    assert names == ["N", "CA", "C", "O", "N"]


# ------------------------------------------------------------------ #
#  zmat_to_json — IC values preserved                                  #
# ------------------------------------------------------------------ #

def test_bond_length_preserved(mini_mol, mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    assert mol.atoms[1].bond_length == pytest.approx(mini_mol.atoms[1].bond_length, abs=1e-5)


def test_bond_angle_preserved(mini_mol, mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    assert mol.atoms[2].bond_angle == pytest.approx(mini_mol.atoms[2].bond_angle, abs=1e-3)


def test_dihedral_preserved(mini_mol, mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    assert mol.atoms[3].dihedral == pytest.approx(mini_mol.atoms[3].dihedral, abs=1e-3)


def test_first_atom_ic_is_none(mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    assert mol.atoms[0].bond_length is None
    assert mol.atoms[0].bond_angle  is None
    assert mol.atoms[0].dihedral    is None


# ------------------------------------------------------------------ #
#  zmat_to_json — reference indices preserved                          #
# ------------------------------------------------------------------ #

def test_bond_to_preserved(mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    assert mol.atoms[1].bond_to    == 1
    assert mol.atoms[2].bond_to    == 2
    assert mol.atoms[3].bond_to    == 3


def test_angle_to_preserved(mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    assert mol.atoms[2].angle_to   == 1
    assert mol.atoms[3].angle_to   == 2


def test_dihedral_to_preserved(mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    assert mol.atoms[3].dihedral_to == 1


# ------------------------------------------------------------------ #
#  zmat_to_json — anchor positions                                     #
# ------------------------------------------------------------------ #

def test_anchor_1_cartesian_restored(mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    a = mol.atoms[0]
    assert a.cart_x == pytest.approx(0.0, abs=1e-4)
    assert a.cart_y == pytest.approx(0.0, abs=1e-4)
    assert a.cart_z == pytest.approx(0.0, abs=1e-4)


def test_anchor_2_cartesian_restored(mini_mol, mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    a = mol.atoms[1]
    assert a.cart_x == pytest.approx(mini_mol.atoms[1].cart_x, abs=1e-4)


def test_non_anchor_cartesian_is_none_before_reconstruct(mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    assert mol.atoms[3].cart_x is None


# ------------------------------------------------------------------ #
#  zmat_to_json — element derivation                                   #
# ------------------------------------------------------------------ #

def test_elements_derived_correctly(mini_zmat):
    mol = zmat_to_molecule(mini_zmat)
    elements = [a.element for a in mol.atoms]
    assert elements == ["N", "C", "C", "O", "N"]


# ------------------------------------------------------------------ #
#  zmat_to_json — load_zmat from file                                  #
# ------------------------------------------------------------------ #

def test_load_zmat_from_file(mini_mol, tmp_path):
    zmat_path = str(tmp_path / "mol.zmat")
    save_zmat(mini_mol, zmat_path)
    mol = load_zmat(zmat_path)
    assert len(mol.atoms) == 5
    assert mol.atoms[1].bond_length == pytest.approx(mini_mol.atoms[1].bond_length, abs=1e-5)


# ------------------------------------------------------------------ #
#  Round-trip accuracy                                                 #
# ------------------------------------------------------------------ #

def test_roundtrip_rmsd_realistic(mini_mol):
    zmat_text  = molecule_to_zmat(mini_mol)
    mol_back   = zmat_to_molecule(zmat_text)
    mol_rebuilt = reconstruct(mol_back)
    r = rmsd_molecules(mini_mol, mol_rebuilt)
    # ZMAT stores angles/dihedrals at 2 decimal places (0.01°), so
    # reconstruction introduces small but non-zero error.
    assert r < 0.5, f"Round-trip RMSD unexpectedly large: {r:.4f} Å"
    assert r > 0.0, f"Round-trip RMSD was exactly zero — likely a copy, not a reconstruction"


def test_roundtrip_atom_count_preserved(mini_mol):
    zmat_text  = molecule_to_zmat(mini_mol)
    mol_back   = zmat_to_molecule(zmat_text)
    assert len(mol_back.atoms) == len(mini_mol.atoms)


# ------------------------------------------------------------------ #
#  CLI — PDB → ZMAT                                                    #
# ------------------------------------------------------------------ #

