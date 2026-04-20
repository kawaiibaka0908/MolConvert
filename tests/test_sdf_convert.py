"""
Tests for SDF ↔ PDB / SDF ↔ JSON / SDF ↔ ZMAT conversions.

Covers:
  - to_sdf builder  : MoleculeIC → SDF string / file
  - bond inference  : correct atom/bond counts, no phantom bonds
  - SDF round-trip  : parse → to_sdf atom/bond counts preserved
  - CLI paths       : .sdf --to pdb/json/sdf/zmat
                      .pdb --to sdf
                      .zmat --to sdf
  - --remove-hydrogens flag with SDF input
  - Error paths     : missing file, unsupported extension
"""

import json
from pathlib import Path

import pytest

from molconvert.parsers.sdf_parser import parse_sdf
from molconvert.parsers.pdb_parser import parse_pdb
from molconvert.builders.reconstruct import reconstruct
from molconvert.builders.to_sdf import (
    molecule_to_sdf,
    save_sdf,
    molecules_to_sdf,
    save_sdf_multi,
)
from molconvert.analysis import rmsd_molecules
from molconvert.cli.main import run_convert

MINI_SDF = str(Path(__file__).parent / "data" / "mini.sdf")
MINI_PDB = str(Path(__file__).parent / "data" / "mini.pdb")


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def ethanol():
    return parse_sdf(MINI_SDF)[0]   # 9 atoms (C, C, O, 6H)


@pytest.fixture
def acetone():
    return parse_sdf(MINI_SDF)[1]   # 4 atoms (C, C, C, O)


@pytest.fixture
def ethanol_sdf(ethanol):
    return molecule_to_sdf(ethanol)


@pytest.fixture
def acetone_sdf(acetone):
    return molecule_to_sdf(acetone)


# ------------------------------------------------------------------ #
#  sdf_parser — new reference fields (bond_to / angle_to / dihedral_to)
# ------------------------------------------------------------------ #

def test_sdf_parser_bond_to_set(ethanol):
    assert ethanol.atoms[1].bond_to == 1
    assert ethanol.atoms[2].bond_to == 2
    assert ethanol.atoms[3].bond_to == 3


def test_sdf_parser_angle_to_set(ethanol):
    assert ethanol.atoms[2].angle_to == 1
    assert ethanol.atoms[3].angle_to == 2


def test_sdf_parser_dihedral_to_set(ethanol):
    assert ethanol.atoms[3].dihedral_to == 1


def test_sdf_parser_anchor_refs_are_none(ethanol):
    assert ethanol.atoms[0].bond_to is None
    assert ethanol.atoms[0].angle_to is None
    assert ethanol.atoms[0].dihedral_to is None


# ------------------------------------------------------------------ #
#  to_sdf — SDF record structure                                       #
# ------------------------------------------------------------------ #

def test_sdf_starts_with_mol_title(ethanol, ethanol_sdf):
    assert ethanol_sdf.splitlines()[0] == "ethanol"


def test_sdf_has_end_marker(ethanol_sdf):
    assert "M  END" in ethanol_sdf


def test_sdf_has_record_separator(ethanol_sdf):
    assert "$$$$" in ethanol_sdf


def test_sdf_counts_line_ethanol(ethanol_sdf):
    counts = [l for l in ethanol_sdf.splitlines() if "V2000" in l][0]
    n_atoms = int(counts[:3])
    n_bonds = int(counts[3:6])
    assert n_atoms == 9
    assert n_bonds == 8   # ethanol has 8 bonds


def test_sdf_counts_line_acetone(acetone_sdf):
    counts = [l for l in acetone_sdf.splitlines() if "V2000" in l][0]
    n_atoms = int(counts[:3])
    n_bonds = int(counts[3:6])
    assert n_atoms == 4
    assert n_bonds == 3   # acetone has 3 bonds


def _atom_coord_lines(sdf_text):
    """Return (x, y, z) tuples for V2000 atom block lines (have decimal points)."""
    coords = []
    for l in sdf_text.splitlines():
        parts = l.split()
        if len(parts) >= 4 and "." in parts[0]:
            try:
                coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                pass
    return coords


def test_sdf_atom_block_length(ethanol_sdf):
    assert len(_atom_coord_lines(ethanol_sdf)) == 9


def test_sdf_atom_coordinates_present(ethanol_sdf):
    assert len(_atom_coord_lines(ethanol_sdf)) == 9


def test_sdf_first_atom_at_origin(ethanol_sdf):
    x, y, z = _atom_coord_lines(ethanol_sdf)[0]
    assert x == pytest.approx(0.0, abs=1e-3)
    assert y == pytest.approx(0.0, abs=1e-3)
    assert z == pytest.approx(0.0, abs=1e-3)


# ------------------------------------------------------------------ #
#  to_sdf — bond inference correctness                                 #
# ------------------------------------------------------------------ #

def test_bond_count_ethanol(ethanol):
    from molconvert.builders.to_sdf import _infer_bonds
    bonds = _infer_bonds(ethanol.atoms)
    assert len(bonds) == 8


def test_bond_count_acetone(acetone):
    from molconvert.builders.to_sdf import _infer_bonds
    bonds = _infer_bonds(acetone.atoms)
    assert len(bonds) == 3


def test_bonds_are_1based(ethanol):
    from molconvert.builders.to_sdf import _infer_bonds
    bonds = _infer_bonds(ethanol.atoms)
    for a1, a2, _ in bonds:
        assert a1 >= 1
        assert a2 >= 1
        assert a1 <= len(ethanol.atoms)
        assert a2 <= len(ethanol.atoms)


def test_all_bond_types_are_single(ethanol):
    from molconvert.builders.to_sdf import _infer_bonds
    bonds = _infer_bonds(ethanol.atoms)
    assert all(btype == 1 for _, _, btype in bonds)


def test_no_self_bonds(ethanol):
    from molconvert.builders.to_sdf import _infer_bonds
    bonds = _infer_bonds(ethanol.atoms)
    assert all(a1 != a2 for a1, a2, _ in bonds)


# ------------------------------------------------------------------ #
#  to_sdf — file I/O                                                   #
# ------------------------------------------------------------------ #

def test_save_sdf_creates_file(ethanol, tmp_path):
    out = str(tmp_path / "out.sdf")
    save_sdf(ethanol, out)
    assert Path(out).exists()


def test_save_sdf_content_matches_string(ethanol, tmp_path):
    out = str(tmp_path / "out.sdf")
    save_sdf(ethanol, out)
    assert Path(out).read_text().strip() == molecule_to_sdf(ethanol).strip()


def test_molecules_to_sdf_has_two_records():
    mols = parse_sdf(MINI_SDF)
    text = molecules_to_sdf(mols)
    assert text.count("$$$$") == 2


