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


def test_save_sdf_multi_creates_file(tmp_path):
    mols = parse_sdf(MINI_SDF)
    out = str(tmp_path / "multi.sdf")
    save_sdf_multi(mols, out)
    assert Path(out).exists()
    content = Path(out).read_text()
    assert content.count("$$$$") == 2


def test_to_sdf_raises_without_positions():
    from molconvert.converters.zmat_to_json import zmat_to_molecule
    from molconvert.converters.json_to_zmat import molecule_to_zmat
    mol = parse_sdf(MINI_SDF)[0]
    zmat_text = molecule_to_zmat(mol)
    mol_no_pos = zmat_to_molecule(zmat_text)
    # Non-anchor atoms have no positions yet
    with pytest.raises(ValueError):
        molecule_to_sdf(mol_no_pos)


# ------------------------------------------------------------------ #
#  Round-trip: SDF → MoleculeIC → SDF                                 #
# ------------------------------------------------------------------ #

def test_sdf_roundtrip_atom_count_ethanol(ethanol_sdf):
    counts = [l for l in ethanol_sdf.splitlines() if "V2000" in l][0]
    assert int(counts[:3]) == 9


def test_sdf_roundtrip_bond_count_ethanol(ethanol_sdf):
    counts = [l for l in ethanol_sdf.splitlines() if "V2000" in l][0]
    assert int(counts[3:6]) == 8


def test_sdf_roundtrip_rmsd_near_zero(ethanol):
    sdf_text = molecule_to_sdf(ethanol)
    # Parse back via RDKit-based sdf_parser (write to tmp file)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".sdf", mode="w", delete=False) as f:
        f.write(sdf_text + "\n")
        tmp = f.name
    try:
        mols_back = parse_sdf(tmp)
        assert len(mols_back) == 1
        r = rmsd_molecules(ethanol, mols_back[0])
        assert r < 1e-3
    finally:
        os.unlink(tmp)


# ------------------------------------------------------------------ #
#  CLI — SDF input paths                                               #
# ------------------------------------------------------------------ #

def test_cli_sdf_to_pdb_stdout(capsys):
    run_convert([MINI_SDF, "--to", "pdb"])
    out, _ = capsys.readouterr()
    atom_lines = [l for l in out.splitlines() if l.startswith("ATOM")]
    assert len(atom_lines) == 13   # ethanol 9 + acetone 4


def test_cli_sdf_to_pdb_uses_model_records(capsys):
    run_convert([MINI_SDF, "--to", "pdb"])
    out, _ = capsys.readouterr()
    assert "MODEL" in out


def test_cli_sdf_to_json_stdout(capsys):
    run_convert([MINI_SDF, "--to", "json"])
    out, _ = capsys.readouterr()
    # Multiple JSON objects — take first
    first_json = out.strip().split("\n{")[0]
    data = json.loads(first_json)
    assert "atoms" in data
    assert data["source_fmt"] == "sdf"


def test_cli_sdf_to_sdf_stdout(capsys):
    run_convert([MINI_SDF, "--to", "sdf"])
    out, _ = capsys.readouterr()
    assert out.count("$$$$") == 2
    assert "V2000" in out


def test_cli_sdf_to_zmat_stdout(capsys):
    run_convert([MINI_SDF, "--to", "zmat"])
    out, _ = capsys.readouterr()
    assert out.startswith("ZMAT ")
    assert "END" in out


def test_cli_sdf_to_pdb_file(tmp_path, capsys):
    out_path = str(tmp_path / "out.pdb")
    run_convert([MINI_SDF, "--to", "pdb", "-o", out_path])
    assert Path(out_path).exists()
    content = Path(out_path).read_text()
    assert "ATOM" in content


def test_cli_sdf_to_sdf_file(tmp_path, capsys):
    out_path = str(tmp_path / "out.sdf")
    run_convert([MINI_SDF, "--to", "sdf", "-o", out_path])
    assert Path(out_path).exists()
    content = Path(out_path).read_text()
    assert content.count("$$$$") == 2


def test_cli_sdf_remove_hydrogens(capsys):
    run_convert([MINI_SDF, "--to", "pdb", "--remove-hydrogens"])
    out, _ = capsys.readouterr()
    atom_lines = [l for l in out.splitlines() if l.startswith("ATOM")]
    # ethanol heavy atoms: C,C,O = 3; acetone heavy atoms: C,C,C,O = 4
    assert len(atom_lines) == 7


# ------------------------------------------------------------------ #
#  CLI — PDB input → SDF output                                        #
# ------------------------------------------------------------------ #

def test_cli_pdb_to_sdf_stdout(capsys):
    run_convert([MINI_PDB, "--to", "sdf"])
    out, _ = capsys.readouterr()
    assert "V2000" in out
    assert "$$$$" in out


def test_cli_pdb_to_sdf_atom_count(capsys):
    run_convert([MINI_PDB, "--to", "sdf"])
    out, _ = capsys.readouterr()
    counts = [l for l in out.splitlines() if "V2000" in l][0]
    assert int(counts[:3]) == 5   # mini.pdb has 5 atoms


def test_cli_pdb_to_sdf_file(tmp_path, capsys):
    out_path = str(tmp_path / "out.sdf")
    run_convert([MINI_PDB, "--to", "sdf", "-o", out_path])
    assert Path(out_path).exists()


# ------------------------------------------------------------------ #
#  CLI — ZMAT input → SDF output                                       #
# ------------------------------------------------------------------ #

def test_cli_zmat_to_sdf_stdout(tmp_path, capsys):
    from molconvert.converters.json_to_zmat import save_zmat
    mol = parse_pdb(MINI_PDB)[0]
    zmat_path = str(tmp_path / "mol.zmat")
    save_zmat(mol, zmat_path)

    run_convert([zmat_path, "--to", "sdf"])
    out, _ = capsys.readouterr()
    assert "V2000" in out
    assert "$$$$" in out


def test_cli_zmat_to_sdf_atom_count(tmp_path, capsys):
    from molconvert.converters.json_to_zmat import save_zmat
    mol = parse_pdb(MINI_PDB)[0]
    zmat_path = str(tmp_path / "mol.zmat")
    save_zmat(mol, zmat_path)

    run_convert([zmat_path, "--to", "sdf"])
    out, _ = capsys.readouterr()
    counts = [l for l in out.splitlines() if "V2000" in l][0]
    assert int(counts[:3]) == 5


# ------------------------------------------------------------------ #
#  CLI — error paths                                                   #
# ------------------------------------------------------------------ #

def test_cli_missing_sdf_exits():
    with pytest.raises(SystemExit):
        run_convert(["ghost.sdf", "--to", "pdb"])


def test_cli_unsupported_extension_exits():
    with pytest.raises(SystemExit):
        run_convert(["molecule.mol2"])
