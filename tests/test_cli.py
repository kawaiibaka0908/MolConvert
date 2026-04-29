"""Tests for cli/main.py — run_convert and run_rmsd."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

from molconvert.cli.main import run_convert, run_rmsd

MINI_PDB = str(Path(__file__).parent / "data" / "mini.pdb")


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def capture_stdout(func, args, capsys):
    """Run func(args), return (stdout, stderr) strings."""
    func(args)
    captured = capsys.readouterr()
    return captured.out, captured.err


# ------------------------------------------------------------------ #
#  convert — json output (default)                                     #
# ------------------------------------------------------------------ #

def test_convert_json_stdout(capsys):
    run_convert([MINI_PDB])
    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert "atoms" in data
    assert len(data["atoms"]) == 5


def test_convert_json_has_ic_fields(capsys):
    run_convert([MINI_PDB])
    out, _ = capsys.readouterr()
    data = json.loads(out)
    atom3 = data["atoms"][3]
    assert atom3["bond_length"] is not None
    assert atom3["bond_angle"]  is not None
    assert atom3["dihedral"]    is not None


def test_convert_json_first_atom_is_anchor(capsys):
    run_convert([MINI_PDB])
    out, _ = capsys.readouterr()
    data = json.loads(out)
    atom0 = data["atoms"][0]
    assert atom0["bond_length"] is None
    assert atom0["bond_angle"]  is None
    assert atom0["dihedral"]    is None


def test_convert_json_to_file(tmp_path, capsys):
    out_path = str(tmp_path / "out.json")
    run_convert([MINI_PDB, "-o", out_path])
    content = Path(out_path).read_text()
    data = json.loads(content)
    assert len(data["atoms"]) == 5


# ------------------------------------------------------------------ #
#  convert — pdb output                                                #
# ------------------------------------------------------------------ #

def test_convert_pdb_stdout(capsys):
    run_convert([MINI_PDB, "-f", "pdb"])
    out, _ = capsys.readouterr()
    atom_lines = [l for l in out.splitlines() if l.startswith("ATOM")]
    assert len(atom_lines) == 5


def test_convert_pdb_ends_with_end(capsys):
    run_convert([MINI_PDB, "-f", "pdb"])
    out, _ = capsys.readouterr()
    assert "END" in out


def test_convert_pdb_to_file(tmp_path, capsys):
    out_path = str(tmp_path / "out.pdb")
    run_convert([MINI_PDB, "-f", "pdb", "-o", out_path])
    content = Path(out_path).read_text()
    assert "ATOM" in content


# ------------------------------------------------------------------ #
#  convert — chain filter                                              #
# ------------------------------------------------------------------ #

def test_convert_chain_a(capsys):
    run_convert([MINI_PDB, "--chain", "A"])
    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["name"].endswith("_A")


def test_convert_invalid_chain_exits():
    with pytest.raises(SystemExit):
        run_convert([MINI_PDB, "--chain", "Z"])


# ------------------------------------------------------------------ #
#  convert — summary flag                                              #
# ------------------------------------------------------------------ #

def test_convert_summary_writes_to_stderr(capsys):
    run_convert([MINI_PDB, "--summary"])
    _, err = capsys.readouterr()
    assert "Bond lengths" in err


# ------------------------------------------------------------------ #
#  convert — error paths                                               #
# ------------------------------------------------------------------ #

def test_convert_missing_file_exits():
    with pytest.raises(SystemExit):
        run_convert(["nonexistent.pdb"])


# ------------------------------------------------------------------ #
#  rmsd — self test                                                    #
# ------------------------------------------------------------------ #

def test_rmsd_self_near_zero(capsys):
    run_rmsd([MINI_PDB, "--self"])
    out, _ = capsys.readouterr()
    # parse "RMSD: 0.0000 Å" line
    rmsd_line = [l for l in out.splitlines() if l.startswith("RMSD")][0]
    value = float(rmsd_line.split(":")[1].strip().split()[0])
    assert value < 1e-4


def test_rmsd_self_per_atom(capsys):
    run_rmsd([MINI_PDB, "--self", "--per-atom"])
    out, _ = capsys.readouterr()
    assert "Max deviation" in out
    # Table data rows are indented with exactly 4 spaces (file path lines use 2)
    data_lines = [l for l in out.splitlines() if l.startswith("    ") and not l.startswith("-----")]
    assert len(data_lines) == 5


def test_rmsd_self_with_filter(capsys):
    run_rmsd([MINI_PDB, "--self", "--filter", "CA"])
    out, _ = capsys.readouterr()
    assert "CA" in out or "RMSD" in out


# ------------------------------------------------------------------ #
#  rmsd — two files                                                    #
# ------------------------------------------------------------------ #

def test_rmsd_two_identical_files(capsys):
    run_rmsd([MINI_PDB, MINI_PDB])
    out, _ = capsys.readouterr()
    rmsd_line = [l for l in out.splitlines() if l.startswith("RMSD")][0]
    value = float(rmsd_line.split(":")[1].strip().split()[0])
    assert value == pytest.approx(0.0)


# ------------------------------------------------------------------ #
#  rmsd — error paths                                                  #
# ------------------------------------------------------------------ #

def test_rmsd_no_file2_no_self_exits():
    with pytest.raises(SystemExit):
        run_rmsd([MINI_PDB])


def test_rmsd_self_and_file2_exits():
    with pytest.raises(SystemExit):
        run_rmsd([MINI_PDB, MINI_PDB, "--self"])


def test_rmsd_missing_file_exits():
    with pytest.raises(SystemExit):
        run_rmsd(["ghost.pdb", "--self"])
