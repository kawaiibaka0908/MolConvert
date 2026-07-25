"""Tests for new CLI features: extended formats, batch mode, validation."""

import json
import os
import sys
import shutil
from pathlib import Path

import pytest

from molconvert.cli.main import run_convert

MINI_PDB = str(Path(__file__).parent / "data" / "mini.pdb")
MINI_SDF = str(Path(__file__).parent / "data" / "mini.sdf")
WATER_LOG = str(Path(__file__).parent / "data" / "water_gaussian.log")
WATER_OUT = str(Path(__file__).parent / "data" / "water_gamess.out")
ETHANOL_MOL2 = str(Path(__file__).parent / "data" / "ethanol.mol2")
DATA_DIR = str(Path(__file__).parent / "data")


# ------------------------------------------------------------------ #
#  New input format tests                                              #
# ------------------------------------------------------------------ #

class TestGaussianInput:
    """Convert Gaussian .log files."""

    def test_gaussian_log_to_json(self, capsys):
        run_convert([WATER_LOG, "--to", "json"])
        out, _ = capsys.readouterr()
        data = json.loads(out)
        assert "atoms" in data
        assert len(data["atoms"]) == 3  # water has 3 atoms

    def test_gaussian_log_to_pdb(self, capsys):
        run_convert([WATER_LOG, "--to", "pdb"])
        out, _ = capsys.readouterr()
        atom_lines = [l for l in out.splitlines() if l.startswith("ATOM")]
        assert len(atom_lines) == 3

    def test_gaussian_log_to_xyz(self, capsys):
        run_convert([WATER_LOG, "--to", "xyz"])
        out, _ = capsys.readouterr()
        lines = out.strip().splitlines()
        assert lines[0].strip() == "3"  # atom count

    def test_gaussian_out_with_input_format(self, tmp_path):
        """Copy the gaussian .log to a .out file and parse with --input-format."""
        out_file = str(tmp_path / "water.out")
        shutil.copy(WATER_LOG, out_file)
        result_path = str(tmp_path / "result.json")
        run_convert([out_file, "--input-format", "gaussian", "--to", "json",
                     "-o", result_path])
        data = json.loads(Path(result_path).read_text())
        assert len(data["atoms"]) == 3


class TestGamessInput:
    """Convert GAMESS .out files."""

    def test_gamess_out_to_json(self, capsys):
        run_convert([WATER_OUT, "--input-format", "gamess", "--to", "json"])
        out, _ = capsys.readouterr()
        data = json.loads(out)
        assert "atoms" in data
        assert len(data["atoms"]) == 3

    def test_gamess_out_to_xyz(self, capsys):
        run_convert([WATER_OUT, "--input-format", "gamess", "--to", "xyz"])
        out, _ = capsys.readouterr()
        lines = out.strip().splitlines()
        assert lines[0].strip() == "3"


class TestMol2Input:
    """Convert MOL2 input files."""

    def test_mol2_to_json(self, capsys):
        run_convert([ETHANOL_MOL2, "--to", "json"])
        out, _ = capsys.readouterr()
        data = json.loads(out)
        assert "atoms" in data

    def test_mol2_to_pdb(self, capsys):
        run_convert([ETHANOL_MOL2, "--to", "pdb"])
        out, _ = capsys.readouterr()
        assert "ATOM" in out

    def test_mol2_to_xyz(self, capsys):
        run_convert([ETHANOL_MOL2, "--to", "xyz"])
        out, _ = capsys.readouterr()
        lines = out.strip().splitlines()
        assert int(lines[0].strip()) > 0


# ------------------------------------------------------------------ #
#  New output format tests                                             #
# ------------------------------------------------------------------ #

class TestNewOutputFormats:
    """Test all new output formats from a PDB source."""

    def test_pdb_to_mol2(self, capsys):
        run_convert([MINI_SDF, "--to", "mol2"])
        out, _ = capsys.readouterr()
        assert "@<TRIPOS>MOLECULE" in out
        assert "@<TRIPOS>ATOM" in out
        assert "@<TRIPOS>BOND" in out

    def test_pdb_to_xyz(self, capsys):
        run_convert([MINI_PDB, "--to", "xyz"])
        out, _ = capsys.readouterr()
        lines = out.strip().splitlines()
        assert lines[0].strip() == "5"  # mini.pdb has 5 atoms

    def test_pdb_to_gro(self, capsys):
        run_convert([MINI_PDB, "--to", "gro"])
        out, _ = capsys.readouterr()
        lines = out.strip().splitlines()
        # title, count, 5 atoms, box = 8 lines
        assert len(lines) == 8

    def test_pdb_to_gjf(self, capsys):
        run_convert([MINI_PDB, "--to", "gjf"])
        out, _ = capsys.readouterr()
        assert "# HF/6-31G(d) opt" in out
        assert "0 1" in out

    def test_pdb_to_gjf_custom_route(self, capsys):
        run_convert([MINI_PDB, "--to", "gjf",
                     "--route", "# B3LYP/6-311G(d,p) opt freq"])
        out, _ = capsys.readouterr()
        assert "# B3LYP/6-311G(d,p) opt freq" in out

    def test_pdb_to_gjf_custom_charge(self, capsys):
        run_convert([MINI_PDB, "--to", "gjf", "--charge", "1", "--multiplicity", "2"])
        out, _ = capsys.readouterr()
        assert "1 2" in out

    def test_pdb_to_inp(self, capsys):
        run_convert([MINI_PDB, "--to", "inp"])
        out, _ = capsys.readouterr()
        assert "$CONTRL" in out
        assert "$BASIS" in out
        assert "$DATA" in out

    def test_pdb_to_inp_custom_charge(self, capsys):
        run_convert([MINI_PDB, "--to", "inp", "--charge", "1", "--multiplicity", "3"])
        out, _ = capsys.readouterr()
        assert "ICHARG=1" in out
        assert "MULT=3" in out

    def test_output_to_file_xyz(self, tmp_path, capsys):
        out_path = str(tmp_path / "out.xyz")
        run_convert([MINI_PDB, "--to", "xyz", "-o", out_path])
        assert Path(out_path).exists()
        content = Path(out_path).read_text()
        assert "5" in content.splitlines()[0]

    def test_output_to_file_mol2(self, tmp_path, capsys):
        out_path = str(tmp_path / "out.mol2")
        run_convert([MINI_SDF, "--to", "mol2", "-o", out_path])
        assert Path(out_path).exists()
        content = Path(out_path).read_text()
        assert "@<TRIPOS>MOLECULE" in content


# ------------------------------------------------------------------ #
#  Batch mode tests                                                    #
# ------------------------------------------------------------------ #

class TestBatchMode:
    """Test directory input (batch conversion)."""

    def test_batch_converts_directory(self, tmp_path, capsys):
        """Copy a couple of files into a temp dir and batch-convert."""
        # Setup: create a source dir with one .pdb
        src = tmp_path / "src"
        src.mkdir()
        shutil.copy(MINI_PDB, str(src / "mini.pdb"))

        out = tmp_path / "out"
        run_convert([str(src), "--to", "xyz", "-o", str(out)])

        _, err = capsys.readouterr()
        assert "[OK]" in err
        assert (out / "mini.xyz").exists()

    def test_batch_output_correct_extension(self, tmp_path, capsys):
        src = tmp_path / "src"
        src.mkdir()
        shutil.copy(MINI_SDF, str(src / "mol.sdf"))

        out = tmp_path / "out"
        run_convert([str(src), "--to", "pdb", "-o", str(out)])

        assert (out / "mol.pdb").exists()

    def test_batch_no_files_exits(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(SystemExit):
            run_convert([str(empty), "--to", "xyz"])

    def test_batch_summary(self, tmp_path, capsys):
        src = tmp_path / "src"
        src.mkdir()
        shutil.copy(MINI_PDB, str(src / "a.pdb"))
        shutil.copy(MINI_PDB, str(src / "b.pdb"))

        out = tmp_path / "out"
        run_convert([str(src), "--to", "xyz", "-o", str(out)])

        _, err = capsys.readouterr()
        assert "Batch complete" in err
        assert "2 converted" in err

    def test_batch_recursive(self, tmp_path, capsys):
        """--recursive flag should find files in subdirectories."""
        src = tmp_path / "src"
        sub = src / "subdir"
        sub.mkdir(parents=True)
        shutil.copy(MINI_PDB, str(sub / "deep.pdb"))

        out = tmp_path / "out"
        run_convert([str(src), "--to", "xyz", "-o", str(out), "--recursive"])

        _, err = capsys.readouterr()
        assert "[OK]" in err
        assert (out / "deep.xyz").exists()


# ------------------------------------------------------------------ #
#  Validation flag tests                                               #
# ------------------------------------------------------------------ #

class TestValidationFlag:
    """Test --validate and --report CLI flags."""

    def test_validate_prints_report(self, capsys):
        run_convert([MINI_SDF, "--to", "xyz", "--validate"])
        _, err = capsys.readouterr()
        assert "Validation Report" in err

    def test_validate_report_to_file(self, tmp_path, capsys):
        report_path = str(tmp_path / "report.txt")
        run_convert([MINI_SDF, "--to", "xyz", "--validate",
                     "--report", report_path])
        assert Path(report_path).exists()
        content = Path(report_path).read_text()
        # Should contain PASS or FAIL for the molecules
        assert "[PASS]" in content or "[FAIL]" in content


# ------------------------------------------------------------------ #
#  Error paths                                                         #
# ------------------------------------------------------------------ #

class TestErrorPaths:
    """Test that bad inputs produce clean errors."""

    def test_unsupported_extension_exits(self):
        with pytest.raises(SystemExit):
            run_convert(["unknown.abc", "--to", "json"])

    def test_missing_file_exits(self):
        with pytest.raises(SystemExit):
            run_convert(["ghost.pdb", "--to", "json"])
