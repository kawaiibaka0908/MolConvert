"""
Tests for Gaussian and GAMESS input file builders (Phase 4).
"""

from pathlib import Path
import pytest

from molconvert.parsers.pdb_parser import parse_pdb
from molconvert.builders.reconstruct import reconstruct
from molconvert.builders.to_gaussian import molecule_to_gaussian, save_gaussian
from molconvert.builders.to_gamess import molecule_to_gamess, save_gamess
from molconvert.core.internal_coords import MoleculeIC, AtomIC

MINI_PDB = str(Path(__file__).parent / "data" / "mini.pdb")


@pytest.fixture
def mini_mol():
    """Parse and reconstruct mini.pdb for builder tests."""
    mols = parse_pdb(MINI_PDB)
    return reconstruct(mols[0])


# ------------------------------------------------------------------ #
#  Gaussian tests                                                      #
# ------------------------------------------------------------------ #

class TestGaussianBuilder:

    def test_gaussian_has_route_section(self, mini_mol):
        output = molecule_to_gaussian(mini_mol)
        assert "# HF/6-31G(d) opt" in output

    def test_gaussian_has_title(self, mini_mol):
        output = molecule_to_gaussian(mini_mol)
        assert mini_mol.name in output

    def test_gaussian_has_charge_multiplicity(self, mini_mol):
        output = molecule_to_gaussian(mini_mol)
        lines = output.split("\n")
        # Find a line that is exactly "0 1"
        assert any(line.strip() == "0 1" for line in lines)

    def test_gaussian_coordinate_lines(self, mini_mol):
        output = molecule_to_gaussian(mini_mol)
        lines = output.split("\n")
        # Coordinate lines come after the "0 1" line, each starts with an element symbol
        # mini.pdb has 5 atoms: N, C, C, O, N
        charge_mult_idx = next(
            i for i, line in enumerate(lines) if line.strip() == "0 1"
        )
        coord_lines = []
        for line in lines[charge_mult_idx + 1:]:
            stripped = line.strip()
            if stripped == "":
                break
            coord_lines.append(stripped)
        assert len(coord_lines) == 5

    def test_gaussian_ends_with_blank_line(self, mini_mol):
        output = molecule_to_gaussian(mini_mol)
        assert output.endswith("\n\n")

    def test_gaussian_custom_route(self, mini_mol):
        output = molecule_to_gaussian(mini_mol, route="# B3LYP/6-311G(d,p) opt freq")
        assert "# B3LYP/6-311G(d,p) opt freq" in output

    def test_gaussian_custom_charge(self, mini_mol):
        output = molecule_to_gaussian(mini_mol, charge=1, multiplicity=2)
        lines = output.split("\n")
        assert any(line.strip() == "1 2" for line in lines)

    def test_gaussian_custom_title(self, mini_mol):
        output = molecule_to_gaussian(mini_mol, title="my title")
        assert "my title" in output

    def test_save_gaussian_creates_file(self, mini_mol, tmp_path):
        out_file = tmp_path / "test.gjf"
        save_gaussian(mini_mol, str(out_file))
        assert out_file.exists()
        content = out_file.read_text()
        assert "# HF/6-31G(d) opt" in content

    def test_gaussian_raises_without_positions(self):
        mol = MoleculeIC(name="empty", source_fmt="test", atoms=[
            AtomIC(
                atom_serial=1, atom_name="X", residue_name="UNK",
                chain_id="A", residue_seq=1, element="C",
            ),
        ])
        with pytest.raises(ValueError):
            molecule_to_gaussian(mol)


# ------------------------------------------------------------------ #
#  GAMESS tests                                                        #
# ------------------------------------------------------------------ #

class TestGamessBuilder:

    def test_gamess_has_contrl_group(self, mini_mol):
        output = molecule_to_gamess(mini_mol)
        assert "$CONTRL" in output

    def test_gamess_has_basis_group(self, mini_mol):
        output = molecule_to_gamess(mini_mol)
        assert "$BASIS" in output

    def test_gamess_has_data_group(self, mini_mol):
        output = molecule_to_gamess(mini_mol)
        assert "$DATA" in output

    def test_gamess_has_c1_symmetry(self, mini_mol):
        output = molecule_to_gamess(mini_mol)
        lines = output.split("\n")
        assert any(line.strip() == "C1" for line in lines)

    def test_gamess_atomic_numbers_correct(self, mini_mol):
        output = molecule_to_gamess(mini_mol)
        lines = output.split("\n")
        # Find atom lines: they are between the blank line after C1 and the final $END
        data_start = next(
            i for i, line in enumerate(lines) if line.strip() == "$DATA"
        )
        # After $DATA: title, C1, blank, then atom lines until $END
        atom_lines = []
        for line in lines[data_start + 4:]:  # skip $DATA, title, C1, blank
            if "$END" in line:
                break
            if line.strip():
                atom_lines.append(line)

        # Expected elements and atomic numbers: N(7), C(6), C(6), O(8), N(7)
        expected = [7.0, 6.0, 6.0, 8.0, 7.0]
        for atom_line, exp_num in zip(atom_lines, expected):
            parts = atom_line.split()
            # Format: element  atomic_num  x  y  z
            assert float(parts[1]) == exp_num

    def test_gamess_coordinate_lines(self, mini_mol):
        output = molecule_to_gamess(mini_mol)
        lines = output.split("\n")
        data_start = next(
            i for i, line in enumerate(lines) if line.strip() == "$DATA"
        )
        atom_lines = []
        for line in lines[data_start + 4:]:
            if "$END" in line:
                break
            if line.strip():
                atom_lines.append(line)
        assert len(atom_lines) == 5

    def test_gamess_dollar_in_column_2(self, mini_mol):
        output = molecule_to_gamess(mini_mol)
        lines = output.split("\n")
        for line in lines:
            if "$" in line:
                # The $ should be at index 1 (column 2), meaning line[0] is a space
                dollar_idx = line.index("$")
                assert dollar_idx == 1, (
                    f"Expected $ at column 2 but found at column {dollar_idx + 1} "
                    f"in line: {line!r}"
                )

    def test_gamess_custom_charge(self, mini_mol):
        output = molecule_to_gamess(mini_mol, charge=1, multiplicity=3)
        assert "ICHARG=1" in output
        assert "MULT=3" in output

    def test_save_gamess_creates_file(self, mini_mol, tmp_path):
        out_file = tmp_path / "test.inp"
        save_gamess(mini_mol, str(out_file))
        assert out_file.exists()
        content = out_file.read_text()
        assert "$CONTRL" in content

    def test_gamess_raises_without_positions(self):
        mol = MoleculeIC(name="empty", source_fmt="test", atoms=[
            AtomIC(
                atom_serial=1, atom_name="X", residue_name="UNK",
                chain_id="A", residue_seq=1, element="C",
            ),
        ])
        with pytest.raises(ValueError):
            molecule_to_gamess(mol)

    def test_gamess_has_end(self, mini_mol):
        output = molecule_to_gamess(mini_mol)
        assert output.rstrip().endswith("$END")
        # Verify the last line has $ in column 2
        last_line = output.rstrip().split("\n")[-1]
        assert last_line == " $END"
