"""Tests for analysis/rmsd.py — kabsch_align, kabsch_rmsd, rmsd, rmsd_molecules."""

import copy
from pathlib import Path

import numpy as np
import pytest

from molconvert.parsers.pdb_parser import parse_pdb
from molconvert.builders.reconstruct import reconstruct
from molconvert.analysis import (
    rmsd,
    rmsd_molecules,
    per_atom_deviation,
    ic_summary,
    ICSummary,
)
from molconvert.analysis.rmsd import kabsch_align, kabsch_superpose, kabsch_rmsd

MINI_PDB = str(Path(__file__).parent / "data" / "mini.pdb")


@pytest.fixture
def mol_orig():
    return parse_pdb(MINI_PDB)[0]


@pytest.fixture
def mol_recon(mol_orig):
    return reconstruct(mol_orig)


# ------------------------------------------------------------------ #
#  kabsch_align — input validation                                     #
# ------------------------------------------------------------------ #

def test_kabsch_align_raises_on_non_3d():
    bad = np.ones((5, 2))
    good = np.ones((5, 3))
    with pytest.raises(ValueError, match="N, 3"):
        kabsch_align(good, bad)
    with pytest.raises(ValueError, match="N, 3"):
        kabsch_align(bad, good)


def test_kabsch_align_raises_on_shape_mismatch():
    c1 = np.ones((5, 3))
    c2 = np.ones((6, 3))
    with pytest.raises(ValueError, match="Shape mismatch"):
        kabsch_align(c1, c2)


def test_kabsch_align_raises_when_fewer_than_3_points():
    c1 = np.ones((2, 3))
    c2 = np.ones((2, 3))
    with pytest.raises(ValueError, match="3"):
        kabsch_align(c1, c2)


def test_kabsch_align_accepts_list_input():
    c1 = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    c2 = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    R, t, r = kabsch_align(c1, c2)
    assert r == pytest.approx(0.0, abs=1e-10)


# ------------------------------------------------------------------ #
#  kabsch_align — return types and shapes                              #
# ------------------------------------------------------------------ #

def test_kabsch_align_returns_tuple_of_three():
    c = np.random.default_rng(0).standard_normal((10, 3))
    result = kabsch_align(c, c)
    assert len(result) == 3


def test_kabsch_align_rotation_is_3x3():
    c = np.random.default_rng(1).standard_normal((10, 3))
    R, t, _ = kabsch_align(c, c)
    assert R.shape == (3, 3)


def test_kabsch_align_translation_is_3():
    c = np.random.default_rng(2).standard_normal((10, 3))
    R, t, _ = kabsch_align(c, c)
    assert t.shape == (3,)


def test_kabsch_align_rmsd_is_scalar():
    c = np.random.default_rng(3).standard_normal((10, 3))
    _, _, r = kabsch_align(c, c)
    assert isinstance(r, float)


# ------------------------------------------------------------------ #
#  kabsch_align — Test 1: identity (coords1 == coords2 → RMSD ≈ 0)   #
# ------------------------------------------------------------------ #

def test_kabsch_identity_rmsd_zero():
    rng = np.random.default_rng(42)
    coords = rng.standard_normal((20, 3))
    R, t, r = kabsch_align(coords, coords)
    assert r == pytest.approx(0.0, abs=1e-10)


def test_kabsch_identity_rotation_is_identity_matrix():
    rng = np.random.default_rng(42)
    coords = rng.standard_normal((20, 3))
    R, t, _ = kabsch_align(coords, coords)
    assert np.allclose(R, np.eye(3), atol=1e-10)


# ------------------------------------------------------------------ #
#  kabsch_align — Test 2: pure translation (RMSD ≈ 0 after alignment) #
# ------------------------------------------------------------------ #

def test_kabsch_translated_structure_rmsd_zero():
    rng = np.random.default_rng(7)
    coords1 = rng.standard_normal((15, 3))
    shift = np.array([3.5, -2.1, 7.8])
    coords2 = coords1 + shift
    R, t, r = kabsch_align(coords1, coords2)
    assert r == pytest.approx(0.0, abs=1e-9)


def test_kabsch_translated_structure_rotation_is_identity():
    rng = np.random.default_rng(8)
    coords1 = rng.standard_normal((15, 3))
    coords2 = coords1 + np.array([1.0, 2.0, 3.0])
    R, t, _ = kabsch_align(coords1, coords2)
    assert np.allclose(R, np.eye(3), atol=1e-9)


def test_kabsch_translation_vector_recovers_shift():
    rng = np.random.default_rng(9)
    coords1 = rng.standard_normal((12, 3))
    shift = np.array([5.0, -3.0, 2.0])
    coords2 = coords1 + shift
    R, t, _ = kabsch_align(coords1, coords2)
    # Applying R and t to coords2 should reproduce coords1
    aligned = coords2 @ R + t
    assert np.allclose(aligned, coords1, atol=1e-9)


# ------------------------------------------------------------------ #
#  kabsch_align — Test 3: pure rotation (RMSD ≈ 0 after alignment)    #
# ------------------------------------------------------------------ #

def _rotation_matrix_x(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([
        [1, 0,          0         ],
        [0, np.cos(a), -np.sin(a) ],
        [0, np.sin(a),  np.cos(a) ],
    ])


def _rotation_matrix_z(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([
        [ np.cos(a), -np.sin(a), 0],
        [ np.sin(a),  np.cos(a), 0],
        [ 0,          0,         1],
    ])


def test_kabsch_rotated_structure_rmsd_zero():
    rng = np.random.default_rng(13)
    coords1 = rng.standard_normal((20, 3))
    Rtrue = _rotation_matrix_x(37.0) @ _rotation_matrix_z(55.0)
    coords2 = coords1 @ Rtrue.T      # rotate coords2 away from coords1
    R, t, r = kabsch_align(coords1, coords2)
    assert r == pytest.approx(0.0, abs=1e-9)


def test_kabsch_recovered_rotation_matches_true():
    rng = np.random.default_rng(14)
    coords1 = rng.standard_normal((20, 3))
    Rtrue = _rotation_matrix_z(90.0)
    coords2 = coords1 @ Rtrue.T
    R, t, _ = kabsch_align(coords1, coords2)
    # R should equal Rtrue (the rotation that maps coords2 back to coords1)
    assert np.allclose(R, Rtrue, atol=1e-9)


def test_kabsch_aligned_coords_match_reference():
    rng = np.random.default_rng(15)
    coords1 = rng.standard_normal((25, 3))
    Rtrue = _rotation_matrix_x(120.0)
    coords2 = coords1 @ Rtrue.T
    R, t, _ = kabsch_align(coords1, coords2)
    aligned = coords2 @ R + t
    assert np.allclose(aligned, coords1, atol=1e-9)


# ------------------------------------------------------------------ #
#  kabsch_align — Test 4: noise (RMSD small but non-zero)             #
# ------------------------------------------------------------------ #

def test_kabsch_noisy_structure_rmsd_small_nonzero():
    rng = np.random.default_rng(20)
    coords1 = rng.standard_normal((50, 3))
    noise = rng.standard_normal((50, 3)) * 0.05   # 0.05 Å noise
    coords2 = coords1 + noise
    R, t, r = kabsch_align(coords1, coords2)
    assert r > 0.0
    assert r < 0.1   # well below naive RMSD, but non-zero


def test_kabsch_noise_rmsd_less_than_naive():
    rng = np.random.default_rng(21)
    coords1 = rng.standard_normal((50, 3))
    noise = rng.standard_normal((50, 3)) * 0.1
    # Add a translation so naive RMSD is inflated
    coords2 = coords1 + noise + np.array([5.0, 5.0, 5.0])
    naive = rmsd(coords1, coords2)
    _, _, r = kabsch_align(coords1, coords2)
    assert r < naive


# ------------------------------------------------------------------ #
#  kabsch_align — Test 5: reflection correction                        #
# ------------------------------------------------------------------ #

def test_kabsch_reflection_det_is_plus_one():
    """Rotation matrix from kabsch_align must always be a proper rotation."""
    rng = np.random.default_rng(30)
    # Construct a reflected (improper) version of coords1
    coords1 = rng.standard_normal((20, 3))
    # Invert x-axis — this is a reflection, not a rotation
    coords2 = coords1 * np.array([-1, 1, 1])
    R, t, r = kabsch_align(coords1, coords2)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


def test_kabsch_reflection_no_negative_det():
    rng = np.random.default_rng(31)
    for seed in range(10):
        coords1 = np.random.default_rng(seed).standard_normal((15, 3))
        coords2 = coords1 * np.array([-1, 1, 1])   # reflection
        R, _, _ = kabsch_align(coords1, coords2)
        assert np.linalg.det(R) > 0, f"Negative determinant on seed {seed}"


# ------------------------------------------------------------------ #
#  kabsch_align — rotation matrix properties                           #
# ------------------------------------------------------------------ #

def test_kabsch_rotation_is_orthogonal():
    rng = np.random.default_rng(40)
    c1 = rng.standard_normal((20, 3))
    c2 = rng.standard_normal((20, 3))
    R, _, _ = kabsch_align(c1, c2)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_kabsch_determinant_is_one():
    rng = np.random.default_rng(41)
    c1 = rng.standard_normal((20, 3))
    c2 = rng.standard_normal((20, 3))
    R, _, _ = kabsch_align(c1, c2)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


def test_kabsch_rmsd_symmetric():
    """RMSD(A, B) should equal RMSD(B, A)."""
    rng = np.random.default_rng(50)
    c1 = rng.standard_normal((20, 3))
    c2 = rng.standard_normal((20, 3))
    _, _, r12 = kabsch_align(c1, c2)
    _, _, r21 = kabsch_align(c2, c1)
    assert r12 == pytest.approx(r21, abs=1e-10)


def test_kabsch_rmsd_non_negative():
    rng = np.random.default_rng(51)
    for _ in range(20):
        c1 = rng.standard_normal((10, 3))
        c2 = rng.standard_normal((10, 3))
        _, _, r = kabsch_align(c1, c2)
        assert r >= 0.0


# ------------------------------------------------------------------ #
#  kabsch_align — debug flag (smoke test, no crash)                   #
# ------------------------------------------------------------------ #

def test_kabsch_debug_mode_runs_without_error(capsys):
    rng = np.random.default_rng(60)
    c1 = rng.standard_normal((10, 3))
    c2 = rng.standard_normal((10, 3))
    kabsch_align(c1, c2, debug=True)
    out = capsys.readouterr().out
    assert "centroid" in out
    assert "RMSD before" in out
    assert "RMSD after" in out
    assert "det(R)" in out


def test_kabsch_debug_reflection_fix_printed(capsys):
    rng = np.random.default_rng(61)
    coords1 = rng.standard_normal((20, 3))
    coords2 = coords1 * np.array([-1, 1, 1])   # force reflection case
    kabsch_align(coords1, coords2, debug=True)
    out = capsys.readouterr().out
    assert "reflection fix" in out


# ------------------------------------------------------------------ #
#  kabsch_superpose and kabsch_rmsd convenience wrappers               #
# ------------------------------------------------------------------ #

def test_kabsch_superpose_shape():
    rng = np.random.default_rng(70)
    c1 = rng.standard_normal((15, 3))
    c2 = rng.standard_normal((15, 3))
    aligned = kabsch_superpose(c1, c2)
    assert aligned.shape == c2.shape


def test_kabsch_superpose_rmsd_matches_kabsch_rmsd():
    rng = np.random.default_rng(71)
    c1 = rng.standard_normal((15, 3))
    c2 = rng.standard_normal((15, 3))
    aligned = kabsch_superpose(c1, c2)
    r_direct = kabsch_rmsd(c1, c2)
    r_manual = float(np.sqrt(np.mean(np.sum((c1 - aligned) ** 2, axis=1))))
    assert r_direct == pytest.approx(r_manual, abs=1e-10)


# ------------------------------------------------------------------ #
#  rmsd (no superposition)                                             #
# ------------------------------------------------------------------ #

def test_rmsd_identical_arrays_is_zero():
    coords = np.random.default_rng(0).standard_normal((10, 3))
    assert rmsd(coords, coords) == pytest.approx(0.0)


def test_rmsd_known_value():
    c1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    c2 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert rmsd(c1, c2) == pytest.approx(1.0)


def test_rmsd_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        rmsd(np.zeros((5, 3)), np.zeros((6, 3)))


def test_rmsd_non_negative():
    rng = np.random.default_rng(1)
    for _ in range(20):
        c1 = rng.standard_normal((8, 3))
        c2 = rng.standard_normal((8, 3))
        assert rmsd(c1, c2) >= 0.0


# ------------------------------------------------------------------ #
#  rmsd_molecules                                                      #
# ------------------------------------------------------------------ #

def test_rmsd_molecules_self_is_zero(mol_orig):
    assert rmsd_molecules(mol_orig, mol_orig) == pytest.approx(0.0)


def test_rmsd_molecules_roundtrip_realistic(mol_orig, mol_recon):
    r = rmsd_molecules(mol_orig, mol_recon)
    # Reconstruction rounds angles to 0.01° — expect small but non-zero RMSD
    assert r < 0.5
    assert r > 0.0


def test_rmsd_molecules_with_filter_realistic(mol_orig, mol_recon):
    r = rmsd_molecules(mol_orig, mol_recon, atom_filter=["N", "CA", "C"])
    assert r < 0.5


def test_rmsd_molecules_filter_no_matches_returns_zero(mol_orig):
    r = rmsd_molecules(mol_orig, mol_orig, atom_filter=["ZZZ"])
    assert r == pytest.approx(0.0)


def test_rmsd_molecules_count_mismatch_raises(mol_orig):
    short_mol = copy.deepcopy(mol_orig)
    short_mol.atoms = short_mol.atoms[:3]
    with pytest.raises(ValueError, match="Atom count mismatch"):
        rmsd_molecules(mol_orig, short_mol)


def test_rmsd_molecules_missing_position_raises(mol_orig):
    bad = copy.deepcopy(mol_orig)
    bad.atoms[0].cart_x = None
    with pytest.raises(ValueError):
        rmsd_molecules(bad, mol_orig)


# ------------------------------------------------------------------ #
#  per_atom_deviation                                                  #
# ------------------------------------------------------------------ #

def test_per_atom_deviation_self_all_zero(mol_orig):
    devs = per_atom_deviation(mol_orig, mol_orig)
    for rec in devs:
        assert rec["deviation"] == pytest.approx(0.0, abs=1e-10)


def test_per_atom_deviation_roundtrip_bounded(mol_orig, mol_recon):
    devs = per_atom_deviation(mol_orig, mol_recon)
    for rec in devs:
        assert rec["deviation"] < 1.0


def test_per_atom_deviation_returns_correct_count(mol_orig, mol_recon):
    devs = per_atom_deviation(mol_orig, mol_recon)
    assert len(devs) == len(mol_orig.atoms)


def test_per_atom_deviation_has_expected_keys(mol_orig, mol_recon):
    rec = per_atom_deviation(mol_orig, mol_recon)[0]
    for key in ("atom_serial", "atom_name", "residue_name", "residue_seq",
                "chain_id", "deviation"):
        assert key in rec


def test_per_atom_deviation_with_filter(mol_orig, mol_recon):
    devs = per_atom_deviation(mol_orig, mol_recon, atom_filter=["CA"])
    assert len(devs) == 1
    assert devs[0]["atom_name"] == "CA"


def test_per_atom_deviation_count_mismatch_raises(mol_orig):
    short = copy.deepcopy(mol_orig)
    short.atoms = short.atoms[:3]
    with pytest.raises(ValueError, match="Atom count mismatch"):
        per_atom_deviation(mol_orig, short)


# ------------------------------------------------------------------ #
#  ic_summary                                                          #
# ------------------------------------------------------------------ #

def test_ic_summary_returns_icsummary(mol_orig):
    assert isinstance(ic_summary(mol_orig), ICSummary)


def test_ic_summary_atom_count(mol_orig):
    assert ic_summary(mol_orig).n_atoms == 5


def test_ic_summary_anchor_count(mol_orig):
    assert ic_summary(mol_orig).n_anchors == 1


def test_ic_summary_bond_length_range(mol_orig):
    s = ic_summary(mol_orig)
    assert s.bond_length_min <= s.bond_length_mean <= s.bond_length_max
    assert s.bond_length_std >= 0.0


def test_ic_summary_bond_angle_range(mol_orig):
    s = ic_summary(mol_orig)
    assert 0.0 <= s.bond_angle_min
    assert s.bond_angle_max <= 180.0


def test_ic_summary_str_contains_keywords(mol_orig):
    text = str(ic_summary(mol_orig))
    assert "Bond lengths" in text
    assert "Bond angles" in text
    assert "Dihedrals" in text


def test_ic_summary_empty_molecule():
    from molconvert.core.internal_coords import MoleculeIC
    empty = MoleculeIC(name="e", source_fmt="pdb")
    s = ic_summary(empty)
    assert s.n_atoms == 0
    assert s.n_anchors == 0
