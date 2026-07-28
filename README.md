# molconvert

A CLI-based molecular structure conversion engine that uses **internal coordinates (IC) as its core intermediate representation**. Convert freely between PDB, SDF, Z-matrix (ZMAT), quantum chemistry formats (Gaussian, GAMESS), MOL2, XYZ, and GRO formats — all routed through a single, lossless JSON-based internal representation. Includes chemical validation and batch conversion.

---

## Table of Contents

- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [CLI Usage](#cli-usage)
  - [convert](#convert)
  - [rmsd](#rmsd)
- [Quick Reference Table](#quick-reference-table)
- [Python API](#python-api)
- [Project Structure](#project-structure)
- [Running the Tests](#running-the-tests)
- [Design Principles](#design-principles)
- [Known Limitations](#known-limitations)

---

## What It Does

molconvert reads molecular structure files, converts them to **internal coordinates** (bond lengths, bond angles, dihedral angles), and can write those internal coordinates back out in any supported format.

**Supported input formats:** PDB, SDF, ZMAT, Gaussian .log/.out, GAMESS .out, MOL2  
**Supported output formats:** PDB, SDF, ZMAT, JSON, MOL2, XYZ, GRO, Gaussian .gjf, GAMESS .inp

Use cases:
- Convert a protein PDB file to a Z-matrix for analysis or ML input
- Convert a small-molecule SDF file to PDB for visualisation
- Inspect the internal coordinate representation of any structure as JSON
- Verify round-trip reconstruction accuracy with RMSD comparison

---

## How It Works

Every conversion goes through a single internal representation — **MoleculeIC** — stored in memory and serialisable as JSON.

```
Input formats               Core IR (JSON/MoleculeIC)          Output formats
─────────────               ─────────────────────────          ──────────────
PDB  ── pdb_parser     ──►  ┌─────────────────────┐  ──►      PDB
SDF  ── sdf_parser     ──►  │    MoleculeIC        │  ──►      SDF
ZMAT ── zmat_to_json   ──►  │  (bond_length,       │  ──►      ZMAT, JSON
LOG  ── gaussian_parser ─►  │   bond_angle,        │  ──►      MOL2, XYZ, GRO
OUT  ── gamess_parser  ──►  │   dihedral + XYZ)    │  ──►      Gaussian .gjf
MOL2 ── mol2_parser    ──►  └─────────────────────┘  ──►      GAMESS .inp
```

**Internal coordinate scheme:**

| Atom index | bond_length | bond_angle | dihedral | Cartesian (XYZ) |
|---|---|---|---|---|
| 1 (anchor) | — | — | — | stored |
| 2 (anchor) | yes | — | — | stored |
| 3 (anchor) | yes | yes | — | stored |
| 4+ | yes | yes | yes | reconstructed via NeRF |

Reconstruction from IC back to Cartesian coordinates uses the **NeRF algorithm** (Natural Extension Reference Frame), which gives a round-trip RMSD of < 0.0001 Å.

**Z-matrix (ZMAT)** is the *external* representation of internal coordinates — it is not used internally. JSON remains the core IR at all times.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9 or newer | Check with `python --version` |
| pip | any recent | Comes with Python |
| numpy | auto-installed | Array math |
| biopython | auto-installed | PDB parsing |
| rdkit | auto-installed | SDF parsing |

No manual installation of numpy, biopython, or rdkit is needed — `pip install -e .` handles all of them.

---

## Installation

**Step 1** — Copy the `molconvert/` folder to the target machine. 

**Step 2** — Open a terminal, navigate into the folder, and install:


```bash
cd molconvert
pip install -e .
```

**Step 3** — Verify the install:

```bash
convert --help
rmsd --help
```

After installation, two commands are available anywhere in the terminal: `convert` and `rmsd`.

---

## CLI Usage

### convert

Converts a molecular structure file between formats.

```
convert INPUT [--to FORMAT] [-o OUTPUT] [options]
```

**Arguments:**

| Argument | Description |
|---|---|
| `INPUT` | Input file/dir — auto-detected by extension (`.pdb`, `.sdf`, `.zmat`, `.log`, `.out`, `.mol2`) |
| `--to FORMAT` | Output format: `json`, `pdb`, `sdf`, `zmat`, `internal`, `mol2`, `xyz`, `gro`, `gjf`, `inp` |
| `-o PATH` | Write output to a file instead of printing to screen |
| `--chain ID` | Process one chain only, e.g. `--chain A` (PDB input only) |
| `--include-hetatm` | Also process ligand/HETATM records (PDB input only) |
| `--remove-hydrogens` | Strip hydrogen atoms (SDF input only) |
| `--summary` | Print bond-length / angle / dihedral statistics to stderr |
| `--model N` | Choose MODEL record index (PDB input only, default: 0) |
| `--input-format` | Explicitly specify input format for `.out` files (`gaussian`, `gamess`) |
| `--step` | Geometry step to extract (`last`, `all`, `N`) (gaussian/gamess) |
| `--charge N` | Molecular charge (gjf/inp output) |
| `--multiplicity N` | Spin multiplicity (gjf/inp output) |
| `--route "..."` | Gaussian route section (gjf output) |
| `--validate` | Run chemical validation after conversion |
| `--report PATH` | Write validation report to file |
| `--recursive` | Recurse subdirectories in batch mode |

---

**PDB → JSON (default)**

```bash
convert input.pdb
convert input.pdb --to json
convert input.pdb --to json -o output.json
```

Example JSON output (first 4 atoms):

```json
{
  "name": "input_A",
  "atoms": [
    { "atom_name": "N",  "bond_length": null,  "bond_angle": null,  "dihedral": null,  "bond_to": null, "angle_to": null, "dihedral_to": null },
    { "atom_name": "CA", "bond_length": 1.483, "bond_angle": null,  "dihedral": null,  "bond_to": 1,    "angle_to": null, "dihedral_to": null },
    { "atom_name": "C",  "bond_length": 1.526, "bond_angle": 113.5, "dihedral": null,  "bond_to": 2,    "angle_to": 1,    "dihedral_to": null },
    { "atom_name": "O",  "bond_length": 1.223, "bond_angle": 117.8, "dihedral": -121.2,"bond_to": 3,    "angle_to": 2,    "dihedral_to": 1    }
  ]
}
```

---

**PDB → ZMAT**

```bash
convert input.pdb --to zmat
convert input.pdb --to zmat -o output.zmat
```

Example ZMAT output:

```
ZMAT input_A
# source_fmt pdb
# n_atoms 304
# anchor 1    -8.901000     4.127000    -0.555000
# anchor 2    -8.608000     3.135000    -1.618000
# anchor 3    -7.117000     2.964000    -1.897000
   1  N     ASN  A     1
   2  CA    ASN  A     1     1    1.483200
   3  C     ASN  A     1     2    1.526487     1   113.507460
   4  O     ASN  A     1     3    1.223043     2   117.845536     1  -121.197001
   ...
END
```

---

**PDB → SDF**

```bash
convert input.pdb --to sdf
convert input.pdb --to sdf -o output.sdf
```

Bond connectivity is inferred automatically from interatomic distances using standard covalent radii.

---

**PDB → reconstructed PDB**

```bash
convert input.pdb --to pdb
convert input.pdb --to pdb -o rebuilt.pdb
```

Converts to IC then reconstructs Cartesian positions via NeRF. Round-trip RMSD < 0.0001 Å.

---

**ZMAT → PDB**

```bash
convert molecule.zmat --to pdb
convert molecule.zmat --to pdb -o rebuilt.pdb
```

---

**ZMAT → JSON (internal)**

```bash
convert molecule.zmat --to internal
convert molecule.zmat --to json
```

---

**ZMAT → SDF**

```bash
convert molecule.zmat --to sdf -o output.sdf
```

---

**SDF → PDB**

```bash
convert molecule.sdf --to pdb
convert molecule.sdf --to pdb -o output.pdb
```

Multiple molecule records are written as MODEL / ENDMDL blocks.

---

**SDF → SDF (re-export / normalise)**

```bash
convert molecule.sdf --to sdf
convert molecule.sdf --to sdf --remove-hydrogens -o heavy_atoms.sdf
```

---

**SDF → ZMAT**

```bash
convert molecule.sdf --to zmat -o output.zmat
```

---

**Single chain, with IC statistics:**

```bash
convert input.pdb --chain A --summary
```

```
[input_A]
IC Summary — 304 atoms (1 anchors)
  Bond lengths (Å) : mean=2.493  std=1.361  [1.221, 8.887]
  Bond angles  (°) : mean=79.01  std=35.28  [9.82, 176.98]
  Dihedrals    (°) : mean=0.61   std=103.48 [-179.63, 179.94]
```

---

**Gaussian .log → PDB**

```bash
convert opt.log --to pdb -o out.pdb
```

---

**GAMESS .out → XYZ (with format override)**

```bash
convert opt.out --input-format gamess --to xyz -o out.xyz
```

---

**MOL2 → SDF**

```bash
convert lig.mol2 --to sdf -o lig.sdf
```

---

**Batch conversion with directory input**

```bash
convert data/ --recursive --to pdb
```

---

**Chemical validation**

```bash
convert lig.sdf --to mol2 --validate --report val.json
```

---

**Other formats (mol2, xyz, gro, gjf, inp)**

```bash
convert mol.sdf --to mol2
convert mol.pdb --to xyz
convert mol.pdb --to gro
convert mol.mol2 --to gjf --charge 0 --multiplicity 1 --route "#P B3LYP/6-31G(d) Opt"
convert mol.mol2 --to inp --charge 0 --multiplicity 1
```

---

### rmsd

Computes the Root Mean Square Deviation (RMSD) between two structures, or tests round-trip reconstruction accuracy.

```
rmsd FILE1 [FILE2] [--self] [--filter ATOMS] [--per-atom] [options]
```

**Arguments:**

| Argument | Description |
|---|---|
| `FILE1` | First PDB file |
| `FILE2` | Second PDB file (omit when using `--self`) |
| `--self` | Round-trip test: reconstruct FILE1 from IC, compare to original |
| `--filter NAMES` | Comma-separated atom names to compare, e.g. `CA` or `N,CA,C` |
| `--per-atom` | Print a per-atom deviation table |
| `--chain ID` | Restrict to one chain |
| `--model N` | Choose MODEL record index (default: 0) |

---

**Round-trip accuracy test:**

```bash
rmsd input.pdb --self
```

```
RMSD: 0.0000 Å
  input.pdb (original)
  input.pdb (reconstructed)
```

**Compare two structures:**

```bash
rmsd structure1.pdb structure2.pdb
```

**Alpha-carbon only (backbone trace):**

```bash
rmsd structure1.pdb structure2.pdb --filter CA
```

**Backbone heavy atoms with per-atom table:**

```bash
rmsd input.pdb --self --filter N,CA,C --per-atom
```

```
Chain   Res  ResSeq  Atom   Dev (Å)
------------------------------------
    A   ALA       1     N    0.0000
    A   ALA       1    CA    0.0000
    A   ALA       1     C    0.0000
------------------------------------
Max deviation: 0.0000 Å
```

---

## Quick Reference Table

| Input | `--to` | What happens |
|---|---|---|
| `.pdb` | `json` | PDB → internal coordinates JSON *(default)* |
| `.pdb` | `pdb` | PDB → reconstructed PDB via NeRF |
| `.pdb` | `sdf` | PDB → SDF (bonds inferred) |
| `.pdb` | `zmat` | PDB → Z-matrix |
| `.pdb` | `internal` | alias for `json` |
| `.sdf` | `pdb` | SDF → PDB |
| `.sdf` | `sdf` | SDF → SDF (round-trip / re-export) |
| `.sdf` | `json` | SDF → internal coordinates JSON |
| `.sdf` | `zmat` | SDF → Z-matrix |
| `.sdf` | `internal` | alias for `json` |
| `.zmat` | `pdb` | ZMAT → reconstructed PDB |
| `.zmat` | `sdf` | ZMAT → SDF (bonds inferred) |
| `.zmat` | `json` | ZMAT → internal coordinates JSON |
| `.zmat` | `internal` | alias for `json` |
| `.log` | `json/pdb/sdf/mol2/xyz/gro/gjf/inp` | Gaussian .log to formats |
| `.out` | `json/pdb/sdf/mol2/xyz/gro/gjf/inp` | GAMESS/Gaussian .out to formats (with `--input-format`) |
| `.mol2` | `json/pdb/sdf/xyz/gro/gjf/inp` | MOL2 to formats |
| `.pdb/.sdf/.zmat` | `mol2/xyz/gro/gjf/inp` | Existing formats to new formats |

---

## Python API

All functionality is accessible programmatically:

```python
from molconvert.parsers.pdb_parser import parse_pdb
from molconvert.parsers.sdf_parser import parse_sdf
from molconvert.parsers.gaussian_parser import parse_gaussian
from molconvert.parsers.gamess_parser import parse_gamess
from molconvert.parsers.mol2_parser import parse_mol2
from molconvert.builders.reconstruct import reconstruct, to_pdb, save_pdb
from molconvert.builders.to_sdf import molecule_to_sdf, save_sdf, molecules_to_sdf
from molconvert.builders.to_mol2 import molecule_to_mol2
from molconvert.builders.to_xyz import molecule_to_xyz
from molconvert.builders.to_gro import molecule_to_gro
from molconvert.builders.to_gaussian import molecule_to_gaussian
from molconvert.builders.to_gamess import molecule_to_gamess
from molconvert.converters.json_to_zmat import molecule_to_zmat, save_zmat
from molconvert.converters.zmat_to_json import zmat_to_molecule, load_zmat
from molconvert.analysis import rmsd_molecules, per_atom_deviation, ic_summary
from molconvert.analysis.validation import validate_molecules, format_report

# --- Parse ---
molecules = parse_pdb("input.pdb")                          # list of MoleculeIC
mols      = parse_sdf("compounds.sdf")                      # one per record
mols      = parse_sdf("compounds.sdf", remove_hydrogens=True)
mol       = load_zmat("input.zmat")                         # from ZMAT file
g_mols    = parse_gaussian("opt.log")
gm_mols   = parse_gamess("opt.out")
mol2_mols = parse_mol2("ligand.mol2")

# --- Reconstruct Cartesian coordinates from IC ---
mol_rebuilt = reconstruct(mol)

# --- Export as PDB ---
pdb_text = to_pdb(mol_rebuilt)
save_pdb(mol_rebuilt, "output.pdb")

# --- Export as SDF ---
sdf_text = molecule_to_sdf(mol)                             # single molecule
save_sdf(mol, "output.sdf")
multi_sdf = molecules_to_sdf(mols)                          # multiple molecules
save_sdf_multi(mols, "output_multi.sdf")

# --- Export as ZMAT ---
zmat_text = molecule_to_zmat(mol)
save_zmat(mol, "output.zmat")

# --- Export to other formats ---
mol2_text = molecule_to_mol2(mol)
xyz_text  = molecule_to_xyz(mol)
gro_text  = molecule_to_gro(mol)
gjf_text  = molecule_to_gaussian(mol, charge=0, multiplicity=1)
inp_text  = molecule_to_gamess(mol, charge=0, multiplicity=1)

# --- Export as JSON ---
json_text = mol.to_json()
mol.save_json("output.json")

# --- Load from JSON ---
from molconvert.core.internal_coords import MoleculeIC
import json
mol = MoleculeIC.from_dict(json.load(open("output.json")))

# --- Validation ---
report = validate_molecules([mol])
print(format_report(report))

# --- RMSD comparison ---
r = rmsd_molecules(mol, mol_rebuilt)                        # all atoms
r = rmsd_molecules(mol, mol_rebuilt, atom_filter=["CA"])   # CA only

# --- Per-atom deviation ---
deviations = per_atom_deviation(mol, mol_rebuilt)
for d in deviations:
    print(d["atom_name"], d["deviation"])

# --- IC statistics ---
summary = ic_summary(mol)
print(summary)
```

---

## Project Structure

```
molconvert/
├── setup.py                        # package config and entry points
├── COMMANDS.txt                    # full CLI reference
├── README.md   
├── report.pdf                      # About the project
├── input.pdb                       # example PDB file
│
├── molconvert/                     # main package
│   ├── core/
│   │   ├── internal_coords.py      # AtomIC + MoleculeIC dataclasses (the IR)
│   │   ├── geometry.py             # bond length, angle, dihedral, NeRF math
│   │   └── rdkit_bridge.py         # RDKit integration for advanced properties
│   │
│   ├── parsers/
│   │   ├── pdb_parser.py           # PDB  → MoleculeIC
│   │   ├── sdf_parser.py           # SDF  → MoleculeIC  (uses RDKit)
│   │   ├── gaussian_parser.py      # Gaussian .log/.out → MoleculeIC
│   │   ├── gamess_parser.py        # GAMESS .out → MoleculeIC
│   │   └── mol2_parser.py          # MOL2 → MoleculeIC
│   │
│   ├── builders/
│   │   ├── reconstruct.py          # MoleculeIC → Cartesian → PDB string
│   │   ├── to_sdf.py               # MoleculeIC → SDF V2000 string
│   │   ├── to_mol2.py              # MoleculeIC → MOL2 string
│   │   ├── to_xyz.py               # MoleculeIC → XYZ string
│   │   ├── to_gro.py               # MoleculeIC → GROMACS GRO string
│   │   ├── to_gaussian.py          # MoleculeIC → Gaussian .gjf
│   │   └── to_gamess.py            # MoleculeIC → GAMESS .inp
│   │
│   ├── converters/
│   │   ├── json_to_zmat.py         # MoleculeIC → ZMAT external format
│   │   └── zmat_to_json.py         # ZMAT → MoleculeIC
│   │
│   ├── analysis/
│   │   ├── rmsd.py                 # RMSD, per-atom deviation, IC statistics
│   │   └── validation.py           # Chemical validation heuristics
│   │
│   └── cli/
│       └── main.py                 # run_convert + run_rmsd entry points
│
└── tests/
    ├── data/
    │   ├── mini.pdb                # 5-atom test structure
    │   └── mini.sdf                # ethanol + acetone test molecules
    ├── test_geometry.py            # geometry math tests
    ├── test_parser.py              # PDB parser tests
    ├── test_sdf_parser.py          # SDF parser tests
    ├── test_reconstruct.py         # NeRF reconstruction tests
    ├── test_analysis.py            # RMSD and IC summary tests
    ├── test_cli.py                 # CLI command tests
    ├── test_zmat.py                # ZMAT converter tests
    ├── test_sdf_convert.py         # SDF builder and SDF CLI tests
    ├── test_gaussian_parser.py     # Gaussian parser tests
    ├── test_gamess_parser.py       # GAMESS parser tests
    ├── test_mol2_parser.py         # MOL2 parser tests
    ├── test_xyz_gro_builders.py    # XYZ/GRO builders tests
    ├── test_quantum_builders.py    # Gaussian/GAMESS builders tests
    └── test_mol2_validation.py     # MOL2 builder and validation tests
```

---

## Running the Tests

```bash
cd molconvert
pytest tests/ -v
```

376 tests covering:

- Geometry math (bond length, angle, dihedral, NeRF)
- PDB parser
- SDF parser
- Reconstruction builder (NeRF)
- RMSD analysis and IC statistics
- CLI commands (all input/output combinations)
- ZMAT converter (JSON ↔ ZMAT, round-trip accuracy)
- SDF builder (bond inference, multi-record output)
- Gaussian parser (42 tests)
- GAMESS parser
- MOL2 parser
- XYZ/GRO builders (15 tests)
- Gaussian/GAMESS input builders (21 tests)
- MOL2 builder + validation (22 tests)
- CLI updates (28 tests)
- Batch mode + --recursive

---

## Design Principles

**1. JSON is always the internal IR.**  
Every parser produces a `MoleculeIC` object. Every builder consumes one. No format talks directly to another format.

**2. ZMAT is an external view of IC, not a replacement.**  
The Z-matrix format is used for import/export only. Internally, IC values are always stored as JSON-serialisable Python dataclasses.

**3. Atom ordering is explicit.**  
Each atom carries `bond_to`, `angle_to`, and `dihedral_to` fields — 1-based indices into the molecule's atom list. This makes the ZMAT representation unambiguous and supports non-sequential reference atoms in future extensions.

**4. Round-trip fidelity.**  
Reconstruction from IC back to Cartesian coordinates uses the NeRF algorithm. Round-trip RMSD (PDB → IC → PDB) is < 0.0001 Å, limited only by floating-point arithmetic.

**5. No format lock-in.**  
Adding a new input or output format means writing one parser or builder module. The core IR and CLI wiring are unchanged.

---

## Known Limitations

| Limitation | Detail |
|---|---|
| **SDF bond orders** | SDF output always writes bond type 1 (single). Double/triple bond orders are not preserved through the IC pipeline — the IC representation encodes geometry, not chemistry. |
| **ZMAT non-sequential bonds** | The ZMAT parser reads arbitrary reference indices, but the NeRF reconstructor currently uses sequential atom ordering. Non-sequential ZMAT files (e.g. from Gaussian) will parse but reconstruct incorrectly. |
| **No PDB CONECT records** | PDB output does not write CONECT (bond) records. Bond connectivity is only preserved in SDF output (via distance inference). |
| **Single conformer** | Each `MoleculeIC` stores one conformer. Multi-conformer SDF files are parsed as separate molecules; PDB MODEL records are handled per-model. |
| **Python 3.9+ required** | The codebase uses modern Python type hints. Python 3.8 and earlier are not supported. |
| **No V3000 support** | No V3000 support for molecules with >999 atoms |
| **GAMESS builder limits** | GAMESS builder only supports elements H-Xe (Z=1-54) |
| **Validation thresholds** | Validation collision threshold (0.4 A) may be aggressive for crystal structures |
