# molconvert

A CLI-based molecular structure conversion engine that uses **internal coordinates (IC) as its core intermediate representation**. Convert freely between PDB, SDF, and Z-matrix (ZMAT) formats — all routed through a single, lossless JSON-based internal representation.

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

**Supported input formats:** PDB, SDF, ZMAT  
**Supported output formats:** PDB, SDF, ZMAT, JSON (internal representation)

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
PDB  ── pdb_parser  ──►     ┌─────────────────────┐  ──►      PDB
SDF  ── sdf_parser  ──►     │    MoleculeIC        │  ──►      SDF
ZMAT ── zmat_to_json ──►    │  (bond_length,       │  ──►      ZMAT
                            │   bond_angle,        │  ──►      JSON
                            │   dihedral + XYZ)    │
                            └─────────────────────┘
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
| `INPUT` | Input file — auto-detected by extension (`.pdb`, `.sdf`, `.zmat`) |
| `--to FORMAT` | Output format: `json`, `pdb`, `sdf`, `zmat`, `internal` (= `json`) |
| `-o PATH` | Write output to a file instead of printing to screen |
| `--chain ID` | Process one chain only, e.g. `--chain A` (PDB input only) |
| `--include-hetatm` | Also process ligand/HETATM records (PDB input only) |
| `--remove-hydrogens` | Strip hydrogen atoms (SDF input only) |
| `--summary` | Print bond-length / angle / dihedral statistics to stderr |
| `--model N` | Choose MODEL record index (PDB input only, default: 0) |

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
