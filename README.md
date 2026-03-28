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

