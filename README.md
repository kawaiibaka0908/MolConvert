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

