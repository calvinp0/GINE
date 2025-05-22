"""gaussian_log_parser.py

Batch‑parse Gaussian *.log* files and emit atom‑, bond‑ and molecule‑level
features ready for Chemprop‑style featurisers.

Changes v1.1  ────────────────────────────────────────────────────────────────
• Robust to over‑flow asterisks in lines like:
  "Rotational constants (GHZ):***************  2.68  2.68"  that crash
  vanilla cclib.
• Added `safe_ccread()` wrapper that sanitises such tokens before handing the
  text to cclib’s Gaussian parser.
• No public API change: `parse_gaussian_log()` now internally calls
  `safe_ccread()` instead of `cclib.io.ccread()`.

Dependencies
────────────
• cclib ≥ 1.8.1  (pip install cclib)
• RDKit 2023     (conda install -c conda-forge rdkit)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem

from cclib.parser.gaussianparser import Gaussian as _GParser

###############################################################################
# 0. ─── Safe ccread wrapper ──────────────────────────────────────────────────
###############################################################################

_STAR_RE = re.compile(r"\*{5,}")


def safe_ccread(path: str | Path):
    """Return a parsed cclib *data* object, replacing any numeric over‑flow
    tokens (runs of '*') with `0.0` so that cclib never raises ValueError.
    """

    with open(path, "r", errors="replace") as fh:
        sanitized_text = _STAR_RE.sub(" 0.0 ", fh.read())

    parser = _GParser(StringIO(sanitized_text))
    parser.logger.setLevel(logging.ERROR)
    return parser.parse()

###############################################################################
# 1. ─── Atom‑level helpers ──────────────────────────────────────────────────
###############################################################################


def _extract_atom_features(data) -> Dict[int, Dict[str, float]]:
    """Safely build per‑atom feature dict even if some attributes are missing."""

    # Robust length test
    natom = getattr(data, "natom", len(getattr(data, "atomnos", [])))
    if natom == 0:
        return {}

    mull = getattr(data, "atomcharges", {}).get("mulliken") if hasattr(data, "atomcharges") else None
    apt = getattr(data, "atomcharges", {}).get("apt") if hasattr(data, "atomcharges") else None
    spin = getattr(data, "atomspins", {}).get("mulliken") if hasattr(data, "atomspins") else None

    atomnos = getattr(data, "atomnos", [0] * natom)
    masses = getattr(data, "atommasses", [0.0] * natom)

    feats: Dict[int, Dict[str, float]] = {}
    for i in range(natom):
        feats[i] = {
            "q_mull": float(mull[i]) if mull is not None else 0.0,
            "q_apt": float(apt[i]) if apt is not None else 0.0,
            "spin": float(spin[i]) if spin is not None else 0.0,
            "Z": int(atomnos[i]),
            "mass": float(masses[i]),
        }
    return feats

###############################################################################
# 2. ─── Bond helpers ────────────────────────────────────────────────────────
###############################################################################

_COV_RAD = {  # Å, Pyykkö 2009, single‑bond
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    35: 1.20,
}


def _build_connectivity(data, mol: rdchem.Mol | None = None) -> Iterable[Tuple[int, int]]:
    if mol is not None:
        for b in mol.GetBonds():
            yield tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        return

    coords = data.atomcoords[-1]
    for i in range(data.natom):
        for j in range(i + 1, data.natom):
            rij = np.linalg.norm(coords[i] - coords[j])
            rcut = 1.2 * (_COV_RAD.get(int(data.atomnos[i]), 0.77) + _COV_RAD.get(int(data.atomnos[j]), 0.77))
            if rij <= rcut:
                yield (i, j)


def _extract_bond_features(data, bonds: Iterable[Tuple[int, int]]) -> Dict[str, Dict[str, float]]:
    coords = data.atomcoords[-1]
    wbo = getattr(data, "wiberg_lowdin_indices", None)
    kphi_dict: Dict[Tuple[int, int], float] = {}

    if hasattr(data, "hessian") and hasattr(data, "internalcoord_labels"):
        for label, atoms, k_au in zip(
            data.internalcoord_labels, data.internalcoord_atoms, np.diag(data.hessian)
        ):
            if label.startswith("D"):
                j, k = atoms[1:3]
                kphi_dict[tuple(sorted((j, k)))] = float(627.509 * k_au)

    feats: Dict[str, Dict[str, float]] = {}
    for i, j in bonds:
        key = f"{i}-{j}"
        dij = float(np.linalg.norm(coords[i] - coords[j]))
        feats[key] = {
            "length": dij,
            "wbo": float(wbo[i][j]) if wbo is not None else 0.0,
            "k_phi": kphi_dict.get((i, j), 0.0),
        }
    return feats

###############################################################################
# 3. ─── Molecule‑level helpers ──────────────────────────────────────────────
###############################################################################


def _extract_mol_features(data) -> Dict[str, Any]:
    mo_energy = np.array(data.moenergies[0])
    homo_idx = data.homos[0]
    homo = float(mo_energy[homo_idx])
    lumo = float(mo_energy[homo_idx + 1])
    dip_vec = np.asarray(data.moments[1]) if hasattr(data, "moments") else np.zeros(3)

    return {
        "E_scf": float(data.scfenergies[-1]) if hasattr(data, "scfenergies") else 0.0,
        "dipole_mag": float(np.linalg.norm(dip_vec)),
        "homo": homo,
        "lumo": lumo,
        "gap": lumo - homo,
    }

###############################################################################
# 4. ─── Top‑level parse function ────────────────────────────────────────────
###############################################################################

def parse_gaussian_log(
    log_path: Path,
    smiles: str | None = None,
    sdf_dir: Path | None = None,
):
    """Return (atom_dict, bond_dict, mol_dict) for *log_path*."""

    data = safe_ccread(log_path)
    if not hasattr(data, "atomcoords"):
        raise ValueError(f"{log_path} contains no Cartesian coordinates – skipping.")


    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is not None:
        mol = Chem.AddHs(mol)
        conf = mol.GetConformer()
        for i, coord in enumerate(data.atomcoords[-1]):
            conf.SetAtomPosition(i, coord.tolist())

    atoms = _extract_atom_features(data)
    bonds_iter = _build_connectivity(data, mol)
    bonds = _extract_bond_features(data, bonds_iter)
    mol_feats = _extract_mol_features(data)

    if sdf_dir is not None and mol is not None:
        sdf_dir.mkdir(parents=True, exist_ok=True)
        Chem.MolToMolFile(mol, str(sdf_dir / (log_path.stem + ".sdf")))

    return atoms, bonds, mol_feats

###############################################################################
# 5. ─── CLI ─────────────────────────────────────────────────────────────────
###############################################################################


def _cli():
    import argparse

    p = argparse.ArgumentParser(description="Parse Gaussian *.log files to JSON feature dicts.")
    p.add_argument("log_root", type=Path, help="Directory or single .log file")
    p.add_argument("pattern", nargs="?", default="*.log", help="glob pattern")
    p.add_argument("--sdf-dir", type=Path, help="dump final geometries to this dir")
    p.add_argument("--smiles-csv", type=Path, help="Optional CSV file: file,smiles")
    args = p.parse_args()

    smiles_map = {}
    if args.smiles_csv and args.smiles_csv.exists():
        import csv

        with open(args.smiles_csv) as fh:
            for row in csv.DictReader(fh):
                smiles_map[row["file"]] = row["smiles"]

    log_files = (
        [args.log_root] if args.log_root.is_file() else list(args.log_root.rglob(args.pattern))
    )

    for log in log_files:
        print(f"Parsing {log.relative_to(args.log_root.parent)} …")
        smi = smiles_map.get(log.name)
        atoms, bonds, mol_feats = parse_gaussian_log(log, smi, args.sdf_dir)

        base = log.with_suffix("")
        base.with_suffix("_atoms.json").write_text(json.dumps(atoms, indent=2))
        base.with_suffix("_bonds.json").write_text(json.dumps(bonds, indent=2))
        base.with_suffix("_mol.json").write_text(json.dumps(mol_feats, indent=2))

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli()
    else:
        print(__doc__)
