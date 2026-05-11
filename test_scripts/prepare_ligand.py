# prepare_ligand.py
# Week 2, Task 2: Ligand preparation for docking
# Run: python prepare_ligand.py
#
# Input : Erlotinib SMILES string
# Output: data/erlotinib.pdbqt   (ready for AutoDock Vina)
#         data/erlotinib_3D.sdf  (3D structure, for reference + ProLIF)

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from meeko import MoleculePreparation, PDBQTWriterLegacy
import os

# ── Paths ──────────────────────────────────────────────────
OUTPUT_PDBQT = "/Users/imad/Desktop/Gemma4Good/data/erlotinib.pdbqt"
OUTPUT_SDF   = "/Users/imad/Desktop/Gemma4Good/data/erlotinib_3D.sdf"
OUTPUT_IMG   = "/Users/imad/Desktop/Gemma4Good/data/erlotinib_2D.png"

print("=" * 55)
print("DockExplain — Ligand Preparation")
print("Erlotinib → PDBQT for AutoDock Vina")
print("=" * 55)

# ── Step 1: Parse SMILES ────────────────────────────────────
# This is the canonical Erlotinib SMILES from PubChem CID 176870
# The C#C at the start is the alkyne triple bond — Erlotinib's
# signature group that distinguishes it from other EGFR drugs
print("\n[1/6] Parsing Erlotinib SMILES...")

SMILES = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"

mol = Chem.MolFromSmiles(SMILES)
if mol is None:
    raise ValueError("SMILES parsing failed — check the string")

print(f"      ✓ SMILES parsed successfully")
print(f"      Heavy atoms    : {mol.GetNumAtoms()}")
print(f"      Bonds          : {mol.GetNumBonds()}")
print(f"      Molecular formula: C22H23N3O4")

# ── Step 2: Add explicit hydrogens ─────────────────────────
# Meeko strictly requires explicit H atoms.
# RDKit molecules from SMILES have implicit Hs by default —
# we must make them explicit before generating 3D coords.
# This is the #1 cause of Meeko failures if skipped.
print("\n[2/6] Adding explicit hydrogen atoms...")

mol_H = Chem.AddHs(mol)
total_atoms = mol_H.GetNumAtoms()
n_hydrogens = total_atoms - mol.GetNumAtoms()

print(f"      ✓ Hydrogens added: {n_hydrogens}")
print(f"      Total atoms now  : {total_atoms}")

# ── Step 3: Generate 3D coordinates ────────────────────────
# EmbedMolecule uses the ETKDG algorithm — it places atoms in
# 3D space using distance geometry, respecting bond lengths,
# angles, and ring planarity. randomSeed=42 makes it
# reproducible — you'll get the same 3D shape every run.
print("\n[3/6] Generating 3D coordinates (ETKDG)...")

params = AllChem.ETKDGv3()
params.randomSeed = 42
result = AllChem.EmbedMolecule(mol_H, params)

if result == -1:
    # Fallback: use random coords if ETKDG fails
    # This can happen with unusual ring systems
    print("      ETKDG failed — trying random coordinates...")
    AllChem.EmbedMolecule(mol_H, randomSeed=42)

print(f"      ✓ 3D coordinates generated")

# Confirm all atoms have positions
conformer = mol_H.GetConformer()
pos = conformer.GetAtomPosition(0)
print(f"      Atom 0 position: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}) Å")

# ── Step 4: Energy minimisation (MMFF94) ───────────────────
# The ETKDG coordinates are geometrically reasonable but not
# chemically optimal. MMFF94 (Merck Molecular Force Field)
# minimises the energy — adjusting bond lengths, angles,
# and torsions to find a realistic low-energy conformation.
# This is the shape Erlotinib is most likely to adopt in solution.
print("\n[4/6] Energy minimisation with MMFF94...")

result = AllChem.MMFFOptimizeMolecule(mol_H, mmffVariant="MMFF94")

if result == 0:
    print("      ✓ Minimisation converged successfully")
elif result == 1:
    print("      ⚠ Minimisation did not fully converge")
    print("        (still usable — structure is reasonable)")
else:
    print("      ⚠ MMFF94 not available — trying UFF...")
    AllChem.UFFOptimizeMolecule(mol_H)
    print("      ✓ UFF minimisation done")

# Save the 3D structure as SDF — useful later for ProLIF
writer = Chem.SDWriter(OUTPUT_SDF)
writer.write(mol_H)
writer.close()
print(f"      ✓ 3D structure saved: {OUTPUT_SDF}")

# ── Step 5: Convert to PDBQT using Meeko ──────────────────
# PDBQT (Protein Data Bank, Partial Charge, Atom Type) is
# AutoDock Vina's proprietary format. It contains:
#   - 3D coordinates (like PDB)
#   - Gasteiger partial charges on each atom
#   - AutoDock atom types (HD, OA, NA, C, A...)
#   - TORSDOF line: number of rotatable bonds
#   - ROOT/BRANCH/ENDROOT: defines the rotatable bond tree
#     so Vina knows which parts can flex during docking
print("\n[5/6] Converting to PDBQT format with Meeko...")

preparator = MoleculePreparation(
    # merge_these_atom_types=("H",) is the default —
    # it merges non-polar H atoms into their parent carbons.
    # This reduces PDBQT file size and speeds up docking.
    # Polar H atoms (on O, N) are kept as they're needed
    # for H-bond detection.
)

# prepare() returns a list of MoleculeSetup objects.
# For a standard ligand this list has exactly one element.
mol_setups = preparator.prepare(mol_H)

if not mol_setups:
    raise RuntimeError("Meeko preparation returned empty list")

setup = mol_setups[0]

# write_string() returns a tuple: (pdbqt_string, is_ok, error_msg)
pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)

if not is_ok:
    raise RuntimeError(f"PDBQT writing failed: {error_msg}")

print(f"      ✓ PDBQT string generated")

# Count rotatable bonds — important for docking accuracy
n_rot = CalcNumRotatableBonds(mol)
print(f"      Rotatable bonds: {n_rot}")
print(f"        (Vina samples {n_rot} torsional degrees of freedom)")

# ── Step 6: Save PDBQT file ────────────────────────────────
print("\n[6/6] Saving PDBQT file...")

os.makedirs(os.path.dirname(OUTPUT_PDBQT), exist_ok=True)
with open(OUTPUT_PDBQT, "w") as f:
    f.write(pdbqt_string)

print(f"      ✓ Saved: {OUTPUT_PDBQT}")

# ── Verify the output ──────────────────────────────────────
print("\n── Verification ──────────────────────────────────────")

with open(OUTPUT_PDBQT) as f:
    pdbqt_lines = f.readlines()

# Parse key lines from PDBQT
atom_lines    = [l for l in pdbqt_lines if l.startswith("ATOM")]
torsdof_lines = [l for l in pdbqt_lines if l.startswith("TORSDOF")]
remark_lines  = [l for l in pdbqt_lines if l.startswith("REMARK")]
root_lines    = [l for l in pdbqt_lines if l.startswith("ROOT")]
branch_lines  = [l for l in pdbqt_lines if l.startswith("BRANCH")]

print(f"  PDBQT lines total : {len(pdbqt_lines)}")
print(f"  ATOM lines        : {len(atom_lines)}")
print(f"  BRANCH lines      : {len(branch_lines)}"
      f"  ← rotatable bonds")

if torsdof_lines:
    print(f"  {torsdof_lines[0].strip()}")

# Print first 3 ATOM lines so you can see the format
print(f"\n  First 3 ATOM lines (showing format):")
for line in atom_lines[:3]:
    print(f"    {line.rstrip()}")
print(f"    ...")
print(f"  Columns: serial name resname chain resSeq x y z"
      f" occupancy bfactor atomtype")

# ── Save 2D image for reference ────────────────────────────
try:
    from rdkit.Chem import Draw
    img = Draw.MolToImage(mol, size=(400, 300))
    img.save(OUTPUT_IMG)
    print(f"\n  2D structure image saved: {OUTPUT_IMG}")
except Exception:
    pass  # image saving is optional

# ── Final summary ──────────────────────────────────────────
print("\n" + "=" * 55)
print("RESULT: Ligand preparation SUCCESSFUL")
print("=" * 55)
print(f"""
  Input  : SMILES string (Erlotinib)
  Output : {OUTPUT_PDBQT}

  Pipeline stages completed:
    ✓ SMILES → RDKit molecule
    ✓ Implicit → explicit hydrogens
    ✓ 2D → 3D coordinates (ETKDG)
    ✓ 3D energy minimisation (MMFF94)
    ✓ RDKit mol → Meeko MoleculeSetup
    ✓ MoleculeSetup → PDBQT string
    ✓ PDBQT saved to disk

  Next step: Run AutoDock Vina docking (Task 3)
""")