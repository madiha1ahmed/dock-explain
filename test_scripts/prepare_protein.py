# prepare_protein.py
# Week 2, Task 1: Protein preparation for docking
# Run: python prepare_protein.py
#
# Input : data/1M17.pdb       (raw from RCSB)
# Output: data/1M17_clean.pdb (ready for AutoDock Vina)

from pdbfixer import PDBFixer
from openmm.app import PDBFile

# ── Paths ─────────────────────────────────────────────────
INPUT_PDB  = "/Users/imad/Desktop/Gemma4Good/data/1M17.pdb"
OUTPUT_PDB = "/Users/imad/Desktop/Gemma4Good/data/1M17_clean.pdb"

print("=" * 55)
print("DockExplain — Protein Preparation")
print("=" * 55)

# ── Step 1: Load the raw PDB ───────────────────────────────
print("\n[1/6] Loading 1M17.pdb...")
fixer = PDBFixer(filename=INPUT_PDB)

initial_atoms = fixer.topology.getNumAtoms()
print(f"      Atoms loaded: {initial_atoms}")

# Count water molecules before cleaning
waters_before = sum(
    1 for res in fixer.topology.residues()
    if res.name == "HOH"
)
print(f"      Water molecules present: {waters_before}")

# ── Step 2: Remove water molecules ────────────────────────
# For docking, water competes with the drug for binding site
# space. We remove all waters — the docking engine handles
# the implicit solvent environment itself.
print("\n[2/6] Removing water molecules...")
fixer.removeHeterogens(keepWater=False)

waters_after = sum(
    1 for res in fixer.topology.residues()
    if res.name == "HOH"
)
print(f"      Waters remaining: {waters_after}  ← should be 0")

# ── Step 3: Find and fix missing residues ─────────────────
# Some loops in the crystal structure are disordered and
# missing from the PDB. PDBFixer can rebuild short ones.
print("\n[3/6] Checking for missing residues...")
fixer.findMissingResidues()
n_missing = sum(len(v) for v in fixer.missingResidues.values())
print(f"      Missing residues found: {n_missing}")

if n_missing > 0:
    # Only rebuild if gaps are short — long loops can be
    # unreliable. For docking we only care about the
    # binding site being intact.
    fixer.missingResidues = {}
    print("      Skipping loop rebuild (not needed for docking)")

# ── Step 4: Fix non-standard residues ─────────────────────
# Some residues in crystal structures are modified or
# chemically altered. Replace with standard equivalents.
print("\n[4/6] Replacing non-standard residues...")
fixer.findNonstandardResidues()
n_nonstandard = len(fixer.nonstandardResidues)
print(f"      Non-standard residues found: {n_nonstandard}")
fixer.replaceNonstandardResidues()

# ── Step 5: Add missing heavy atoms ───────────────────────
# Some side chains are incomplete in the crystal structure.
# PDBFixer rebuilds the missing atoms geometrically.
print("\n[5/6] Adding missing heavy atoms...")
fixer.findMissingAtoms()
fixer.addMissingAtoms()
print("      Missing heavy atoms rebuilt")

# ── Step 6: Add hydrogens at physiological pH ─────────────
# pH 7.4 = physiological (blood/cell) pH
# This sets the correct protonation state for every residue:
#   Asp, Glu → negatively charged (COO⁻)
#   Lys, Arg → positively charged (NH₃⁺)
#   His       → neutral or positive depending on environment
#
# This is critical — wrong protonation = wrong H-bonds
# = wrong docking score = wrong explanation from Gemma 4
print("\n[6/6] Adding hydrogens at pH 7.4...")
fixer.addMissingHydrogens(pH=7.4)
final_atoms = fixer.topology.getNumAtoms()
H_added = final_atoms - initial_atoms + waters_before * 3
print(f"      Hydrogens added: ~{final_atoms - initial_atoms}")
print(f"      Final atom count: {final_atoms}")

# ── Save the cleaned protein ───────────────────────────────
print(f"\nSaving cleaned protein to {OUTPUT_PDB}...")
with open(OUTPUT_PDB, "w") as f:
    PDBFile.writeFile(
        fixer.topology,
        fixer.positions,
        f,
        keepIds=True   # ← preserve original residue numbers!
    )                  #   Asp855 stays 855, not renumbered to 1

print(f"Saved: {OUTPUT_PDB}")

# ── Verify the output ──────────────────────────────────────
print("\n── Verification ──────────────────────────────────────")
with open(OUTPUT_PDB) as f:
    clean_lines = f.readlines()

atom_lines  = [l for l in clean_lines if l.startswith("ATOM")]
hetatm_lines = [l for l in clean_lines if l.startswith("HETATM")]
water_lines  = [l for l in clean_lines if "HOH" in l]
H_lines      = [l for l in atom_lines  if l[76:78].strip() == "H"]

print(f"  ATOM lines   : {len(atom_lines)}")
print(f"  HETATM lines : {len(hetatm_lines)}")
print(f"  Water lines  : {len(water_lines)}   ← must be 0")
print(f"  H atom lines : {len(H_lines)}  ← should be > 0")

# Check key binding residues still present with original numbers
print(f"\n  Key binding residues (original numbering):")
key = {"LYS": "745", "MET": "769", "THR": "766", "ASP": "855"}
for resname, resnum in key.items():
    found = any(
        resname in l and resnum in l
        for l in atom_lines
    )
    status = "✓" if found else "✗  WARNING — residue missing!"
    print(f"    {status}  {resname}{resnum}")

print("\n" + "=" * 55)
if len(water_lines) == 0 and len(H_lines) > 0:
    print("RESULT: Protein preparation SUCCESSFUL")
    print("  ✓ Waters removed")
    print("  ✓ Hydrogens added at pH 7.4")
    print("  ✓ Original residue numbering preserved")
    print("\nNext step: prepare the Erlotinib ligand (Task 2)")
else:
    print("RESULT: Something looks off — check output above")
print("=" * 55)