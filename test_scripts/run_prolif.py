# run_prolif.py
# Week 2, Task 4: Extract interaction fingerprint from docked pose
# Run: python run_prolif.py
#
# Inputs : data/1M17_clean.pdb      (protein with hydrogens)
#          data/docking_best.pdbqt  (best docked pose)
#          data/erlotinib_3D.sdf    (template for bond orders)
# Outputs: data/interactions.csv    (structured table → Gemma 4 input)
#          data/interactions.json   (same data, JSON format)

import warnings
warnings.filterwarnings("ignore")   # suppress MDAnalysis mass warnings

import json
import numpy as np
import pandas as pd
import MDAnalysis as mda
import prolif as plf
from rdkit import Chem
from rdkit.Chem import AllChem

# ── Paths ────────────────────────────────────────────────────
PROTEIN_PDB   = "/Users/imad/Desktop/Gemma4Good/data/1M17_clean.pdb"
LIGAND_PDBQT  = "/Users/imad/Desktop/Gemma4Good/data/docking_best.pdbqt"
LIGAND_SDF    = "/Users/imad/Desktop/Gemma4Good/data/erlotinib_3D.sdf"
OUT_CSV       = "/Users/imad/Desktop/Gemma4Good/data/interactions.csv"
OUT_JSON      = "/Users/imad/Desktop/Gemma4Good/data/interactions.json"

ERLOTINIB_SMILES = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"

print("=" * 58)
print("DockExplain — ProLIF Interaction Fingerprint")
print("Erlotinib ↔ EGFR binding site analysis")
print("=" * 58)


# ════════════════════════════════════════════════════════════
# STEP 1: Load the protein with MDAnalysis
#
# Critical requirement: protein must have explicit hydrogens.
# Our 1M17_clean.pdb has them from PDBFixer — good.
# Critical fix: call guess_bonds() immediately after loading.
# Without it MDAnalysis treats the protein as a point cloud
# and ProLIF returns ONLY VdWContact — nothing useful.
# ════════════════════════════════════════════════════════════
print("\n[1/5] Loading protein with MDAnalysis...")

u = mda.Universe(PROTEIN_PDB)

# THE CRITICAL LINE — do not skip this
u.atoms.guess_bonds()

n_atoms    = len(u.atoms)
n_residues = len(u.residues)
n_bonds    = len(u.bonds)

print(f"  ✓ Protein loaded")
print(f"    Atoms    : {n_atoms}")
print(f"    Residues : {n_residues}")
print(f"    Bonds    : {n_bonds}  ← must be > 0, confirms guess_bonds worked")

if n_bonds == 0:
    raise RuntimeError(
        "Bond guessing failed — ProLIF will only find VdW contacts.\n"
        "Check that 1M17_clean.pdb has valid atom names."
    )

# Convert to ProLIF Molecule
prot = plf.Molecule.from_mda(u)
print(f"  ✓ ProLIF protein molecule created")


# ════════════════════════════════════════════════════════════
# STEP 2: Load the docked pose (ligand)
#
# PDBQT format strips bond orders — Vina doesn't need them,
# but ProLIF does to correctly classify aromatic atoms.
# Solution: provide the original SMILES as a "template" that
# tells ProLIF which bonds are single/double/aromatic.
# pdbqt_supplier() handles the matching automatically.
# ════════════════════════════════════════════════════════════
print("\n[2/5] Loading docked ligand pose...")

# Build template from SMILES (no 3D needed — just bond orders)
template = Chem.MolFromSmiles(ERLOTINIB_SMILES)

# pdbqt_supplier reads the PDBQT file and assigns bond orders
# using the template. Returns an iterable of prolif.Molecule.
lig_supplier = plf.pdbqt_supplier(
    [LIGAND_PDBQT],   # list of PDBQT files — one per pose
    template,          # SMILES template for bond order assignment
)

# Get the first (and only) pose from our best pose file
ligand_poses = list(lig_supplier)

if not ligand_poses:
    raise RuntimeError(
        "No ligand poses loaded. Check that docking_best.pdbqt exists."
    )

print(f"  ✓ Ligand pose loaded")
print(f"    Poses available : {len(ligand_poses)}")
print(f"    Ligand atoms    : {ligand_poses[0].GetNumAtoms()}")
print(f"    (Template atoms): {template.GetNumAtoms()}")


# ════════════════════════════════════════════════════════════
# STEP 3: Run the interaction fingerprint
#
# We check ALL 7 interaction types — the 4 you learned in
# Week 1 plus 3 more that commonly appear in kinase pockets.
# ProLIF automatically finds all protein residues within 6 Å
# of the ligand (residues=None is the default).
# ════════════════════════════════════════════════════════════
print("\n[3/5] Computing interaction fingerprint...")

fp = plf.Fingerprint(
    interactions=[
        "HBDonor",       # drug donates H to protein residue
        "HBAcceptor",    # drug accepts H from protein residue
        "Hydrophobic",   # greasy contacts (C↔C within 5.5 Å)
        "PiStacking",    # aromatic ring stacks face-to-face
        "EdgeToFace",    # aromatic ring T-shaped stack
        "Cationic",      # drug (+) attracted to residue (-)
        "Anionic",       # drug (-) attracted to residue (+)
        "CationPi",      # cation attracted to aromatic π cloud
        "PiCation",      # aromatic π cloud attracted to cation
    ]
)

# run_from_iterable is the correct method for docking poses
# (as opposed to fp.run() which is for MD trajectories)
fp.run_from_iterable(
    ligand_poses,  # iterable of prolif.Molecule objects
    prot,          # prolif.Molecule of the protein
)

print(f"  ✓ Fingerprint computed")


# ════════════════════════════════════════════════════════════
# STEP 4: Parse results into a clean interaction table
#
# This table is THE INPUT to Gemma 4 in Week 3.
# Format: one row per interaction found.
# Columns: residue, interaction_type, description
# ════════════════════════════════════════════════════════════
print("\n[4/5] Parsing interaction table...")

# Export to pandas DataFrame
# index_col="Pose" labels rows by pose number
df = fp.to_dataframe(index_col="Pose")

if df.empty:
    print("  ⚠ DataFrame is empty — no interactions found")
    print("    This usually means bond guessing failed or")
    print("    the ligand didn't dock inside the binding pocket")
else:
    print(f"  ✓ DataFrame shape: {df.shape}")

# ── Parse the multi-level column DataFrame ─────────────────
# ProLIF uses a MultiIndex: (ligand_resid, protein_resid, interaction)
# We flatten it into a simple table for Gemma 4

rows = []

# Descriptions for each interaction type — Gemma 4 will expand these
interaction_descriptions = {
    "HBDonor"    : "Drug donates a hydrogen bond to this residue",
    "HBAcceptor" : "Drug accepts a hydrogen bond from this residue",
    "Hydrophobic": "Hydrophobic contact — greasy surfaces touching",
    "PiStacking" : "Face-to-face π–π stacking of aromatic rings",
    "EdgeToFace" : "T-shaped edge-to-face aromatic ring interaction",
    "Cationic"   : "Electrostatic attraction — drug(+) to residue(−)",
    "Anionic"    : "Electrostatic attraction — drug(−) to residue(+)",
    "CationPi"   : "Cation attracted to aromatic π electron cloud",
    "PiCation"   : "Aromatic π cloud attracted to cation",
}

# Walk through MultiIndex columns
for col in df.columns:
    # col is a tuple: (ligand_id, protein_res, interaction_type)
    if len(col) == 3:
        lig_id, prot_res, interaction = col
    elif len(col) == 2:
        prot_res, interaction = col
    else:
        continue

    # Check if this interaction was detected (True in pose 0)
    try:
        value = df[col].iloc[0]
        if value is True or value == True:
            rows.append({
                "residue"         : str(prot_res),
                "interaction_type": interaction,
                "description"     : interaction_descriptions.get(
                                        interaction, interaction
                                    ),
            })
    except (IndexError, KeyError):
        continue

# ── Build clean interaction DataFrame ──────────────────────
if rows:
    result_df = pd.DataFrame(rows)

    # Sort: H-bonds first, then hydrophobic, then others
    priority = {
        "HBDonor": 1, "HBAcceptor": 2,
        "Cationic": 3, "Anionic": 4,
        "PiStacking": 5, "EdgeToFace": 6,
        "CationPi": 7, "PiCation": 8,
        "Hydrophobic": 9,
    }
    result_df["_priority"] = result_df["interaction_type"].map(
        lambda x: priority.get(x, 99)
    )
    result_df = result_df.sort_values("_priority").drop(
        columns=["_priority"]
    ).reset_index(drop=True)

else:
    result_df = pd.DataFrame(
        columns=["residue", "interaction_type", "description"]
    )

# ── Print the interaction table ─────────────────────────────
print(f"\n  ══ Interaction Fingerprint Table ══════════════════")
print(f"  {'#':<4} {'Residue':<14} {'Interaction':<16} {'Description'}")
print(f"  {'─'*72}")

if result_df.empty:
    print("  No interactions detected — see troubleshooting below")
else:
    for i, row in result_df.iterrows():
        print(f"  {i+1:<4} {row['residue']:<14} "
              f"{row['interaction_type']:<16} {row['description']}")

print(f"\n  Total interactions found: {len(result_df)}")

# ── Count by type ───────────────────────────────────────────
if not result_df.empty:
    type_counts = result_df["interaction_type"].value_counts()
    print(f"\n  ── Breakdown by type ───────────────────────────")
    for itype, count in type_counts.items():
        bar = "█" * count
        print(f"  {itype:<16} {bar} ({count})")


# ════════════════════════════════════════════════════════════
# STEP 5: Save outputs for Gemma 4 (Week 3)
#
# Two formats:
#   CSV  — easy to read and inspect
#   JSON — structured for building the Gemma 4 prompt
# ════════════════════════════════════════════════════════════
print("\n[5/5] Saving outputs...")

# Save CSV
result_df.to_csv(OUT_CSV, index=False)
print(f"  ✓ CSV saved: {OUT_CSV}")

# Build the JSON payload — this is exactly what Week 3
# will pass to Gemma 4 as the structured context
gemma_payload = {
    "protein"       : "EGFR kinase domain",
    "pdb_id"        : "1M17",
    "drug"          : "Erlotinib",
    "drug_smiles"   : ERLOTINIB_SMILES,
    "docking_score" : -7.1,
    "score_units"   : "kcal/mol",
    "n_interactions": len(result_df),
    "interactions"  : result_df.to_dict(orient="records"),
    "summary_for_gemma": (
        f"Erlotinib docked to EGFR with a binding affinity of "
        f"-7.1 kcal/mol. ProLIF detected {len(result_df)} "
        f"molecular interactions between the drug and the "
        f"protein binding site residues."
    )
}

with open(OUT_JSON, "w") as f:
    json.dump(gemma_payload, f, indent=2)
print(f"  ✓ JSON saved: {OUT_JSON}")

# ── Preview the JSON Gemma 4 will receive ──────────────────
print(f"\n  ── Preview: JSON payload for Gemma 4 ─────────────")
print(f"  {json.dumps(gemma_payload, indent=2)[:600]}...")

# ── Final summary ───────────────────────────────────────────
print("\n" + "=" * 58)
if len(result_df) > 0:
    print(f"RESULT: SUCCESS — {len(result_df)} interactions extracted")
    print(f"\n  These interactions explain WHY Erlotinib scores")
    print(f"  -7.1 kcal/mol. In Week 3, this JSON goes to")
    print(f"  Gemma 4, which turns it into a plain-language")
    print(f"  explanation — the core of DockExplain.")
else:
    print("RESULT: No interactions found — see troubleshooting")
    print("  1. Confirm 1M17_clean.pdb has hydrogens")
    print("  2. Confirm docking_best.pdbqt is not empty")
    print("  3. Check that u.atoms.guess_bonds() ran above")
print("=" * 58)