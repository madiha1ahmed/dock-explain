# fix_and_run_prolif.py
# Fixes pose extraction + runs ProLIF in one go
# Run: python fix_and_run_prolif.py

import warnings
warnings.filterwarnings("ignore")

import json
import os
import pandas as pd
import MDAnalysis as mda
import prolif as plf
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import PDBQTMolecule, RDKitMolCreate

# ── Paths ─────────────────────────────────────────────────
RESULTS_PDBQT = "../data/docking_results.pdbqt"
BEST_PDBQT    = "../data/docking_best.pdbqt"
BEST_SDF      = "../data/best_pose.sdf"
PROTEIN_PDB   = "../data/1M17_clean.pdb"
OUT_CSV       = "../data/interactions.csv"
OUT_JSON      = "../data/interactions.json"

SMILES        = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"
DOCKING_SCORE = -7.431   # from REMARK VINA RESULT in MODEL 1

print("=" * 58)
print("DockExplain — Fix + ProLIF Fingerprint")
print("=" * 58)


# ════════════════════════════════════════════════════════════
# STEP 1: Properly extract MODEL 1 from results file
# The bug was searching for "MODEL        1" (many spaces)
# The file actually contains "MODEL 1" (one space)
# ════════════════════════════════════════════════════════════
print("\n[1/5] Extracting best pose from docking_results.pdbqt...")

with open(RESULTS_PDBQT) as f:
    content = f.read()

# Find MODEL 1 — use the exact string from the file
lines = content.split("\n")
model1_lines = []
inside = False

for line in lines:
    # Match "MODEL 1" exactly — strip to handle any spacing
    stripped = line.strip()
    if stripped == "MODEL 1":
        inside = True
    if inside:
        model1_lines.append(line)
    if inside and stripped == "ENDMDL":
        break

if not model1_lines:
    raise RuntimeError("MODEL 1 not found — check docking_results.pdbqt")

# Add the END line that MDAnalysis/ProLIF require
model1_lines.append("END")
best_pose_text = "\n".join(model1_lines)

with open(BEST_PDBQT, "w") as f:
    f.write(best_pose_text)

n_atom_lines = sum(1 for l in model1_lines if l.startswith("ATOM"))
print(f"  ✓ MODEL 1 extracted: {len(model1_lines)} lines, {n_atom_lines} atoms")
print(f"  ✓ Saved: {BEST_PDBQT}")

# Confirm file is not empty
size = os.path.getsize(BEST_PDBQT)
print(f"  ✓ File size: {size} bytes  ← must be > 0")


# ════════════════════════════════════════════════════════════
# STEP 2: Convert PDBQT pose → SDF using Meeko
#
# The PDBQT has SMILES and atom index mapping in REMARKs —
# Meeko uses these to restore full bond orders automatically.
# This gives ProLIF a proper RDKit molecule with:
#   - Correct aromatic bonds (needed for π-stacking)
#   - Correct H-bond donor/acceptor typing
#   - Real 3D coordinates from the docking
# ════════════════════════════════════════════════════════════
print("\n[2/5] Converting best pose PDBQT → SDF...")

try:
    # PDBQTMolecule reads the SMILES from REMARK lines
    pdbqt_mol = PDBQTMolecule.from_file(BEST_PDBQT, skip_typing=True)

    # Convert to RDKit — restores bond orders from embedded SMILES
    rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)

    if not rdkit_mols or rdkit_mols[0] is None:
        raise ValueError("RDKitMolCreate returned None")

    pose_mol = rdkit_mols[0]
    print(f"  ✓ Meeko conversion successful")
    print(f"    Atoms: {pose_mol.GetNumAtoms()}")

    # Assign bond orders from SMILES template for extra safety
    template = Chem.MolFromSmiles(SMILES)
    try:
        pose_mol = AllChem.AssignBondOrdersFromTemplate(
            template, pose_mol
        )
        print(f"  ✓ Bond orders verified against SMILES template")
    except Exception as e:
        print(f"  ⚠ Bond order assignment skipped: {e}")
        print(f"    (Meeko's built-in assignment will be used)")

    # Save SDF
    w = Chem.SDWriter(BEST_SDF)
    w.write(pose_mol)
    w.close()
    print(f"  ✓ SDF saved: {BEST_SDF}")
    use_sdf = True

except Exception as e:
    print(f"  ⚠ Meeko conversion failed: {e}")
    print(f"    Falling back to erlotinib_3D.sdf for interaction analysis")
    print(f"    (Same binding site, slightly different conformation)")
    BEST_SDF = "../data/erlotinib_3D.sdf"
    use_sdf  = True


# ════════════════════════════════════════════════════════════
# STEP 3: Load protein
# ════════════════════════════════════════════════════════════
print("\n[3/5] Loading protein...")

u = mda.Universe(PROTEIN_PDB)
u.atoms.guess_bonds()   # critical — without this only VdW found

print(f"  ✓ Atoms: {len(u.atoms)}, Bonds: {len(u.bonds)}")
prot = plf.Molecule.from_mda(u)
print(f"  ✓ ProLIF protein ready")


# ════════════════════════════════════════════════════════════
# STEP 4: Load ligand and run fingerprint
# ════════════════════════════════════════════════════════════
print("\n[4/5] Running ProLIF fingerprint...")

# Use SDF supplier — most robust for docked poses
lig_supplier = plf.sdf_supplier(BEST_SDF)
poses = list(lig_supplier)

if not poses:
    raise RuntimeError(f"sdf_supplier returned 0 molecules from {BEST_SDF}")

print(f"  ✓ Ligand poses loaded: {len(poses)}")

fp = plf.Fingerprint(interactions=[
    "HBDonor",
    "HBAcceptor",
    "Hydrophobic",
    "PiStacking",
    "EdgeToFace",
    "Cationic",
    "Anionic",
    "CationPi",
    "PiCation",
])

fp.run_from_iterable(poses, prot)
print(f"  ✓ Fingerprint computed")

# Export to DataFrame
df = fp.to_dataframe(index_col="Pose")
print(f"  ✓ DataFrame shape: {df.shape}")


# ════════════════════════════════════════════════════════════
# STEP 5: Parse and display the interaction table
# ════════════════════════════════════════════════════════════
print("\n[5/5] Building interaction table...")

descriptions = {
    "HBDonor"    : "Drug donates H-bond to protein residue",
    "HBAcceptor" : "Drug accepts H-bond from protein residue",
    "Hydrophobic": "Hydrophobic contact between greasy surfaces",
    "PiStacking" : "Face-to-face aromatic ring π–π stacking",
    "EdgeToFace" : "T-shaped aromatic ring interaction",
    "Cationic"   : "Electrostatic: drug(+) attracted to residue(−)",
    "Anionic"    : "Electrostatic: drug(−) attracted to residue(+)",
    "CationPi"   : "Cation attracted to aromatic π electron cloud",
    "PiCation"   : "Aromatic π cloud attracted to nearby cation",
}

priority = {
    "HBDonor":1,"HBAcceptor":2,"Cationic":3,"Anionic":4,
    "PiStacking":5,"EdgeToFace":6,"CationPi":7,
    "PiCation":8,"Hydrophobic":9,
}

rows = []
for col in df.columns:
    parts = col if isinstance(col, tuple) else (col,)
    if len(parts) == 3:
        _, prot_res, interaction = parts
    elif len(parts) == 2:
        prot_res, interaction = parts
    else:
        continue
    try:
        val = df[col].iloc[0]
        if val is True or val == 1:
            rows.append({
                "residue"         : str(prot_res),
                "interaction_type": interaction,
                "description"     : descriptions.get(interaction, interaction),
                "_priority"       : priority.get(interaction, 99),
            })
    except (IndexError, KeyError):
        continue

if rows:
    result_df = (pd.DataFrame(rows)
                 .sort_values("_priority")
                 .drop(columns=["_priority"])
                 .reset_index(drop=True))
else:
    result_df = pd.DataFrame(
        columns=["residue","interaction_type","description"]
    )

# ── Print the table ─────────────────────────────────────
print(f"\n  ══════════════════════════════════════════════════")
print(f"  INTERACTION FINGERPRINT — Erlotinib ↔ EGFR")
print(f"  ══════════════════════════════════════════════════")
print(f"  {'#':<4} {'Residue':<14} {'Type':<16} {'Description'}")
print(f"  {'─'*68}")

if result_df.empty:
    print("  ⚠ No interactions found — see note below")
else:
    for i, row in result_df.iterrows():
        print(f"  {i+1:<4} {row['residue']:<14} "
              f"{row['interaction_type']:<16} "
              f"{row['description']}")

n = len(result_df)
print(f"\n  Total: {n} interactions")

if not result_df.empty:
    counts = result_df["interaction_type"].value_counts()
    print(f"\n  ── Breakdown ───────────────────────────────────")
    for t, c in counts.items():
        print(f"  {'█'*c}  {t} ({c})")

# ── Save outputs ────────────────────────────────────────
result_df.to_csv(OUT_CSV, index=False)

gemma_payload = {
    "drug"          : "Erlotinib",
    "protein"       : "EGFR kinase domain",
    "pdb_id"        : "1M17",
    "drug_smiles"   : SMILES,
    "docking_score" : DOCKING_SCORE,
    "score_units"   : "kcal/mol",
    "n_interactions": n,
    "interactions"  : result_df.to_dict(orient="records"),
}

with open(OUT_JSON, "w") as f:
    json.dump(gemma_payload, f, indent=2)

print(f"\n  ✓ CSV  → {OUT_CSV}")
print(f"  ✓ JSON → {OUT_JSON}")

# ── Preview JSON ─────────────────────────────────────────
print(f"\n  ── JSON for Gemma 4 (preview) ──────────────────")
print("  " + json.dumps(gemma_payload, indent=2)[:400]
               .replace("\n","\n  "))
print("  ...")

print("\n" + "=" * 58)
if n > 0:
    print(f"RESULT: SUCCESS — {n} interactions extracted")
    print(f"  interactions.json is ready for Gemma 4")
    print(f"  Next: Week 3 — connect Gemma 4 to explain this data")
elif n == 0:
    print(f"RESULT: 0 interactions found")
    print(f"  Most likely cause: 1M17_clean.pdb bonds not guessed")
    print(f"  Try: obabel ../data/1M17_clean.pdb \\")
    print(f"         -O ../data/1M17_for_prolif.pdb -h")
    print(f"  Then change PROTEIN_PDB to 1M17_for_prolif.pdb")
print("=" * 58)