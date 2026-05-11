# debug_prolif.py  (fixed version)
# Run: python debug_prolif.py

import warnings
warnings.filterwarnings("ignore")

import MDAnalysis as mda
import prolif as plf
import numpy as np

PROTEIN_PDB = "../data/results/mode_a/1M17_clean.pdb"
LIGAND_SDF  = "../data/results/mode_a/erlotinib_best_pose.sdf"

print("=" * 58)
print("ProLIF Interaction Debugger")
print("=" * 58)
print(f"ProLIF version: {plf.__version__}")

# Load protein
u = mda.Universe(PROTEIN_PDB)
u.atoms.guess_bonds()
prot = plf.Molecule.from_mda(u)

# Load ligand
poses = list(plf.sdf_supplier(LIGAND_SDF))
lig   = poses[0]

print(f"Ligand atoms  : {lig.GetNumAtoms()}")
print(f"Protein residues: {len(u.residues)}")


# ── Test 1: Default fingerprint, ALL residues ─────────────────
print("\n── Test 1: Default settings, all residues ──────────────")
fp1 = plf.Fingerprint(
    interactions=["HBDonor","HBAcceptor","Hydrophobic",
                  "PiStacking","EdgeToFace",
                  "Cationic","Anionic","CationPi","PiCation"]
)
fp1.run_from_iterable(poses, prot, residues="all")
df1 = fp1.to_dataframe(index_col="Pose")

found1 = []
for col in df1.columns:
    parts       = col if isinstance(col, tuple) else (col,)
    prot_res    = parts[-2] if len(parts) >= 2 else parts[0]
    interaction = parts[-1]
    try:
        if bool(df1[col].iloc[0]):
            found1.append((str(prot_res), interaction))
    except Exception:
        pass

print(f"  Total found: {len(found1)}")
for res, itype in found1:
    print(f"    {res:<16} {itype}")


# ── Test 2: Check residue numbering ──────────────────────────
print("\n── Test 2: Residue numbering in cleaned PDB ────────────")
print("  Looking for key EGFR residues...")
key = ["MET769","THR766","LYS745","PHE771",
       "VAL726","CYS773","ASN770","LEU694"]
all_resnames = [str(r) for r in u.residues.resnames]
all_resnums  = list(u.residues.resnums)

for k in key:
    resname = k[:3]
    resnum  = int(k[3:])
    # Check exact match
    exact = any(
        rn == resname and ri == resnum
        for rn, ri in zip(all_resnames, all_resnums)
    )
    if exact:
        print(f"    ✓ {k} present")
    else:
        # Find nearest residue with same name
        matches = [
            ri for rn, ri in zip(all_resnames, all_resnums)
            if rn == resname
        ]
        if matches:
            nearest = min(matches, key=lambda x: abs(x - resnum))
            print(f"    ✗ {k} NOT found — nearest {resname}: {nearest}")
        else:
            print(f"    ✗ {k} NOT found — {resname} not in protein at all")


# ── Test 3: Ligand position ───────────────────────────────────
print("\n── Test 3: Ligand coordinates ──────────────────────────")
conf      = lig.GetConformer()
positions = conf.GetPositions()
centroid  = positions.mean(axis=0)
print(f"  Ligand centroid: "
      f"X={centroid[0]:.2f}, Y={centroid[1]:.2f}, Z={centroid[2]:.2f}")
print(f"  Expected site  : X≈22,  Y≈0,   Z≈53")
dist = np.sqrt(
    (centroid[0]-22)**2 +
    (centroid[1]-0)**2  +
    (centroid[2]-53)**2
)
print(f"  Distance from expected site: {dist:.2f} Å")
if dist < 5.0:
    print(f"  ✓ Ligand correctly positioned in binding site")
else:
    print(f"  ✗ Ligand is displaced — coordinate problem!")


# ── Test 4: VdW contacts (widest net) ────────────────────────
print("\n── Test 4: VdWContact (finds anything within ~4Å) ─────")
fp_vdw = plf.Fingerprint(interactions=["VdWContact"])
fp_vdw.run_from_iterable(poses, prot, residues="all")
df_vdw = fp_vdw.to_dataframe(index_col="Pose")

vdw_residues = []
for col in df_vdw.columns:
    parts    = col if isinstance(col, tuple) else (col,)
    prot_res = parts[-2] if len(parts) >= 2 else parts[0]
    try:
        if bool(df_vdw[col].iloc[0]):
            vdw_residues.append(str(prot_res))
    except Exception:
        pass

print(f"  Residues within VdW contact distance: {len(vdw_residues)}")
for r in sorted(vdw_residues):
    print(f"    {r}")


# ── Test 5: Print what parameters ProLIF accepts ─────────────
print("\n── Test 5: ProLIF interaction parameters ───────────────")
import inspect
for name in ["HBAcceptor", "HBDonor", "Hydrophobic"]:
    try:
        cls = getattr(plf.interactions, name, None)
        if cls:
            sig = inspect.signature(cls.__init__)
            params = [p for p in sig.parameters if p != "self"]
            print(f"  {name}: {params}")
    except Exception as e:
        print(f"  {name}: could not inspect — {e}")