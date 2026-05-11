# check_pdb.py — confirms 1M17.pdb downloaded correctly
# Run: python check_pdb.py

with open("../data/1M17.pdb", "r") as f:
    lines = f.readlines()

# Count what's in the file
atom_lines   = [l for l in lines if l.startswith("ATOM")]
hetatm_lines = [l for l in lines if l.startswith("HETATM")]
remark_lines = [l for l in lines if l.startswith("REMARK")]

print("── 1M17.pdb contents ──────────────────────────")
print(f"  Total lines    : {len(lines)}")
print(f"  ATOM lines     : {len(atom_lines)}  ← protein residues")
print(f"  HETATM lines   : {len(hetatm_lines)}  ← ligands (Erlotinib is here)")
print(f"  REMARK lines   : {len(remark_lines)}  ← metadata")

# Find Erlotinib — its 3-letter code in PDB is AQ4
erlotinib = [l for l in hetatm_lines if "AQ4" in l]
print(f"\n  Erlotinib (AQ4): {len(erlotinib)} atoms found in structure")

# Find key binding residues we'll explain later
key_residues = ["ASP855", "LYS745", "THR766", "MET769"]
print(f"\n  Key binding residues present:")
for res in key_residues:
    resname = res[:3]
    resnum  = res[3:]
    found = [l for l in atom_lines if resname in l and resnum in l]
    status = "✓" if found else "✗"
    print(f"    {status} {res}")

print("\n── What this means for DockExplain ────────────")
print("  The HETATM lines contain Erlotinib already bound")
print("  in the crystal structure. We'll REMOVE it before")
print("  docking, then let AutoDock Vina re-dock it from")
print("  scratch — and see if it finds the same pose.")
print("  If it does, our pipeline is validated.")