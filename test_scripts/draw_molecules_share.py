# ── draw_molecules.py ──────────────────────────────────────────────
# Week 1 task: Download 5 drugs from PubChem and draw them with RDKit
# Requirements: pip install rdkit pubchempy pillow
# ──────────────────────────────────────────────────────────────────

import pubchempy as pcp                          # talks to PubChem API
from rdkit import Chem                           # parses SMILES strings
from rdkit.Chem import Draw, AllChem             # drawing and 3D tools
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image                            # for displaying the grid
import io

# ── Step 1: Define our 5 drugs by their PubChem CID ───────────────
# CID = Compound ID — the unique number PubChem uses for each molecule
drugs = {
    "Aspirin":     2244,    # pain relief, COX inhibitor
    "Ibuprofen":   3672,    # NSAID
    "Caffeine":    2519,    # stimulant, adenosine receptor blocker
    "Erlotinib":   176870,  # our hero! EGFR cancer drug
    "Oseltamivir": 65028,   # Tamiflu, antiviral
}

# ── Step 2: Fetch SMILES strings from PubChem ─────────────────────
print("Fetching molecules from PubChem...\n")

molecules = []   # will store (name, RDKit mol object) tuples
smiles_dict = {} # will store (name, smiles string) for printing

for name, cid in drugs.items():
    # pcp.Compound.from_cid() calls the PubChem REST API
    compound = pcp.Compound.from_cid(cid)
    
    # isomeric_smiles preserves 3D stereochemistry (the @@ symbols)
    smiles = compound.isomeric_smiles
    smiles_dict[name] = smiles
    
    # Chem.MolFromSmiles() parses the SMILES into an RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        print(f"  WARNING: Could not parse {name}")
        continue
    
    molecules.append((name, mol))
    print(f"  {name}")
    print(f"    SMILES : {smiles}")
    print(f"    Atoms  : {mol.GetNumAtoms()} heavy atoms")
    print(f"    Bonds  : {mol.GetNumBonds()} bonds\n")

# ── Step 3: Draw all 5 molecules in a grid ────────────────────────
print("Drawing molecules...")

# Extract just the mol objects and labels
mol_list  = [mol  for name, mol  in molecules]
name_list = [name for name, name2 in [(n,n) for n,_ in molecules]]

# MolsToGridImage creates a single image with all molecules arranged in a grid
img = Draw.MolsToGridImage(
    mol_list,
    molsPerRow=3,           # 3 molecules per row
    subImgSize=(400, 300),  # size of each individual molecule image (pixels)
    legends=name_list, 
    returnPNG=False # labels shown below each molecule
)

# Save to file
img.save("drug_molecules.png")
print("Saved: drug_molecules.png")

# ── Step 4: Also save each molecule individually ──────────────────
print("\nSaving individual images...")

for name, mol in molecules:
    # Add hydrogens and generate 2D coordinates for a clean layout
    AllChem.Compute2DCoords(mol)
    
    # Draw at higher resolution individually
    drawer = rdMolDraw2D.MolDraw2DSVG(500, 400)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    filename = f"{name.lower()}.svg"
    with open(filename, "w") as f:
        f.write(svg)
    print(f"  Saved: {filename}")