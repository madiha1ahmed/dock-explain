# verify_install.py
# Run: python verify_install.py
# All 3 sections must print ✓ PASS

# ── Test 1: RDKit ──────────────────────────────────
print("Testing RDKit...")
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    assert mol is not None
    assert mol.GetNumAtoms() == 13
    print(f"  ✓ PASS — RDKit loaded aspirin: {mol.GetNumAtoms()} atoms")
except Exception as e:
    print(f"  ✗ FAIL — RDKit: {e}")

# ── Test 2: AutoDock Vina ──────────────────────────
print("\nTesting AutoDock Vina...")
try:
    from vina import Vina
    v = Vina(sf_name="vina", verbosity=0)
    print(f"  ✓ PASS — AutoDock Vina imported successfully")
    print(f"  ✓ Vina object created (scoring function: vina)")
except Exception as e:
    print(f"  ✗ FAIL — AutoDock Vina: {e}")

# ── Test 3: ProLIF ──────────────────────────────────
print("\nTesting ProLIF...")
try:
    import prolif
    import MDAnalysis as mda
    print(f"  ✓ PASS — ProLIF version: {prolif.__version__}")
    print(f"  ✓ MDAnalysis version: {mda.__version__}")
except Exception as e:
    print(f"  ✗ FAIL — ProLIF: {e}")

# ── Test 4: Meeko ───────────────────────────────────
print("\nTesting Meeko...")
try:
    from meeko import MoleculePreparation
    preparator = MoleculePreparation()
    print(f"  ✓ PASS — Meeko imported successfully")
except Exception as e:
    print(f"  ✗ FAIL — Meeko: {e}")

# ── Summary ─────────────────────────────────────────
print("\n" + "─"*45)
print("If all 4 lines show ✓ PASS, your environment")
print("is ready. Proceed to the docking test.")