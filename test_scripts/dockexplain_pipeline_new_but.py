# dockexplain_pipeline.py  (smart input detection version)
# 
# Usage A — co-crystal, analyse existing ligand (no SMILES needed):
#   python dockexplain_pipeline.py \
#       --protein ../data/1M17.pdb \
#       --drug    Erlotinib
#
# Usage B — co-crystal, dock a NEW drug to same pocket:
#   python dockexplain_pipeline.py \
#       --protein ../data/1M17.pdb \
#       --smiles  "CC(C)Cc1ccc(cc1)C(C)C(=O)O" \
#       --drug    Ibuprofen
#
# Usage C — apo protein, SMILES required:
#   python dockexplain_pipeline.py \
#       --protein ../data/my_protein.pdb \
#       --smiles  "CC(=O)Oc1ccccc1C(=O)O" \
#       --drug    Aspirin

import argparse, json, os, warnings, subprocess, urllib.request
import tempfile
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import MDAnalysis as mda
import prolif as plf
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from meeko import PDBQTMolecule, RDKitMolCreate
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from vina import Vina
from rdkit.Geometry import Point3D
from validation_suite import run_all_checks, print_report, save_report
from docking_engine import run_docking, detect_docking_engine


# ════════════════════════════════════════════════════════════
# INPUT DETECTION FUNCTIONS
# ════════════════════════════════════════════════════════════

def get_protein_name(pdb_path):
    """Extract protein name from HEADER/COMPND or filename."""
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HEADER"):
                name = line[10:50].strip()
                if name: return name
            if line.startswith("COMPND") and "MOLECULE:" in line:
                name = line.split("MOLECULE:")[-1].strip().rstrip(";")
                if name: return name
    return os.path.splitext(os.path.basename(pdb_path))[0]


def find_ligand_code(pdb_path):
    """
    Scan HETATM records in any PDB file.
    Skip water and common buffer/crystallographic agents.
    Return the 3-letter code with the most atoms (= the drug).
    Returns None if no drug-like ligand found (apo structure).
    """
    SKIP = {
        "HOH","WAT","SO4","GOL","EDO","PEG","MPD","TRS","MES",
        "ACT","NO3","CL","NA","MG","ZN","CA","K","FE","MN","CO",
        "BME","DTT","DMS","IOD","BR","F","PO4","EPE","HED","FMT",
        "IMD","ACE","NH2","SEP","TPO","PTR","CSO","HIC","MLY",
    }
    found = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM"):
                code = line[17:20].strip()
                if code and code not in SKIP:
                    found[code] = found.get(code, 0) + 1
    return max(found, key=found.get) if found else None


def fetch_smiles_from_rcsb(ligand_code):
    """
    Fetch canonical SMILES for a 3-letter PDB ligand code.
    Tries 3 sources in order — falls back gracefully if any fail.

    Source 1: RCSB Data API (JSON)
    Source 2: RCSB ideal SDF file → parse with RDKit
    Source 3: PubChem by CCD code name

    Returns SMILES string or None.
    """
    code = ligand_code.upper()

    # ── Source 1: RCSB Data API ───────────────────────────
    # Correct field path as of 2025: pdbx_chem_comp_descriptor
    # is a LIST of dicts, each with 'type' and 'descriptor' keys
    try:
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}"
        req = urllib.request.Request(
            url, headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())

        # pdbx_chem_comp_descriptor is a list of descriptor objects
        descriptors = data.get("pdbx_chem_comp_descriptor", [])

        # Prefer SMILES_CANONICAL, fall back to SMILES
        smiles = None
        for d in descriptors:
            dtype = d.get("type","").upper()
            if dtype == "SMILES_CANONICAL":
                smiles = d.get("descriptor")
                break
        if not smiles:
            for d in descriptors:
                dtype = d.get("type","").upper()
                if "SMILES" in dtype:
                    smiles = d.get("descriptor")
                    break

        if smiles:
            # Validate with RDKit before returning
            if Chem.MolFromSmiles(smiles) is not None:
                print(f"  ✓ SMILES from RCSB Data API")
                return smiles

    except Exception as e:
        print(f"  ⚠ RCSB Data API failed: {e}")

    # ── Source 2: RCSB ideal SDF file ────────────────────
    # RCSB hosts an ideal-geometry SDF for every CCD entry
    # This is the most reliable source — direct file, no JSON parsing
    try:
        url = f"https://files.rcsb.org/ligands/download/{code}_ideal.sdf"
        with urllib.request.urlopen(url, timeout=10) as r:
            sdf_data = r.read().decode("utf-8")

        mol = Chem.MolFromMolBlock(sdf_data, removeHs=True)
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            if smiles:
                print(f"  ✓ SMILES from RCSB ideal SDF")
                return smiles

    except Exception as e:
        print(f"  ⚠ RCSB SDF download failed: {e}")

    # ── Source 3: PubChem by CCD code ────────────────────
    # PubChem indexes PDB CCD codes as compound names
    try:
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
               f"/name/{code}/property/IsomericSMILES/JSON")
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())

        smiles = (data.get("PropertyTable", {})
                      .get("Properties", [{}])[0]
                      .get("IsomericSMILES"))
        if smiles and Chem.MolFromSmiles(smiles) is not None:
            print(f"  ✓ SMILES from PubChem")
            return smiles

    except Exception as e:
        print(f"  ⚠ PubChem lookup failed: {e}")

    print(f"  ✗ All 3 SMILES sources failed for {code}")
    return None


def extract_ligand_centroid(pdb_path, ligand_code):
    """Compute binding site centroid from any ligand in any PDB."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM") and ligand_code in line[17:21]:
                try:
                    coords.append([
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ])
                except ValueError:
                    continue
    if not coords:
        return None, 0
    return np.array(coords).mean(axis=0).tolist(), len(coords)


def extract_ligand_as_sdf(pdb_path, ligand_code, smiles, out_path):
    """
    Get the crystal ligand pose in SDF format.
    
    Strategy (tries in order):
      1. RCSB ModelServer — crystal coords + correct bond orders in one file
      2. PDB HETATM block + AssignBondOrdersFromTemplate
      3. ETKDG embedding translated to crystal centroid (last resort)
    """
    from rdkit.Geometry import Point3D
    code = ligand_code.upper()

    # Extract PDB ID from filename — needed for ModelServer URL
    # e.g. "1M17.pdb" → "1m17"
    pdb_id = os.path.splitext(
        os.path.basename(pdb_path)
    )[0].replace("_clean","").lower()

    # ── Source 1: RCSB ModelServer SDF ───────────────────────
    # Returns the ligand from the crystal structure in SDF format
    # with correct bond orders AND experimental coordinates
    try:
        url = (f"https://models.rcsb.org/v1/{pdb_id}/ligand"
               f"?label_comp_id={code}&encoding=sdf"
               f"&copy_all_categories=false&download=false")
        print(f"  Fetching crystal SDF from RCSB ModelServer...")
        req = urllib.request.Request(url, headers={"Accept": "text/plain"})
        with urllib.request.urlopen(req, timeout=15) as r:
            sdf_data = r.read().decode("utf-8")

        if sdf_data and len(sdf_data) > 100:
            mol = Chem.MolFromMolBlock(sdf_data, removeHs=False,
                                       sanitize=True)
            if mol is not None and mol.GetNumAtoms() > 0:
                # Add any missing Hs with coordinates
                mol = Chem.AddHs(mol, addCoords=True)
                writer = Chem.SDWriter(out_path)
                writer.write(mol)
                writer.close()
                print(f"  ✓ Crystal SDF from RCSB ModelServer "
                      f"({mol.GetNumAtoms()} atoms)")
                return out_path

    except Exception as e:
        print(f"  ⚠ ModelServer failed: {e}")

    # ── Source 2: PDB HETATM block + template bond orders ────
    # Read PDB block, assign bond orders from SMILES template
    try:
        print(f"  Trying PDB HETATM block + template matching...")
        hetatm_lines = []
        with open(pdb_path) as f:
            for line in f:
                if (line.startswith("HETATM")
                        and code in line[17:21]):
                    hetatm_lines.append(line)

        if hetatm_lines:
            pdb_block = "".join(hetatm_lines) + "END\n"
            # sanitize=False is critical — PDB often has valence issues
            raw_mol = Chem.MolFromPDBBlock(pdb_block,
                                           removeHs=True,
                                           sanitize=False)
            if raw_mol is not None:
                template = Chem.MolFromSmiles(smiles)
                if template is not None:
                    mol = AllChem.AssignBondOrdersFromTemplate(
                        template, raw_mol
                    )
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception:
                        pass
                    mol = Chem.AddHs(mol, addCoords=True)
                    writer = Chem.SDWriter(out_path)
                    writer.write(mol)
                    writer.close()
                    n = mol.GetNumAtoms()
                    print(f"  ✓ Crystal coords from PDB block "
                          f"({n} atoms)")
                    return out_path

    except Exception as e:
        print(f"  ⚠ PDB block approach failed: {e}")

    # ── Source 3: ETKDG at crystal centroid (last resort) ────
    print(f"  Using ETKDG near crystal centroid (last resort)...")
    try:
        coords = []
        with open(pdb_path) as f:
            for line in f:
                if (line.startswith("HETATM")
                        and code in line[17:21]
                        and line[76:78].strip() != "H"):
                    try:
                        coords.append([float(line[30:38]),
                                       float(line[38:46]),
                                       float(line[46:54])])
                    except ValueError:
                        pass

        centroid = np.array(coords).mean(axis=0) if coords else None

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        p = AllChem.ETKDGv3(); p.randomSeed = 42
        AllChem.EmbedMolecule(mol, p)
        AllChem.MMFFOptimizeMolecule(mol)

        if centroid is not None:
            conf = mol.GetConformer()
            positions = conf.GetPositions()
            mol_centroid = positions.mean(axis=0)
            offset = centroid - mol_centroid
            for i in range(mol.GetNumAtoms()):
                p = positions[i] + offset
                conf.SetAtomPosition(i, Point3D(*p))

        writer = Chem.SDWriter(out_path)
        writer.write(mol)
        writer.close()
        print(f"  ✓ ETKDG pose at crystal centroid")
        return out_path

    except Exception as e:
        print(f"  ✗ All sources failed: {e}")
        return None


# ════════════════════════════════════════════════════════════
# PROTEIN PREPARATION
# ════════════════════════════════════════════════════════════

def prepare_protein(raw_pdb, out_path):
    """Clean any raw PDB — preserves original residue numbers."""
    fixer = PDBFixer(filename=raw_pdb)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.4)

    with open(out_path, "w") as f:
        PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            f,
            keepIds=True    # preserves original residue numbers
        )

    # ── Verify residue numbering wasn't corrupted ────────────
    # PDBFixer sometimes resets numbers despite keepIds=True.
    # Check by comparing a known residue from raw vs clean.
    raw_nums   = set()
    clean_nums = set()
    with open(raw_pdb) as f:
        for line in f:
            if line.startswith("ATOM"):
                try: raw_nums.add(int(line[22:26]))
                except ValueError: pass
    with open(out_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                try: clean_nums.add(int(line[22:26]))
                except ValueError: pass

    if raw_nums and not raw_nums.issubset(clean_nums):
        print(f"  ⚠ Residue renumbering detected — "
              f"interaction labels may differ from literature")
    return out_path


# ════════════════════════════════════════════════════════════
# DOCKING FUNCTIONS (only used when new SMILES provided)
# ════════════════════════════════════════════════════════════

def smiles_to_pdbqt(smiles, out_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    if AllChem.EmbedMolecule(mol, p) == -1:
        AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94")
    preparator = MoleculePreparation()
    setups = preparator.prepare(mol)
    pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not is_ok:
        raise RuntimeError(f"Meeko failed: {err}")
    with open(out_path, "w") as f:
        f.write(pdbqt_str)
    return pdbqt_str


def pdb_to_pdbqt_receptor(pdb_path, out_path):
    r = subprocess.run(
        ["obabel", pdb_path, "-O", out_path, "-xr"],
        capture_output=True, text=True
    )
    if not os.path.exists(out_path):
        raise RuntimeError(f"OpenBabel failed: {r.stderr}")
    return out_path


def run_vina(receptor_pdbqt, ligand_pdbqt, center,
             box_size, exhaustiveness=8, n_poses=10):
    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(center=center, box_size=box_size)
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    poses_str = v.poses(n_poses=n_poses)
    scores = []
    for line in poses_str.split("\n"):
        if "VINA RESULT" in line:
            try: scores.append(float(line.split()[3]))
            except (IndexError, ValueError): pass
    return scores[0] if scores else None, scores, poses_str


def extract_best_pose(poses_str, out_path):
    lines = poses_str.split("\n")
    model1, inside = [], False
    for line in lines:
        if line.strip() == "MODEL 1": inside = True
        if inside: model1.append(line)
        if inside and line.strip() == "ENDMDL": break
    if not model1:
        raise RuntimeError("MODEL 1 not found in Vina output")
    model1.append("END")
    with open(out_path, "w") as f:
        f.write("\n".join(model1))
    return out_path


def pdbqt_pose_to_sdf(pdbqt_path, smiles, out_path):
    pdbqt_mol  = PDBQTMolecule.from_file(pdbqt_path, skip_typing=True)
    rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    if not rdkit_mols or rdkit_mols[0] is None:
        raise RuntimeError("Meeko pose conversion failed")
    pose_mol = rdkit_mols[0]
    try:
        pose_mol = AllChem.AssignBondOrdersFromTemplate(
            Chem.MolFromSmiles(smiles), pose_mol)
    except Exception:
        pass
    w = Chem.SDWriter(out_path); w.write(pose_mol); w.close()
    return out_path


# ════════════════════════════════════════════════════════════
# ProLIF (same for all modes)
# ════════════════════════════════════════════════════════════

def run_prolif(protein_pdb, pose_sdf):
    u = mda.Universe(protein_pdb)
    u.atoms.guess_bonds()

    # Convert to ProLIF molecule with error tolerance
    # NoImplicit=False allows MDAnalysis to handle
    # unusual valence states from PDBFixer H placement
    try:
        prot = plf.Molecule.from_mda(u)
    except Exception as e:
        print(f"  ⚠ Standard conversion failed: {e}")
        print(f"  Retrying with relaxed valence checking...")
        prot = plf.Molecule.from_mda(
            u,
            NoImplicit=False,      # tolerate unusual valence
            max_iter=0,            # skip standardization
        )

    poses = list(plf.sdf_supplier(pose_sdf))
    if not poses:
        raise RuntimeError(f"No molecules in {pose_sdf}")

    fp = plf.Fingerprint(
        interactions=[
            "HBDonor","HBAcceptor","Hydrophobic",
            "PiStacking","EdgeToFace",
            "Cationic","Anionic","CationPi","PiCation",
        ],
        parameters={
            "HBAcceptor" : {"distance": 4.0, "DHA_angle": (100, 180)},
            "HBDonor"    : {"distance": 4.0, "DHA_angle": (100, 180)},
            "Hydrophobic": {"distance": 6.0},
        }
    )

    fp.run_from_iterable(poses, prot, residues="all")
    df = fp.to_dataframe(index_col="Pose")

    descriptions = {
        "HBDonor"    : "Drug donates H-bond to protein residue",
        "HBAcceptor" : "Drug accepts H-bond from protein residue",
        "Hydrophobic": "Hydrophobic contact between greasy surfaces",
        "PiStacking" : "Face-to-face aromatic π–π ring stacking",
        "EdgeToFace" : "T-shaped aromatic ring interaction",
        "Cationic"   : "Electrostatic: drug(+) to residue(−)",
        "Anionic"    : "Electrostatic: drug(−) to residue(+)",
        "CationPi"   : "Cation attracted to aromatic π cloud",
        "PiCation"   : "Aromatic π cloud attracted to cation",
    }
    priority = {
        "HBDonor":1,"HBAcceptor":2,"Cationic":3,"Anionic":4,
        "PiStacking":5,"EdgeToFace":6,"CationPi":7,
        "PiCation":8,"Hydrophobic":9,
    }

    rows = []
    for col in df.columns:
        parts       = col if isinstance(col, tuple) else (col,)
        prot_res    = parts[-2] if len(parts) >= 2 else parts[0]
        interaction = parts[-1]
        try:
            if bool(df[col].iloc[0]):
                rows.append({
                    "residue"         : str(prot_res),
                    "interaction_type": interaction,
                    "description"     : descriptions.get(
                                            interaction, interaction),
                    "_p"              : priority.get(interaction, 99),
                })
        except (IndexError, KeyError):
            continue

    if not rows:
        return pd.DataFrame(
            columns=["residue","interaction_type","description"])
    return (pd.DataFrame(rows)
            .sort_values("_p").drop(columns=["_p"])
            .reset_index(drop=True))


# ════════════════════════════════════════════════════════════
# MAIN PIPELINE — smart input handling
# ════════════════════════════════════════════════════════════

def run_pipeline(raw_protein_pdb, drug_name,
                 drug_smiles=None, out_dir="results/",
                 exhaustiveness=8):
    """
    Smart DockExplain pipeline.

    Mode A — Co-crystal PDB, no SMILES:
        Extracts ligand automatically, fetches SMILES from RCSB,
        uses crystallographic pose directly (no docking needed).

    Mode B — Co-crystal PDB + new SMILES:
        Uses crystal binding site but docks the new drug.

    Mode C — Apo PDB + SMILES:
        Full docking with protein centroid (or P2Rank in production).

    Mode D — Apo PDB, no SMILES:
        Raises an error — cannot proceed.
    """
    os.makedirs(out_dir, exist_ok=True)
    safe      = drug_name.lower().replace(" ","_")
    prot_base = os.path.splitext(os.path.basename(raw_protein_pdb))[0]

    print(f"\n{'='*58}")
    print(f"DockExplain Pipeline")
    print(f"  Protein : {os.path.basename(raw_protein_pdb)}")
    print(f"  Drug    : {drug_name}")
    print(f"{'='*58}")

    # ── 1. Detect what kind of PDB we have ─────────────────
    print("\n[1/7] Analysing input PDB...")
    protein_name = get_protein_name(raw_protein_pdb)
    lig_code     = find_ligand_code(raw_protein_pdb)

    print(f"  Protein name     : {protein_name}")
    print(f"  Co-crystal ligand: {lig_code or 'none (apo structure)'}")
    print(f"  User SMILES      : {'provided' if drug_smiles else 'not provided'}")

    # ── 2. Determine mode ───────────────────────────────────
    if lig_code and not drug_smiles:
        mode = "A"
        print(f"\n  → Mode A: co-crystal PDB, no SMILES")
        print(f"    Will analyse existing ligand ({lig_code})")
        print(f"    No docking needed — using crystal pose directly")

    elif lig_code and drug_smiles:
        mode = "B"
        print(f"\n  → Mode B: co-crystal PDB + new drug SMILES")
        print(f"    Will dock {drug_name} into the {lig_code} binding site")

    elif not lig_code and drug_smiles:
        mode = "C"
        print(f"\n  → Mode C: apo PDB + SMILES")
        print(f"    Will run blind docking")

    else:  # no ligand, no SMILES
        raise ValueError(
            "Cannot proceed: your PDB has no co-crystal ligand "
            "and no SMILES was provided.\n"
            "Please either:\n"
            "  • Use a co-crystal PDB (contains the drug already), or\n"
            "  • Provide a --smiles string for the drug to dock."
        )

    # ── 3. Get binding site from raw PDB ───────────────────
    print("\n[2/7] Detecting binding site...")
    if lig_code:
        center, n_atoms = extract_ligand_centroid(raw_protein_pdb, lig_code)
        method = f"centroid of {lig_code} ({n_atoms} atoms)"
        print(f"  ✓ Binding site from {lig_code}: "
              f"X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}")
    else:
        # Apo — use protein centroid as fallback
        coords = []
        with open(raw_protein_pdb) as f:
            for line in f:
                if line.startswith("ATOM"):
                    try:
                        coords.append([float(line[30:38]),
                                       float(line[38:46]),
                                       float(line[46:54])])
                    except ValueError:
                        pass
        center = np.array(coords).mean(axis=0).tolist()
        method = "protein geometric centroid (apo — consider P2Rank)"
        print(f"  ⚠ Apo structure — using protein centroid")
        print(f"    X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}")

    box_size = [25.0, 25.0, 25.0]

    # ── 4. Resolve SMILES ───────────────────────────────────
    print("\n[3/7] Resolving drug SMILES...")
    if mode == "A":
        # Fetch SMILES automatically from RCSB
        print(f"  Fetching SMILES for {lig_code} from RCSB CCD API...")
        drug_smiles = fetch_smiles_from_rcsb(lig_code)
        if drug_smiles:
            print(f"  ✓ SMILES fetched: {drug_smiles[:60]}...")
        else:
            raise RuntimeError(
                f"Could not fetch SMILES for {lig_code} from RCSB.\n"
                f"Please provide --smiles manually."
            )
    else:
        print(f"  ✓ Using provided SMILES: {drug_smiles[:60]}...")

    # ── 5. Clean protein ────────────────────────────────────
    print("\n[4/7] Preparing protein (PDBFixer)...")
    clean_pdb = os.path.join(out_dir, f"{prot_base}_clean.pdb")
    prepare_protein(raw_protein_pdb, clean_pdb)
    n_atoms = sum(1 for l in open(clean_pdb) if l.startswith("ATOM"))
    print(f"  ✓ Cleaned PDB: {n_atoms} ATOM lines")

    # ── 6. Get ligand pose ──────────────────────────────────
    print("\n[5/7] Preparing ligand pose...")
    best_sdf = os.path.join(out_dir, f"{safe}_best_pose.sdf")
    docking_score  = None
    all_scores     = []

    if mode == "A":
        # Use crystal pose directly — no docking
        print(f"  Extracting {lig_code} crystal pose from PDB...")
        result = extract_ligand_as_sdf(
            raw_protein_pdb, lig_code, drug_smiles, best_sdf
        )
        if result is None:
            raise RuntimeError(
                f"Could not extract {lig_code} from PDB. "
                f"Try providing --smiles to trigger docking instead."
            )
        docking_score = None   # crystal pose has no docking score
        print(f"  ✓ Crystal pose extracted: {best_sdf}")
        print(f"  ✓ No docking needed — crystallographic pose used")

    else:
    # Modes B and C — run full docking
        print(f"  Detecting docking engine...")
        engine_name, _ = detect_docking_engine()

        docking_result = run_docking(
            receptor_pdb   = clean_pdb,
            drug_smiles    = drug_smiles,
            drug_name      = drug_name,
            center         = center,
            box_size       = box_size,
            out_dir        = out_dir,
            exhaustiveness = exhaustiveness,
            n_poses        = 10,
            cpu            = 4,       # safe for M2 Air 8GB
            seed           = None,    # random each run
        )

        if not docking_result["success"]:
            raise RuntimeError(
                "Docking failed — check docking_engine output above."
            )

        docking_score = docking_result["best_score"]
        all_scores    = docking_result["all_scores"]
        best_pdbqt    = docking_result["best_pdbqt"]

        # Convert best pose PDBQT → SDF for ProLIF
        from meeko import PDBQTMolecule, RDKitMolCreate
        pdbqt_mol  = PDBQTMolecule.from_file(
            best_pdbqt, skip_typing=True
        )
        rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        if rdkit_mols and rdkit_mols[0] is not None:
            pose_mol = rdkit_mols[0]
            try:
                pose_mol = AllChem.AssignBondOrdersFromTemplate(
                    Chem.MolFromSmiles(drug_smiles), pose_mol
                )
            except Exception:
                pass
            w = Chem.SDWriter(best_sdf)
            w.write(pose_mol)
            w.close()
        print(f"  ✓ Best pose SDF: {best_sdf}")

    # ── 7. ProLIF ───────────────────────────────────────────
    print("\n[6/7] Running ProLIF interaction fingerprint...")
    interactions_df = run_prolif(clean_pdb, best_sdf)
    n = len(interactions_df)
    print(f"  ✓ {n} interactions found")

    if n > 0:
        print(f"\n  {'Residue':<14} {'Type':<16} Description")
        print(f"  {'─'*62}")
        for _, row in interactions_df.iterrows():
            print(f"  {row['residue']:<14} "
                  f"{row['interaction_type']:<16} "
                  f"{row['description']}")
            
    
    # Right after the ProLIF block in run_pipeline()


    # ── 8. Save results ─────────────────────────────────────
    print("\n[7/7] Saving results...")
    result = {
        "mode"           : mode,
        "drug"           : drug_name,
        "drug_smiles"    : drug_smiles,
        "protein"        : protein_name,
        "protein_file"   : os.path.basename(raw_protein_pdb),
        "cocrystal_code" : lig_code,
        "docking_score"  : docking_score,
        "score_units"    : "kcal/mol" if docking_score else "N/A (crystal pose)",
        "binding_site"   : {"center": center, "box_size": box_size,
                            "detected_by": method},
        "n_interactions" : n,
        "interactions"   : interactions_df.to_dict(orient="records"),
        "all_pose_scores": all_scores,
    }

    csv_path  = os.path.join(out_dir, f"{safe}_interactions.csv")
    json_path = os.path.join(out_dir, f"{safe}_results.json")
    interactions_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  ✓ CSV  : {csv_path}")
    print(f"  ✓ JSON : {json_path}")

    print(f"\n{'='*58}")
    print(f"PIPELINE COMPLETE  [Mode {mode}]")
    print(f"  Drug         : {drug_name}")
    score_str = (f"{docking_score:.3f} kcal/mol"
                 if docking_score else "crystal pose (no docking)")
    print(f"  Score        : {score_str}")
    print(f"  Interactions : {n}")
    print(f"  Results JSON : {json_path}")
    print(f"{'='*58}")

   
    

    return result


# ── CLI ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DockExplain — smart docking pipeline"
    )
    parser.add_argument("--protein",  required=True,
        help="Raw PDB file from RCSB")
    parser.add_argument("--drug",     required=True,
        help="Drug name (e.g. Erlotinib)")
    parser.add_argument("--smiles",   default=None,
        help="SMILES of drug to dock (optional if PDB is co-crystal)")
    parser.add_argument("--out",      default="../data/results/",
        help="Output directory")
    parser.add_argument("--exhaustiveness", type=int, default=8,
        help="Vina exhaustiveness (default 8)")
    args = parser.parse_args()

    run_pipeline(
        raw_protein_pdb = args.protein,
        drug_name       = args.drug,
        drug_smiles     = args.smiles,
        out_dir         = args.out,
        exhaustiveness  = args.exhaustiveness,
    )