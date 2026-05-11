# docking_engine.py
# Generalizable docking score computation.
# Supports: QuickVina 2, QuickVina-W, AutoDock Vina
# Automatically selects best available engine.
# Zero hardcoding — all parameters derived from inputs.
#
# Replaces the hardcoded adock/calculateGPX4DockingScore pattern.
# Works for any protein-drug pair.

import os
import re
import shutil
import subprocess
import tempfile
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import AllChem


# ════════════════════════════════════════════════════════════
# ENGINE DETECTION
# Finds the best available docking binary automatically.
# Priority: qvina2 > qvina-w > vina
# ════════════════════════════════════════════════════════════

def detect_docking_engine() -> tuple[str, str]:
    """
    Find the best available docking engine on this machine.
    Returns (engine_name, executable_path).

    Priority order:
      1. qvina2      — fastest, best for standard docking
      2. qvina-w     — good for wide search boxes
      3. vina        — AutoDock Vina, universal fallback
      4. vina_1.2.5  — alternate Vina naming

    All use identical config file format — fully interchangeable.
    """
    candidates = [
        ("qvina2",    "qvina2"),
        ("qvina2",    "qvina2.1"),
        ("qvina-w",   "qvina-w"),
        ("vina",      "vina"),
        ("vina",      "vina_1.2.5"),
        ("vina",      "autodock_vina"),
    ]

    for name, exe in candidates:
        path = shutil.which(exe)
        if path:
            print(f"  ✓ Docking engine: {name} ({path})")
            return name, path

    # Last resort: try vina Python package as subprocess
    try:
        result = subprocess.run(
            ["python", "-c", "from vina import Vina; print('ok')"],
            capture_output=True, text=True, timeout=5
        )
        if "ok" in result.stdout:
            return "vina_python", "python"
    except Exception:
        pass

    raise RuntimeError(
        "No docking engine found.\n"
        "Install one of:\n"
        "  conda install -c conda-forge autodock-vina\n"
        "  pip install vina\n"
        "  Download qvina2 from: https://qvina.github.io/"
    )


# ════════════════════════════════════════════════════════════
# LIGAND PREPARATION
# Converts SMILES → 3D → PDBQT using obabel (matching the
# original adock script's approach) with RDKit+Meeko fallback.
# ════════════════════════════════════════════════════════════

def smiles_to_pdbqt_obabel(smiles: str,
                             out_path: str,
                             timeout: int = 120) -> bool:
    """
    Convert SMILES → PDBQT using OpenBabel (same as original script).
    obabel handles 3D generation and hydrogens in one step.

    Returns True on success, False on failure.
    """
    cmd = (
        f'obabel -:"{smiles}" -O "{out_path}" '
        f'-h --gen3d --minimize --ff MMFF94 '
        f'> /dev/null 2>&1'
    )
    try:
        with subprocess.Popen(
            f"exec {cmd}",
            shell=True,
            start_new_session=True
        ) as proc:
            proc.wait(timeout=timeout)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True
    except subprocess.TimeoutExpired:
        print(f"  ⚠ obabel timed out for ligand preparation")
    except Exception as e:
        print(f"  ⚠ obabel failed: {e}")
    return False


def smiles_to_pdbqt_rdkit(smiles: str, out_path: str) -> bool:
    """
    Convert SMILES → PDBQT using RDKit + Meeko.
    Fallback when obabel is not available.
    """
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        p = AllChem.ETKDGv3()
        p.randomSeed = 42
        if AllChem.EmbedMolecule(mol, p) == -1:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94")

        preparator = MoleculePreparation()
        setups     = preparator.prepare(mol)
        pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(setups[0])

        if not is_ok:
            return False
        with open(out_path, "w") as f:
            f.write(pdbqt_str)
        return True

    except Exception as e:
        print(f"  ⚠ RDKit/Meeko ligand prep failed: {e}")
        return False


def prepare_ligand_pdbqt(smiles: str,
                          out_path: str,
                          timeout: int = 120) -> bool:
    """
    Prepare ligand PDBQT from SMILES.
    Tries obabel first (faster, matches original script),
    falls back to RDKit+Meeko.
    Returns True on success.
    """
    # Skip if already prepared
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  ✓ Ligand PDBQT already exists: {out_path}")
        return True

    # Try obabel first
    if shutil.which("obabel"):
        print(f"  Preparing ligand with obabel...")
        if smiles_to_pdbqt_obabel(smiles, out_path, timeout):
            print(f"  ✓ Ligand prepared (obabel): {out_path}")
            return True
        print(f"  ⚠ obabel failed, trying RDKit+Meeko...")

    # Fallback: RDKit + Meeko
    print(f"  Preparing ligand with RDKit+Meeko...")
    if smiles_to_pdbqt_rdkit(smiles, out_path):
        print(f"  ✓ Ligand prepared (RDKit+Meeko): {out_path}")
        return True

    print(f"  ✗ All ligand preparation methods failed")
    return False


# ════════════════════════════════════════════════════════════
# RECEPTOR PREPARATION
# Converts cleaned protein PDB → PDBQT using obabel.
# ════════════════════════════════════════════════════════════

def prepare_receptor_pdbqt(pdb_path: str,
                             out_path: str) -> bool:
    """
    Convert protein PDB → PDBQT for docking.
    Uses obabel with -xr (rigid receptor) flag.
    """
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  ✓ Receptor PDBQT already exists: {out_path}")
        return True

    if not shutil.which("obabel"):
        raise RuntimeError(
            "obabel not found. Install: "
            "conda install -c conda-forge openbabel"
        )

    result = subprocess.run(
        ["obabel", pdb_path, "-O", out_path, "-xr"],
        capture_output=True, text=True
    )
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  ✓ Receptor PDBQT ready: {out_path}")
        return True

    print(f"  ✗ obabel receptor conversion failed: {result.stderr}")
    return False


# ════════════════════════════════════════════════════════════
# CONFIG FILE BUILDER
# Creates the Vina/QVina config file — same format for all engines.
# Parameters come from our binding site detection, not hardcoded.
# ════════════════════════════════════════════════════════════

def write_config(receptor_pdbqt: str,
                 ligand_pdbqt  : str,
                 output_pdbqt  : str,
                 center        : list[float],
                 box_size      : list[float],
                 config_path   : str,
                 exhaustiveness: int = 32,
                 n_poses       : int = 10,
                 cpu           : int = 4,
                 seed          : int | None = None) -> str:
    """
    Write a Vina-compatible config file.
    All parameters are passed in — nothing hardcoded.
    Works for qvina2, qvina-w, and vina identically.
    """
    lines = [
        f"receptor = {receptor_pdbqt}",
        f"ligand   = {ligand_pdbqt}",
        f"out      = {output_pdbqt}",
        f"center_x = {center[0]:.4f}",
        f"center_y = {center[1]:.4f}",
        f"center_z = {center[2]:.4f}",
        f"size_x   = {box_size[0]:.1f}",
        f"size_y   = {box_size[1]:.1f}",
        f"size_z   = {box_size[2]:.1f}",
        f"exhaustiveness = {exhaustiveness}",
        f"num_modes      = {n_poses}",
        f"cpu            = {cpu}",
    ]
    if seed is not None:
        lines.append(f"seed = {seed}")

    config_text = "\n".join(lines) + "\n"
    with open(config_path, "w") as f:
        f.write(config_text)
    return config_text


# ════════════════════════════════════════════════════════════
# SCORE PARSER
# Reads the best docking score from any Vina-format output PDBQT.
# The REMARK VINA RESULT format is identical across all engines.
# ════════════════════════════════════════════════════════════

def parse_scores_from_pdbqt(output_pdbqt: str) -> list[float]:
    """
    Extract all binding scores from a Vina-format output PDBQT.
    Handles both:
      REMARK VINA RESULT:   -9.2  0.000  0.000  (AutoDock Vina)
      REMARK VINA RESULT:   -9.2                (QVina2)

    Returns list of scores sorted ascending (best first).
    Returns empty list if file not found or no scores.
    """
    scores = []
    try:
        with open(output_pdbqt) as f:
            for line in f:
                if "VINA RESULT" in line:
                    # Extract first number after the colon
                    nums = re.findall(r"[-+]?\d*\.?\d+", line.split(":")[-1])
                    if nums:
                        scores.append(float(nums[0]))
    except FileNotFoundError:
        pass
    return sorted(scores)   # best (most negative) first


# ════════════════════════════════════════════════════════════
# MAIN DOCKING FUNCTION
# Generalizable — works for any protein-drug pair.
# Matches the pattern of the original adock() function but
# with zero hardcoding and full integration into DockExplain.
# ════════════════════════════════════════════════════════════

def run_docking(
    receptor_pdb  : str,
    drug_smiles   : str,
    drug_name     : str,
    center        : list[float],
    box_size      : list[float],
    out_dir       : str,
    exhaustiveness: int = 32,
    n_poses       : int = 10,
    cpu           : int = 4,
    seed          : int | None = None,
    timeout       : int = 600,
    force_rerun   : bool = False,
) -> dict:
    """
    Run molecular docking for any protein-drug pair.

    Args:
        receptor_pdb   : path to cleaned protein PDB
        drug_smiles    : canonical SMILES of the drug
        drug_name      : human-readable name (for filenames)
        center         : [cx, cy, cz] binding site centroid in Å
                         (from crystal ligand or P2Rank)
        box_size       : [sx, sy, sz] search box dimensions in Å
        out_dir        : directory to save all docking files
        exhaustiveness : search thoroughness (8=default, 32=thorough)
        n_poses        : number of binding poses to generate
        cpu            : CPU cores to use
        seed           : random seed for reproducibility (None=random)
        timeout        : max seconds per docking run
        force_rerun    : if True, redo docking even if output exists

    Returns:
        dict with keys:
            best_score   : float (kcal/mol), best binding affinity
            all_scores   : list of floats, all pose scores
            poses_pdbqt  : path to output PDBQT with all poses
            best_pdbqt   : path to PDBQT with only best pose
            engine       : name of docking engine used
            config       : config file content (for reproducibility)
            success      : bool
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Clean filename from drug name ──────────────────────
    # Same approach as original script — replace problematic chars
    safe_name = (drug_name
                 .lower()
                 .replace(" ", "_")
                 .replace("(", "{")
                 .replace(")", "}")
                 .replace("#", "-")
                 .replace("/", "_"))

    # ── File paths (all derived from drug_name, not hardcoded) ──
    prot_base      = os.path.splitext(
        os.path.basename(receptor_pdb))[0]
    receptor_pdbqt = os.path.join(out_dir, f"{prot_base}_receptor.pdbqt")
    ligand_pdbqt   = os.path.join(out_dir, f"{safe_name}.pdbqt")
    output_pdbqt   = os.path.join(out_dir, f"{safe_name}_out.pdbqt")
    best_pdbqt     = os.path.join(out_dir, f"{safe_name}_best_pose.pdbqt")
    config_path    = os.path.join(out_dir, f"{safe_name}.conf")
    log_path       = os.path.join(out_dir, f"{safe_name}_log.txt")

    result = {
        "best_score" : None,
        "all_scores" : [],
        "poses_pdbqt": output_pdbqt,
        "best_pdbqt" : best_pdbqt,
        "engine"     : "",
        "config"     : "",
        "success"    : False,
    }

    # ── Detect engine ──────────────────────────────────────
    try:
        engine_name, engine_path = detect_docking_engine()
    except RuntimeError as e:
        print(f"  ✗ {e}")
        return result
    result["engine"] = engine_name

    # ── Prepare receptor ───────────────────────────────────
    print(f"\n  [1/3] Preparing receptor...")
    if not prepare_receptor_pdbqt(receptor_pdb, receptor_pdbqt):
        print(f"  ✗ Receptor preparation failed")
        return result

    # ── Prepare ligand ─────────────────────────────────────
    print(f"  [2/3] Preparing ligand ({drug_name})...")
    if not prepare_ligand_pdbqt(drug_smiles, ligand_pdbqt, timeout):
        print(f"  ✗ Ligand preparation failed")
        return result

    # ── Run docking ────────────────────────────────────────
    print(f"  [3/3] Running {engine_name} docking...")
    print(f"    Center   : {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f} Å")
    print(f"    Box      : {box_size[0]:.1f} × {box_size[1]:.1f} × {box_size[2]:.1f} Å")
    print(f"    Exhaust  : {exhaustiveness}")

    # Skip if output already exists and force_rerun is False
    if (os.path.exists(output_pdbqt)
            and os.path.getsize(output_pdbqt) > 0
            and not force_rerun):
        print(f"  ✓ Output already exists: {output_pdbqt}")
    else:
        if engine_name == "vina_python":
            # Use vina Python package directly
            _run_vina_python(
                receptor_pdbqt, ligand_pdbqt,
                output_pdbqt, center, box_size,
                exhaustiveness, n_poses
            )
        else:
            # Use binary (qvina2, qvina-w, or vina)
            config_text = write_config(
                receptor_pdbqt = receptor_pdbqt,
                ligand_pdbqt   = ligand_pdbqt,
                output_pdbqt   = output_pdbqt,
                center         = center,
                box_size       = box_size,
                config_path    = config_path,
                exhaustiveness = exhaustiveness,
                n_poses        = n_poses,
                cpu            = cpu,
                seed           = seed,
            )
            result["config"] = config_text

            cmd = (
                f"{engine_path} "
                f"--config {config_path} "
                f"--log {log_path} "
                f"> /dev/null 2>&1"
            )
            try:
                with subprocess.Popen(
                    f"exec {cmd}",
                    shell=True,
                    start_new_session=True
                ) as proc:
                    proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"  ⚠ Docking timed out after {timeout}s")
                try:
                    proc.kill()
                except Exception:
                    pass

    # ── Parse scores ───────────────────────────────────────
    scores = parse_scores_from_pdbqt(output_pdbqt)

    if not scores:
        print(f"  ✗ No scores found in output — docking may have failed")
        # Check log for clues
        if os.path.exists(log_path):
            with open(log_path) as lf:
                log_tail = lf.read()[-500:]
            if log_tail.strip():
                print(f"  Log tail:\n{log_tail}")
        return result

    best_score = scores[0]
    result["best_score"] = best_score
    result["all_scores"] = scores
    result["success"]    = True

    print(f"\n  ── Docking scores ({engine_name}) ─────────────────")
    print(f"  {'Pose':<6} {'Score (kcal/mol)':>18}  Strength")
    print(f"  {'─'*42}")
    for i, s in enumerate(scores, 1):
        strength = ("Excellent" if s < -9 else
                    "Good"      if s < -7 else
                    "Moderate"  if s < -5 else "Weak")
        tag = " ← best" if i == 1 else ""
        print(f"  {i:<6} {s:>18.3f}  {strength}{tag}")

    # ── Extract best pose ──────────────────────────────────
    _extract_best_pose(output_pdbqt, best_pdbqt)

    print(f"\n  ✓ Best score    : {best_score:.3f} kcal/mol")
    print(f"  ✓ All poses     : {output_pdbqt}")
    print(f"  ✓ Best pose     : {best_pdbqt}")
    print(f"  ✓ Engine used   : {engine_name}")

    return result


def _run_vina_python(receptor_pdbqt, ligand_pdbqt,
                     output_pdbqt, center, box_size,
                     exhaustiveness, n_poses):
    """Use the vina Python package when no binary is available."""
    from vina import Vina
    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(center=center, box_size=box_size)
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    poses = v.poses(n_poses=n_poses)
    with open(output_pdbqt, "w") as f:
        f.write(poses)


def _extract_best_pose(poses_pdbqt: str,
                       out_pdbqt   : str) -> str:
    """Extract MODEL 1 from a multi-model PDBQT output."""
    try:
        with open(poses_pdbqt) as f:
            content = f.read()
        lines    = content.split("\n")
        model1   = []
        inside   = False
        for line in lines:
            if line.strip() == "MODEL 1":
                inside = True
            if inside:
                model1.append(line)
            if inside and line.strip() == "ENDMDL":
                break
        if model1:
            model1.append("END")
            with open(out_pdbqt, "w") as f:
                f.write("\n".join(model1))
    except Exception as e:
        print(f"  ⚠ Best pose extraction failed: {e}")
    return out_pdbqt


# ════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER
# Drop-in replacement for calculateGPX4DockingScore() pattern.
# Fully generalizable — accepts any protein + drug.
# ════════════════════════════════════════════════════════════

def calculate_docking_score(
    receptor_pdb  : str,
    drug_smiles   : str,
    drug_name     : str,
    center        : list[float],
    box_size      : list[float] = [25.0, 25.0, 25.0],
    out_dir       : str = "./docking_output/",
    exhaustiveness: int = 32,
    cpu           : int = 4,
    seed          : int | None = None,
) -> float:
    """
    Single-function interface matching the original script's pattern.
    Returns the best docking score (kcal/mol) or 0.0 on failure.

    This is the generalizable equivalent of calculateGPX4DockingScore()
    — it works for any protein and any drug, not just GPX4.

    Args:
        receptor_pdb  : path to cleaned protein PDB
        drug_smiles   : SMILES string of the drug
        drug_name     : drug name (for file naming)
        center        : binding site center [x, y, z] in Å
                        → from crystal ligand centroid or P2Rank
        box_size      : search box size [x, y, z] in Å
        out_dir       : output directory
        exhaustiveness: Vina/QVina search depth
        cpu           : CPU cores
        seed          : random seed (None = random)

    Returns:
        float: best binding affinity in kcal/mol (negative = better)
               0.0 if docking failed
    """
    result = run_docking(
        receptor_pdb   = receptor_pdb,
        drug_smiles    = drug_smiles,
        drug_name      = drug_name,
        center         = center,
        box_size       = box_size,
        out_dir        = out_dir,
        exhaustiveness = exhaustiveness,
        cpu            = cpu,
        seed           = seed,
    )
    return result["best_score"] if result["success"] else 0.0